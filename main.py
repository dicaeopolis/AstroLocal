import os
import shutil
import subprocess
import uuid
import asyncio
import logging
import time
import math
from pathlib import Path
from io import BytesIO

import requests
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.data import download_file

# --- 1. 严谨的日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AstroServer")

# --- 2. 路径与常量配置 ---
BASE_DIR = Path(".")
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
# 真实数据源 URL (OpenNGC master branch)
NGC_CSV_URL = "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files/NGC.csv"
LOCAL_DB_PATH = BASE_DIR / "NGC.csv"

# 镜头针对性参数
SOLVE_OPTS = [
    "--downsample", "2",
    "--scale-units", "degwidth",
    "--scale-low", "3",
    "--scale-high", "30",
    "--no-plots", "--overwrite", "--cpulimit", "60", "--no-verify",
    "--crpix-center" # 强制使用图像中心作为参考点，提高宽视场精度
]

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory="templates")
JOBS = {}
DF_NGC = None # 全局 DataFrame 缓存

# --- 3. 严谨的数据加载模块 ---

def download_db_if_missing():
    """下载数据库，带进度提示，不使用 wget 命令，而是使用 python requests 以确保跨平台稳定性"""
    if LOCAL_DB_PATH.exists():
        return

    logger.info(f"正在下载 OpenNGC 数据库 ({NGC_CSV_URL})...")
    try:
        with requests.get(NGC_CSV_URL, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(LOCAL_DB_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    # 简单的进度打印，防止用户以为死机
                    if total_size > 0 and downloaded % (1024*1024) == 0:
                        logger.info(f"Downloading... {downloaded/1024/1024:.1f} MB")
        logger.info("数据库下载完成。")
    except Exception as e:
        logger.error(f"数据库下载失败: {e}")
        # 这里不应该有 fallback，如果下载失败就是失败，抛出异常让用户检查网络
        raise RuntimeError("无法下载 NGC.csv，请检查网络连接。")

def parse_coordinate_columns(df):
    """
    使用 Astropy 解析 HH:MM:SS 和 DD:MM:SS 格式。
    不使用简化的字符串分割，而是利用 SkyCoord 的解析能力，虽然慢几秒但绝对准确。
    """
    logger.info("正在利用 Astropy 解析坐标列 (这可能需要几秒钟)...")
    
    # 构建坐标字符串数组，Astropy 可以批量处理
    # 格式拼接: "HH:MM:SS.ss +DD:MM:SS.s"
    coords_str = df['RA'].astype(str) + " " + df['Dec'].astype(str)
    
    # 批量转换
    # unit=(u.hourangle, u.deg) 明确告诉 Astropy 第一列是时角，第二列是度数
    sc = SkyCoord(coords_str.to_numpy(), unit=(u.hourangle, u.deg))
    
    df['RA_deg'] = sc.ra.deg
    df['Dec_deg'] = sc.dec.deg
    return df

def load_catalog():
    """加载、清洗、转换 OpenNGC 数据"""
    global DF_NGC
    download_db_if_missing()

    logger.info("正在读取 CSV 文件...")
    # 1. 读取: 明确指定 sep=';'，处理 DtypeWarning
    df = pd.read_csv(LOCAL_DB_PATH, sep=';', low_memory=False)

    # 2. 清洗: 必须有坐标
    df = df.dropna(subset=['RA', 'Dec'])

    # 3. 坐标转换 (严谨版)
    df = parse_coordinate_columns(df)

    # 4. 数值列处理
    # V-Mag: 视星等，转为 float，无效值设为 99 (不可见)
    df['V-Mag'] = pd.to_numeric(df['V-Mag'], errors='coerce').fillna(99.0)
    
    # MajAx: 长轴尺寸 (角分)，转为 float，无效值设为 0
    df['MajAx'] = pd.to_numeric(df['MajAx'], errors='coerce').fillna(0.0)

    # M: 梅西耶编号，转为 Int 便于显示 M31 而不是 M31.0
    # 注意: 有些行 M 是空的
    df['M'] = pd.to_numeric(df['M'], errors='coerce') # 变成 float (含 NaN)

    DF_NGC = df
    logger.info(f"数据库就绪: {len(df)} 个天体 loaded.")

# --- 4. 严谨的绘图逻辑 ---

def get_pixel_scale(wcs_obj):
    scales = wcs_obj.proj_plane_pixel_scales()
    
    # --- PROBE START ---
    print(f"\n[DEBUG PROBE] Type of scales: {type(scales)}")
    if isinstance(scales, (list, tuple, np.ndarray)):
        print(f"[DEBUG PROBE] Length: {len(scales)}")
        if len(scales) > 0:
            print(f"[DEBUG PROBE] Element 0 type: {type(scales[0])}")
            print(f"[DEBUG PROBE] Element 0 value: {scales[0]}")
    else:
        print(f"[DEBUG PROBE] Value: {scales}")
    # --- PROBE END ---

    # 临时尝试最暴力的修复，希望能跑通，跑不通也能看到 Log
    try:
        # 尝试方案 A: 它是 Quantity 数组
        if hasattr(scales, 'to_value'):
            return np.mean(scales.to_value(u.deg))
            
        # 尝试方案 B: 它是包含 Quantity 的 list
        # 剥离单位
        raw = [s.to_value(u.deg) if hasattr(s, 'to_value') else s for s in scales]
        return np.mean(raw)
        
    except Exception as e:
        print(f"[DEBUG PROBE] Conversion failed: {e}")
        # 最后的保底: 假设它是纯数字
        return np.mean(scales)

def draw_annotations(wcs_path, img_path, out_path, task_id):
    """绘图主函数"""
    global DF_NGC
    if DF_NGC is None:
        logger.error("Database not loaded!")
        return 0

    # 1. 准备 WCS 和 图像
    with fits.open(wcs_path) as hdul:
        header = hdul[0].header
        wcs = WCS(header)
    
    with Image.open(img_path) as im:
        im = im.convert("RGBA")
        width, height = im.size
        draw = ImageDraw.Draw(im)
        
        # 2. 计算精确的视场包围盒 (Bounding Box)
        # 获取图像边缘的四个角和中心
        # pixel coordinates: (x, y)
        corners_pix = np.array([[0,0], [width, 0], [width, height], [0, height], [width/2, height/2]])
        corners_world = wcs.wcs_pix2world(corners_pix, 0) # RA, Dec in degrees
        
        ra_vals = corners_world[:, 0]
        dec_vals = corners_world[:, 1]
        
        min_ra, max_ra = np.min(ra_vals), np.max(ra_vals)
        min_dec, max_dec = np.min(dec_vals), np.max(dec_vals)
        
        # 计算视场对角线长度 (用于动态过滤星等)
        center_sc = SkyCoord(corners_world[4][0], corners_world[4][1], unit='deg')
        corner_sc = SkyCoord(corners_world[0][0], corners_world[0][1], unit='deg')
        fov_radius_deg = center_sc.separation(corner_sc).degree
        
        # 动态星等限制逻辑
        # 视场越大(radius大)，只标越亮的星(mag小)
        # 20度视场 -> mag < 6
        # 2度视场 -> mag < 12
        limit_mag = 14.0 - (fov_radius_deg / 3.0)
        limit_mag = np.clip(limit_mag, 2, 15.0) # 限制在合理区间
        
        logger.info(f"[{task_id}] FOV Radius: {fov_radius_deg:.2f}°, Mag Limit: {limit_mag:.2f}")

        # 3. Pandas 筛选 (Vectorized filtering)
        # 处理 RA 跨越 0/360 度的边缘情况 (严谨处理)
        # 如果 max_ra - min_ra > 180，说明跨越了 0 度 (例如 359度 到 1度)
        df = DF_NGC
        dec_mask = (df['Dec_deg'] >= min_dec - 1.0) & (df['Dec_deg'] <= max_dec + 1.0)
        mag_mask = (df['V-Mag'] <= limit_mag)
        
        if max_ra - min_ra > 180:
            # 跨越 0 度：保留 (RA > max_ra) OR (RA < min_ra) *这里的逻辑反过来*
            # 实际上更简单的做法是：不用简单的 min/max，而是用 separation
            # 但为了性能，我们先做粗筛，再做精确计算
            # 粗筛: 取所有 dec 符合的
            candidates = df[dec_mask & mag_mask].copy()
        else:
            # 正常情况
            ra_mask = (df['RA_deg'] >= min_ra - 1.0) & (df['RA_deg'] <= max_ra + 1.0)
            candidates = df[dec_mask & ra_mask & mag_mask].copy()

        # 4. 精确坐标转换与绘制
        count = 0
        pixel_scale = get_pixel_scale(wcs) # deg/pix
        
        # 字体加载
        font_size = max(12, int(height / 50))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # 批量转换 candidate 坐标 (比 iterrows 快)
        if len(candidates) > 0:
            world_coords = np.vstack((candidates['RA_deg'], candidates['Dec_deg'])).T
            try:
                pix_coords = wcs.wcs_world2pix(world_coords, 0) # 得到 (x, y) 数组
            except Exception as e:
                logger.error(f"WCS conversion failed: {e}")
                return 0

            # 遍历绘制
            for i, (idx, row) in enumerate(candidates.iterrows()):
                px, py = pix_coords[i]
                
                # 检查是否在图像范围内 (加一点缓冲)
                if -50 <= px <= width+50 and -50 <= py <= height+50:
                    name = row['Name']
                    maj_ax = row['MajAx'] # 角分
                    m_num = row['M']
                    
                    # 优先显示 Messier 编号
                    if pd.notna(m_num):
                        display_name = f"M{int(m_num)}"
                    else:
                        display_name = name

                    # 计算绘制圆圈的半径 (像素)
                    # MajAx 是直径(角分) -> 半径(角分) -> 半径(度) -> 像素
                    if maj_ax > 0:
                        radius_deg = (maj_ax / 2.0) / 60.0
                        radius_pix = radius_deg / pixel_scale
                    else:
                        # 如果没有大小数据，给一个默认的可视大小 (比如 20px)
                        radius_pix = 20

                    # 最小半径限制，防止太小看不见
                    radius_pix = max(radius_pix, 15)

                    # 绘图
                    # 黄色虚线圈 (用实线代替，PIL原生不支持虚线)
                    bbox = [px - radius_pix, py - radius_pix, px + radius_pix, py + radius_pix]
                    draw.ellipse(bbox, outline="yellow", width=2)
                    
                    # 文字标注 (带描边以防背景太亮)
                    text_pos = (px + radius_pix/1.4, py + radius_pix/1.4)
                    draw.text(text_pos, display_name, fill="yellow", font=font, stroke_width=2, stroke_fill="black")
                    
                    count += 1

        im.save(out_path, "PNG")
        return count

# --- 5. 任务与API ---

async def processing_pipeline(task_id: str, file_path: Path):
    job = JOBS[task_id]
    temp_files = [] # 用于清理
    
    try:
        # Step 1: Astrometry 盲解
        job["status"] = "solving"
        job["log"] = "正在运行 Astrometry 引擎..."
        
        cmd = ["solve-field", str(file_path)] + SOLVE_OPTS
        start_t = time.time()
        
        # 实时打印命令以便调试
        logger.info(f"EXEC: {' '.join(cmd)}")
        
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            err = stderr.decode()
            if "Time limit reached" in err:
                raise TimeoutError("解析超时，请增加 cpulimit 或检查星点")
            else:
                # 记录最后几行错误
                raise RuntimeError(f"解析失败: {err[-300:]}")
        
        logger.info(f"[{task_id}] Solve OK. Time: {time.time()-start_t:.2f}s")
        
        wcs_file = file_path.with_suffix(".wcs")
        if not wcs_file.exists():
            raise FileNotFoundError("WCS 文件未生成，虽然返回码为0")
        
        temp_files.append(wcs_file)
        # 收集 solve-field 生成的所有杂文件 (.axy, .corr, .match 等)
        temp_files.extend(file_path.parent.glob(f"{file_path.stem}.*"))

        # Step 2: 生成星座连线底图 (使用 plot-constellations C工具)
        job["status"] = "plotting_lines"
        job["log"] = "绘制星座连线 (底层)..."
        
        temp_ppm = file_path.with_suffix(".ppm")
        temp_files.append(temp_ppm)
        # 必须转 PPM，PIL 转换
        await asyncio.to_thread(lambda: Image.open(file_path).convert("RGB").save(temp_ppm))
        
        temp_base_png = file_path.parent / f"{task_id}_base.png"
        temp_files.append(temp_base_png)
        
        # 只画线 (-C) 和 边界 (-B) 和 星座名 (-N, 如 Cyg)
        plot_cmd = [
            "plot-constellations", "-w", str(wcs_file), "-i", str(temp_ppm), "-o", str(temp_base_png),
            "-C", "-B", "-N"
        ]
        proc_plot = await asyncio.create_subprocess_exec(*plot_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc_plot.communicate() # 忽略返回值，因为可能是 255
        
        # Step 3: Python 标注深空天体
        job["status"] = "annotating"
        job["log"] = "查询 OpenNGC 并标注..."
        
        final_png = STATIC_DIR / f"{task_id}_annotated.png"
        
        # 如果 plot-constellations 失败没生成图，就回退到原图
        source_img = temp_base_png if temp_base_png.exists() else file_path
        
        count = await asyncio.to_thread(draw_annotations, wcs_file, source_img, final_png, task_id)
        
        job["status"] = "done"
        job["log"] = f"完成! 标注了 {count} 个天体"
        job["result_url"] = f"/static/{final_png.name}"

    except Exception as e:
        logger.exception(f"[{task_id}] Pipeline Error")
        job["status"] = "failed"
        job["log"] = f"错误: {str(e)}"
    finally:
        # 清理垃圾
        for p in temp_files:
            if p.exists() and p != file_path: # 不要删原图
                try: os.remove(p)
                except: pass

@app.on_event("startup")
async def startup():
    UPLOADS_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(exist_ok=True)
    load_catalog()

@app.post("/upload")
async def upload(bg: BackgroundTasks, file: UploadFile = File(...)):
    tid = str(uuid.uuid4())
    fpath = UPLOADS_DIR / f"{tid}{Path(file.filename).suffix}"
    with open(fpath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    JOBS[tid] = {"status": "pending", "log": "已上传", "result": None}
    bg.add_task(processing_pipeline, tid, fpath)
    
    return HTMLResponse(f"""
        <div hx-get='/status/{tid}' hx-trigger='load delay:1s' hx-swap='outerHTML'>
            <div class='bg-gray-800 p-4 border border-blue-500 rounded animate-pulse text-blue-300'>
                等待处理...
            </div>
        </div>
    """)

@app.get("/status/{tid}")
async def status(tid: str):
    job = JOBS.get(tid)
    if not job: return HTMLResponse("404 Task Not Found")
    
    status = job['status']
    log = job['log']
    
    if status == "done":
        return HTMLResponse(f"""
            <div class='bg-gray-900 border border-green-500 p-4 rounded'>
                <div class='text-green-400 font-bold text-lg mb-2'>{log}</div>
                <img src='{job['result_url']}' class='w-full rounded shadow-2xl'>
                <a href='/' class='block w-full bg-blue-600 text-white text-center py-2 mt-4 rounded hover:bg-blue-700'>解析下一张</a>
            </div>
        """)
    elif status == "failed":
        return HTMLResponse(f"""
            <div class='bg-red-900/50 border border-red-500 p-4 rounded text-red-200'>
                <p class='font-bold'>FAILED</p>
                <p class='font-mono text-sm'>{log}</p>
                <a href='/' class='underline mt-2 inline-block'>重试</a>
            </div>
        """)
    
    return HTMLResponse(f"""
        <div hx-get='/status/{tid}' hx-trigger='load delay:1s' hx-swap='outerHTML'>
            <div class='bg-gray-800 p-4 border border-blue-500 rounded'>
                <div class='flex items-center gap-3'>
                    <div class='w-4 h-4 bg-blue-400 rounded-full animate-ping'></div>
                    <span class='text-blue-100 font-mono'>{log}</span>
                </div>
            </div>
        </div>
    """)

@app.get("/", response_class=HTMLResponse)
async def index(req: Request):
    return templates.TemplateResponse("index.html", {"request": req})