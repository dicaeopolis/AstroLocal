import os
import shutil
import uuid
import asyncio
import logging
import math
import time
from pathlib import Path
from datetime import datetime

# --- 第三方库：网络与数据处理 ---
import requests
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ExifTags

# --- 第三方库：Web 框架 ---
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# --- 第三方库：天文学计算 ---
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
from astropy.time import Time
import astropy.units as u

# --- 第三方库：时区处理 ---
from timezonefinder import TimezoneFinder
import pytz

# ==========================================
# 1. 配置与全局初始化
# ==========================================

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("AstroServer")

# 路径配置
BASE_DIR = Path(".")
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
LOCAL_DB_PATH = BASE_DIR / "NGC.csv"

# 外部资源链接
NGC_CSV_URL = "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files/NGC.csv"

# 字体回退路径列表
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "C:\\Windows\\Fonts\\arial.ttf",
]

# Astrometry.net 求解参数
SOLVE_OPTS = [
    "--downsample", "2",
    "--scale-units", "degwidth",
    "--scale-low", "3",
    "--scale-high", "30",
    "--no-plots",
    "--overwrite",
    "--cpulimit", "60",
    "--no-verify",
    "--crpix-center"
]

# 全局对象初始化
tf_engine = TimezoneFinder(in_memory=True)
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory="templates")

# 全局状态存储
JOBS = {}      # 存储任务状态
DF_NGC = None  # 存储加载的 NGC 星表数据


# ==========================================
# 2. 基础辅助功能
# ==========================================

def download_db_if_missing():
    """检查并下载 NGC 数据库文件"""
    if LOCAL_DB_PATH.exists():
        return
    
    logger.info("Downloading NGC DB...")
    try:
        with requests.get(NGC_CSV_URL, stream=True) as r:
            with open(LOCAL_DB_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception:
        raise RuntimeError("DB download failed")


def load_catalog():
    """加载并预处理 NGC 星表数据到内存"""
    global DF_NGC
    download_db_if_missing()
    
    # 读取 CSV，处理缺失坐标
    df = pd.read_csv(LOCAL_DB_PATH, sep=';', low_memory=False)
    df = df.dropna(subset=['RA', 'Dec'])
    
    # 转换坐标系统
    coords = df['RA'].astype(str) + " " + df['Dec'].astype(str)
    sc = SkyCoord(coords.to_numpy(), unit=(u.hourangle, u.deg))
    df['RA_deg'] = sc.ra.deg
    df['Dec_deg'] = sc.dec.deg
    
    # 处理数值列，填充默认值
    df['V-Mag'] = pd.to_numeric(df['V-Mag'], errors='coerce').fillna(99.0)
    df['MajAx'] = pd.to_numeric(df['MajAx'], errors='coerce').fillna(0.0)
    df['M'] = pd.to_numeric(df['M'], errors='coerce')
    
    DF_NGC = df
    logger.info(f"Catalog loaded: {len(df)} objects")


def get_utc_time(img_path, lat, lon):
    """
    尝试从图片 EXIF 获取拍摄时间并转换为 UTC。
    如果 EXIF 缺失，回退到当前时间。
    """
    local_dt = datetime.now()
    is_fallback = True
    
    # 1. 尝试读取 EXIF 时间
    try:
        with Image.open(img_path) as img:
            exif = img._getexif()
            if exif:
                # 尝试常见的日期时间 Tag ID
                for tag in [36867, 36868, 306]:
                    d = exif.get(tag)
                    if d:
                        try:
                            local_dt = datetime.strptime(d, "%Y:%m:%d %H:%M:%S")
                            is_fallback = False
                            break
                        except ValueError:
                            continue
    except Exception:
        pass

    # 2. 根据经纬度确定时区并转换为 UTC
    try:
        tz_str = tf_engine.timezone_at(lng=lon, lat=lat)
        if tz_str:
            local_tz = pytz.timezone(tz_str)
            # 假设 EXIF 时间为当地时间，转换为 UTC
            utc_dt = local_tz.localize(local_dt).astimezone(pytz.utc)
        else:
            # 无法确定时区，默认视为 UTC
            utc_dt = local_dt.replace(tzinfo=pytz.utc)
    except Exception:
        utc_dt = local_dt.replace(tzinfo=pytz.utc)
        
    return Time(utc_dt), is_fallback


def get_font(size):
    """获取系统中可用的字体，回退到默认字体"""
    for p in FONT_PATHS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ==========================================
# 3. 核心算法：网格计算与绘制
# ==========================================

def calculate_dynamic_step(fov_deg, is_ra=False):
    """
    根据视场角 (FOV) 自动计算网格线的最佳步长。
    
    Args:
        fov_deg (float): 视场角度.
        is_ra (bool): 是否计算赤经 (RA) 步长.
        
    Returns:
        float: 最佳步长（度）.
    """
    target_step = fov_deg / 4.0
    if target_step <= 0:
        target_step = 1.0 
    
    if is_ra:
        # 赤经转换：1小时 = 15度
        target_min = target_step * 60.0 / 15.0 
        
        if target_min >= 60: 
            step_h = round(target_min / 60)
            step_h = max(1, step_h)
            return step_h * 15.0
        else:
            candidates_min = [1, 2, 5, 10, 15, 20, 30]
            best = min(candidates_min, key=lambda x: abs(x - target_min))
            return best * 15.0 / 60.0
    else:
        # 赤纬步长：按 1, 2, 5, 10 序列对齐
        exponent = math.floor(math.log10(target_step))
        fraction = target_step / (10**exponent)
        
        if fraction < 1.5:
            base = 1
        elif fraction < 3.5:
            base = 2
        elif fraction < 7.5:
            base = 5
        else:
            base = 10
        
        return base * (10**exponent)


def format_ra_label(deg):
    """格式化赤经标签 (HHh MMm)"""
    a = Angle(deg, u.degree)
    h = int(a.hour)
    m = int(round((a.hour * 60) % 60))
    if m == 60: 
        h += 1
        m = 0
    return f"{h}h {m}m" if m != 0 else f"{h}h"


def format_dec_label(deg):
    """格式化赤纬标签 (+DD° MM')"""
    a = Angle(deg, u.degree)
    d = int(a.deg)
    m = abs(int(round((a.deg - d) * 60)))
    if m == 60:
        d += 1 if d >= 0 else -1
        m = 0
    sign = "+" if d >= 0 else "-"
    return f"{sign}{abs(d)}° {m}'" if m != 0 else f"{sign}{abs(d)}°"


def draw_smart_label(draw, pos, text, font, img_w, img_h, color, align_type='center'):
    """
    使用 anchor 参数绘制贴边文本，防止标签溢出图像。
    """
    x, y = pos
    margin = 4
    stroke_fill = (0, 0, 0, 255)
    
    anchor = 'mm'  # 默认居中
    tx, ty = x, y

    # 根据对齐类型调整锚点和坐标
    if align_type == 'left':
        anchor = 'lm'
        tx += margin
    elif align_type == 'right':
        anchor = 'rm'
        tx -= margin
    elif align_type == 'top':
        anchor = 'ma'
        ty += margin
    elif align_type == 'bottom':
        anchor = 'md'
        ty -= margin
    else:
        anchor = 'md'
        ty -= 2

    draw.text(
        (tx, ty), text, font=font, fill=color,
        anchor=anchor, stroke_width=2, stroke_fill=stroke_fill
    )


def draw_projected_line(draw, wcs, world_coords, width, height, color, label_func=None, font=None, text_color=None, is_dec=False):
    """
    将天球坐标投影到像素平面并绘制线条。
    包含断点检测、边界穿越检测以及智能标签放置逻辑。
    """
    try:
        pix_coords = wcs.wcs_world2pix(world_coords, 0)
    except Exception:
        return

    # --- 1. 线条分段 (处理断点) ---
    segments = []
    current_segment = []
    
    # 允许的绘图缓冲区，确保线条在边界处连贯
    draw_padding = 100 
    
    for px, py in pix_coords:
        if np.isnan(px) or np.isnan(py):
            if current_segment:
                segments.append(current_segment)
            current_segment = []
            continue
            
        in_bounds = -draw_padding < px < width + draw_padding and -draw_padding < py < height + draw_padding
        
        if in_bounds:
            if current_segment:
                last_x, last_y = current_segment[-1]
                # 断点检测：如果相邻点距离过大，视为断裂（如 RA 0/360 跳变）
                dist = math.hypot(px - last_x, py - last_y)
                if dist > max(width, height) / 4: 
                    segments.append(current_segment)
                    current_segment = []
            current_segment.append((px, py))
        else:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
    
    if current_segment:
        segments.append(current_segment)
    
    label_candidates = [] # 格式: (x, y, align_type)

    # --- 2. 绘制与标签位置计算 ---
    for seg in segments:
        if len(seg) < 2:
            continue
        
        # 绘制线条
        draw.line(seg, fill=color, width=1)
        
        if not label_func:
            continue

        # --- 核心逻辑：线段跨越检测 (Edge Crossing) ---
        has_horizontal_crossing = False 
        
        for i in range(len(seg) - 1):
            p1 = seg[i]
            p2 = seg[i+1]
            x1, y1 = p1
            x2, y2 = p2
            
            # 左边缘检测 (x=0)
            if (x1 < 0 and x2 >= 0) or (x1 >= 0 and x2 < 0):
                if x2 != x1:
                    y_cross = y1 + (y2 - y1) * (0 - x1) / (x2 - x1)
                    if 0 <= y_cross <= height:
                        label_candidates.append((0, y_cross, 'left'))
                        has_horizontal_crossing = True

            # 右边缘检测 (x=width)
            if (x1 <= width and x2 > width) or (x1 > width and x2 <= width):
                if x2 != x1:
                    y_cross = y1 + (y2 - y1) * (width - x1) / (x2 - x1)
                    if 0 <= y_cross <= height:
                        label_candidates.append((width, y_cross, 'right'))
                        has_horizontal_crossing = True

            # 上边缘检测 (y=0)
            if (y1 < 0 and y2 >= 0) or (y1 >= 0 and y2 < 0):
                if y2 != y1:
                    x_cross = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
                    if 0 <= x_cross <= width:
                        label_candidates.append((x_cross, 0, 'top'))

            # 下边缘检测 (y=height)
            if (y1 <= height and y2 > height) or (y1 > height and y2 <= height):
                if y2 != y1:
                    x_cross = x1 + (x2 - x1) * (height - y1) / (y2 - y1)
                    if 0 <= x_cross <= width:
                        label_candidates.append((x_cross, height, 'bottom'))

        # --- 特殊情况：纬线中轴标注 ---
        # 如果是 Dec 线且未穿过左右边缘（如极区圆环），则在图像垂直中轴线附近标注
        if is_dec and not has_horizontal_crossing:
            visible_points = [p for p in seg if 0 <= p[0] <= width and 0 <= p[1] <= height]
            
            if visible_points:
                # 找离 x 中心最近的点
                center_cand = min(visible_points, key=lambda p: abs(p[0] - width/2))
                label_candidates.append((center_cand[0], center_cand[1], 'center'))

    # --- 3. 执行标注 (简单的空间去重) ---
    drawn_labels = []
    # 优先级：左右边缘 > 上下边缘 > 中间
    priority_map = {'left': 0, 'right': 0, 'top': 1, 'bottom': 1, 'center': 2}
    label_candidates.sort(key=lambda item: priority_map.get(item[2], 2))

    for cx, cy, calign in label_candidates:
        # 避让检查
        is_cluttered = False
        for dx, dy in drawn_labels:
            if math.hypot(cx-dx, cy-dy) < 60:  # 60px 避让半径
                is_cluttered = True
                break
        
        if not is_cluttered:
            draw_smart_label(draw, (cx, cy), label_func, font, width, height, text_color, align_type=calign)
            drawn_labels.append((cx, cy))


def draw_celestial_grid(draw, wcs, width, height):
    """
    主网格绘制逻辑：采样视场，计算步长，生成经纬线路径。
    """
    # --- 1. 视场探测与步长计算 ---
    # 在图像上均匀采样点以估算视场范围
    sample_grid_x = np.linspace(0, width, 10)
    sample_grid_y = np.linspace(0, height, 10)
    xx, yy = np.meshgrid(sample_grid_x, sample_grid_y)
    sample_pix = np.vstack([xx.ravel(), yy.ravel()]).T
    
    try:
        sample_world = wcs.wcs_pix2world(sample_pix, 0)
    except Exception:
        return
        
    ra_samples = sample_world[:, 0]
    dec_samples = sample_world[:, 1]
    
    # 计算视场中心和像素比例
    c_world = wcs.pixel_to_world(width/2, height/2)
    center_ra = c_world.ra.deg
    
    e_world = wcs.pixel_to_world(width/2 + 10, height/2)
    pix_scale = c_world.separation(e_world).deg / 10.0
    fov_deg = pix_scale * max(width, height)
    
    # 计算动态步长
    step_ra = calculate_dynamic_step(fov_deg, is_ra=True)
    step_dec = calculate_dynamic_step(fov_deg, is_ra=False)
    
    # 极区步长修正
    max_abs_dec = np.max(np.abs(dec_samples))
    if max_abs_dec > 80:
        step_ra = max(step_ra, 15.0)

    # --- 2. 统一计算绘制范围 ---
    
    # Dec 范围 (纬线)
    min_dec, max_dec = np.min(dec_samples), np.max(dec_samples)
    d_start = math.floor(min_dec / step_dec) * step_dec
    d_end = math.ceil(max_dec / step_dec) * step_dec
    dec_lines = np.arange(d_start, d_end + step_dec + 0.001, step_dec)
    
    # RA 范围 (经线)
    # 将 RA 映射到以中心点为基准的相对范围 [-180, 180] 以处理 0/360 边界
    ra_diffs = (ra_samples - center_ra + 180) % 360 - 180
    min_diff = np.min(ra_diffs)
    max_diff = np.max(ra_diffs)
    
    ra_lines_raw = []
    search_min = min_diff - step_ra
    search_max = max_diff + step_ra
    
    # 在相对范围内搜索符合步长的 RA
    current_rel = math.floor(search_min / step_ra) * step_ra
    while current_rel <= search_max:
        abs_ra = (center_ra + current_rel) % 360
        ra_lines_raw.append(abs_ra)
        current_rel += step_ra
    ra_lines = np.unique(ra_lines_raw)

    # --- 3. 绘制参数准备 ---
    font_size = max(11, int(height / 55))
    font = get_font(font_size)
    grid_color = (255, 255, 255, 60)
    text_color = (220, 220, 255, 240)

    # --- 4. 绘制循环 ---

    # A. 绘制经线 (Meridians)
    for ra_val in ra_lines:
        norm_ra = ra_val % 360
        # 经线总是从 -90 到 90
        path_dec = np.linspace(-90, 90, 300)
        path_ra = np.full_like(path_dec, norm_ra)
        
        world_coords = np.vstack([path_ra, path_dec]).T
        draw_projected_line(
            draw, wcs, world_coords, width, height, grid_color, 
            label_func=format_ra_label(norm_ra), font=font, 
            text_color=text_color, is_dec=False
        )

    # B. 绘制纬线 (Parallels)
    for dec_val in dec_lines:
        if dec_val <= -90 or dec_val >= 90:
            continue
        
        span = max_diff - min_diff
        
        # 如果视场极大或跨度超过 360，画全圆
        if span > 120 or span * 3 > 360:
             path_ra = np.linspace(0, 360, 400)
        else:
            # 普通视场：左右扩展一定比例
            pad = max(span, 10.0)
            start_rel = min_diff - pad
            end_rel = max_diff + pad
            rel_path = np.linspace(start_rel, end_rel, 400)
            path_ra = center_ra + rel_path
            
        path_dec = np.full_like(path_ra, dec_val)
        # 取模以确保数据在 0-360 范围内，draw_projected_line 会处理断点
        world_coords = np.vstack([path_ra % 360, path_dec]).T
        
        draw_projected_line(
            draw, wcs, world_coords, width, height, grid_color,
            label_func=format_dec_label(dec_val), font=font, 
            text_color=text_color, is_dec=True
        )
        
    # C. 绘制中心十字丝
    cx, cy = width/2, height/2
    len_cross = min(width, height) * 0.03
    draw.line([(cx - len_cross, cy), (cx + len_cross, cy)], fill=(255, 50, 50, 180), width=2)
    draw.line([(cx, cy - len_cross), (cx, cy + len_cross)], fill=(255, 50, 50, 180), width=2)


def draw_final_output(wcs_path, img_path, out_path, task_id):
    """
    主绘图入口：加载资源，调用网格绘制，叠加 NGC 标注。
    """
    if DF_NGC is None:
        return 0, None

    # 1. 加载 WCS 信息
    with fits.open(wcs_path) as hdul:
        header = hdul[0].header
        wcs = WCS(header)
    
    # 2. 加载图像并修复 WCS 尺寸
    with Image.open(img_path) as im:
        im = im.convert("RGBA")
        w, h = im.size
        wcs.pixel_shape = (w, h)

        draw = ImageDraw.Draw(im)
        
        # 3. 绘制高级网格与标注
        draw_celestial_grid(draw, wcs, w, h)
        
        # 4. NGC 天体标注
        # 计算覆盖范围 (Footprint)
        fp = wcs.calc_footprint()
        if fp is None:
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
            fp = wcs.wcs_pix2world(corners, 0)

        dec_min, dec_max = np.min(fp[:, 1]), np.max(fp[:, 1])
        c_sky = wcs.pixel_to_world(w/2, h/2)
        corner = wcs.pixel_to_world(0, 0)
        radius = c_sky.separation(corner).deg
        
        # 根据视场大小动态计算极限星等
        limit_mag = np.clip(14.0 - (radius / 2.5), 3.5, 12.0)
        
        # 筛选候选天体
        df = DF_NGC
        cands = df[
            (df['Dec_deg'] >= dec_min - 2) & 
            (df['Dec_deg'] <= dec_max + 2) & 
            (df['V-Mag'] <= limit_mag)
        ]
        
        count = 0
        font = get_font(max(11, int(h/55)))
        
        if len(cands) > 0:
            pts_world = np.vstack((cands['RA_deg'], cands['Dec_deg'])).T
            try:
                pts_pix = wcs.wcs_world2pix(pts_world, 0)
            except Exception:
                pts_pix = []
            
            for i, (_, row) in enumerate(cands.iterrows()):
                if i >= len(pts_pix):
                    break
                px, py = pts_pix[i]
                if np.isnan(px) or not (0 <= px <= w and 0 <= py <= h):
                    continue
                
                # 确定显示名称和大小
                name = f"M{int(row['M'])}" if pd.notna(row['M']) else row['Name']
                maj = row['MajAx']
                r_pix = 15
                if maj > 0:
                    r_deg = (maj/2) / 60
                    pix_scale = c_sky.separation(wcs.pixel_to_world(w/2+1, h/2)).deg
                    if pix_scale > 0:
                        r_pix = r_deg / pix_scale
                
                r_pix = max(r_pix, 8)
                bbox = [px - r_pix, py - r_pix, px + r_pix, py + r_pix]
                
                # 绘制标记
                draw.ellipse(bbox, outline="yellow", width=1)
                draw.text(
                    (px + r_pix*0.8, py + r_pix*0.8), name,
                    font=font, fill="yellow", stroke_width=2, stroke_fill="black"
                )
                count += 1

        im.save(out_path, "PNG")
        return count, wcs


# ==========================================
# 4. 任务管道与路由
# ==========================================

async def processing_pipeline(task_id, fpath, lat, lon):
    """
    异步任务处理管道：时间推断 -> 盲解 -> 星座连线 -> 网格标注 -> 数据计算
    """
    job = JOBS[task_id]
    temps = []
    
    try:
        utc_time, is_fb = get_utc_time(fpath, lat, lon)
        job["obs_time_utc"] = utc_time.to_datetime()

        # 1. Astrometry 解析
        job.update({"status": "solving", "log": "Astrometry 解析中..."})
        cmd = ["solve-field", str(fpath)] + SOLVE_OPTS
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        
        wcs_f = fpath.with_suffix(".wcs")
        if not wcs_f.exists():
            raise RuntimeError("解析失败: 星点不足")
        
        # 记录临时文件以便后续清理
        temps.append(wcs_f)
        temps.extend(fpath.parent.glob(f"{fpath.stem}.*"))

        # 2. 绘制星座连线 (plot-constellations)
        job.update({"status": "plotting", "log": "绘制连线..."})
        ppm = fpath.with_suffix(".ppm")
        base_png = fpath.parent / f"{task_id}_base.png"
        temps.extend([ppm, base_png])
        
        await asyncio.to_thread(lambda: Image.open(fpath).convert("RGB").save(ppm))
        plot_cmd = [
            "plot-constellations", "-w", str(wcs_f), "-i", str(ppm),
            "-o", str(base_png), "-C", "-B", "-N"
        ]
        proc_plot = await asyncio.create_subprocess_exec(
            *plot_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc_plot.communicate()

        # 3. 标注网格与 NGC (Annotate)
        job.update({"status": "annotating", "log": "标注天体与网格..."})
        final_png = STATIC_DIR / f"{task_id}_annotated.png"
        src = base_png if base_png.exists() else fpath
        
        cnt, wcs_obj = await asyncio.to_thread(draw_final_output, wcs_f, src, final_png, task_id)

        # 4. 数据计算 (FOV, AltAz 等)
        w, h = wcs_obj.pixel_shape if wcs_obj.pixel_shape else (0, 0)
        if w == 0: 
             with Image.open(fpath) as im:
                 w, h = im.size
             
        c_sky = wcs_obj.pixel_to_world(w/2, h/2)
        
        # 计算视场大小 (Great Circle)
        fov_w = wcs_obj.pixel_to_world(0, h/2).separation(wcs_obj.pixel_to_world(w, h/2)).deg
        fov_h = wcs_obj.pixel_to_world(w/2, 0).separation(wcs_obj.pixel_to_world(w/2, h)).deg
        
        # 计算地平坐标
        loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=10*u.m)
        altaz = c_sky.transform_to(AltAz(obstime=utc_time, location=loc))

        job["data"] = {
            "fov": f"{fov_w:.1f}° x {fov_h:.1f}°",
            "ra": f"{c_sky.ra.to_string(unit=u.hour, sep='h', precision=1)}", 
            "dec": f"{c_sky.dec.to_string(unit=u.deg, sep='°', precision=1)}",
            "az": f"{altaz.az.deg:.1f}°",
            "alt": f"{altaz.alt.deg:.1f}°",
            "time": utc_time.strftime("%Y-%m-%d %H:%M UTC") + (" (Est.)" if is_fb else "")
        }
        job.update({
            "status": "done",
            "log": f"完成: {cnt} objects",
            "result_url": f"/static/{final_png.name}"
        })

    except Exception as e:
        logger.exception("Task Failed")
        job.update({"status": "failed", "log": str(e)})
    finally:
        # 清理临时文件
        for p in temps:
            if p.exists() and p != fpath: 
                try:
                    os.remove(p)
                except Exception:
                    pass


@app.on_event("startup")
async def startup():
    UPLOADS_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(exist_ok=True)
    load_catalog()


@app.post("/upload")
async def upload(bg: BackgroundTasks, file: UploadFile = File(...), lat: str = Form(None), lon: str = Form(None)):
    if not lat or not lon:
        return HTMLResponse("<div class='text-red-400'>Error: GPS Required</div>")
    
    tid = str(uuid.uuid4())
    fp = UPLOADS_DIR / f"{tid}{Path(file.filename).suffix}"
    
    with open(fp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    JOBS[tid] = {"status": "pending", "log": "Queued"}
    bg.add_task(processing_pipeline, tid, fp, float(lat), float(lon))
    
    return HTMLResponse(
        f"<div hx-get='/status/{tid}' hx-trigger='load delay:1s' hx-swap='outerHTML' class='text-blue-300 animate-pulse'>Starting...</div>"
    )


@app.get("/status/{tid}")
async def status(tid: str):
    job = JOBS.get(tid)
    if not job:
        return HTMLResponse("404")
    
    if job['status'] == 'done':
        d = job['data']
        # 使用多行字符串优化 HTML 拼接可读性
        html_content = f"""
            <div class='bg-gray-900 border border-green-600 p-4 rounded'>
                <div class='text-green-400 font-bold mb-2'>{job['log']}</div>
                <div class='grid grid-cols-2 gap-x-4 gap-y-1 text-sm font-mono text-gray-300 mb-3'>
                    <div>FOV: <span class='text-white'>{d['fov']}</span></div>
                    <div>Time: <span class='text-white'>{d['time']}</span></div>
                    <div>RA : <span class='text-yellow-300'>{d['ra']}</span></div>
                    <div>Dec: <span class='text-yellow-300'>{d['dec']}</span></div>
                    <div>Az : <span class='text-blue-300'>{d['az']}</span></div>
                    <div>Alt: <span class='text-blue-300'>{d['alt']}</span></div>
                </div>
                <img src='{job['result_url']}' class='w-full rounded shadow-lg mb-4'>
                <a href='/' class='block w-full bg-blue-700 text-white text-center py-2 rounded'>Next</a>
            </div>
        """
        return HTMLResponse(html_content)
    
    elif job['status'] == 'failed':
        return HTMLResponse(
            f"<div class='text-red-400 border-red-600 border p-4 rounded'>"
            f"Failed: {job['log']} <br><a href='/' class='underline'>Retry</a></div>"
        )
    
    return HTMLResponse(
        f"<div hx-get='/status/{tid}' hx-trigger='load delay:1s' hx-swap='outerHTML' class='text-blue-200'>"
        f"{job.get('log', 'Processing...')}</div>"
    )


@app.get("/", response_class=HTMLResponse)
async def index(req: Request):
    return templates.TemplateResponse("index.html", {"request": req})