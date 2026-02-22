import os
import shutil
import uuid
import asyncio
import logging
import time
import math
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ExifTags

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
from astropy.time import Time
import astropy.units as u

from timezonefinder import TimezoneFinder
import pytz

# --- 1. 配置与日志 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AstroServer")

BASE_DIR = Path(".")
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
NGC_CSV_URL = "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files/NGC.csv"
LOCAL_DB_PATH = BASE_DIR / "NGC.csv"

# 字体回退策略
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "C:\\Windows\\Fonts\\arial.ttf",
]

SOLVE_OPTS = [
    "--downsample", "2",
    "--scale-units", "degwidth",
    "--scale-low", "3",
    "--scale-high", "30",
    "--no-plots", "--overwrite", "--cpulimit", "60", "--no-verify",
    "--crpix-center"
]

tf_engine = TimezoneFinder(in_memory=True)
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory="templates")
JOBS = {}
DF_NGC = None

# --- 2. 基础功能 (保持不变) ---
def download_db_if_missing():
    if LOCAL_DB_PATH.exists(): return
    logger.info("Downloading NGC DB...")
    try:
        with requests.get(NGC_CSV_URL, stream=True) as r:
            with open(LOCAL_DB_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    except Exception:
        raise RuntimeError("DB download failed")

def load_catalog():
    global DF_NGC
    download_db_if_missing()
    df = pd.read_csv(LOCAL_DB_PATH, sep=';', low_memory=False)
    df = df.dropna(subset=['RA', 'Dec'])
    coords = df['RA'].astype(str) + " " + df['Dec'].astype(str)
    sc = SkyCoord(coords.to_numpy(), unit=(u.hourangle, u.deg))
    df['RA_deg'] = sc.ra.deg
    df['Dec_deg'] = sc.dec.deg
    df['V-Mag'] = pd.to_numeric(df['V-Mag'], errors='coerce').fillna(99.0)
    df['MajAx'] = pd.to_numeric(df['MajAx'], errors='coerce').fillna(0.0)
    df['M'] = pd.to_numeric(df['M'], errors='coerce')
    DF_NGC = df
    logger.info(f"Catalog loaded: {len(df)}")

def get_utc_time(img_path, lat, lon):
    local_dt = datetime.now()
    is_fallback = True
    try:
        with Image.open(img_path) as img:
            exif = img._getexif()
            if exif:
                for tag in [36867, 36868, 306]:
                    d = exif.get(tag)
                    if d:
                        try:
                            local_dt = datetime.strptime(d, "%Y:%m:%d %H:%M:%S")
                            is_fallback = False
                            break
                        except: continue
    except: pass

    try:
        tz_str = tf_engine.timezone_at(lng=lon, lat=lat)
        if tz_str:
            local_tz = pytz.timezone(tz_str)
            utc_dt = local_tz.localize(local_dt).astimezone(pytz.utc)
        else:
            utc_dt = local_dt.replace(tzinfo=pytz.utc)
    except:
        utc_dt = local_dt.replace(tzinfo=pytz.utc)
        
    return Time(utc_dt), is_fallback

def get_font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: continue
    return ImageFont.load_default()

# --- 3. 高级网格绘制核心算法 (Gemini修改版代码V3) ---

def calculate_dynamic_step(fov_deg, is_ra=False):
    """
    算法化计算最佳步长。
    """
    target_step = fov_deg / 4.0
    if target_step <= 0: target_step = 1.0 
    
    if is_ra:
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
        exponent = math.floor(math.log10(target_step))
        fraction = target_step / (10**exponent)
        
        if fraction < 1.5: base = 1
        elif fraction < 3.5: base = 2
        elif fraction < 7.5: base = 5
        else: base = 10
        
        return base * (10**exponent)

def format_ra_label(deg):
    a = Angle(deg, u.degree)
    h = int(a.hour)
    m = int(round((a.hour * 60) % 60))
    if m == 60: 
        h += 1
        m = 0
    return f"{h}h {m}m" if m != 0 else f"{h}h"

def format_dec_label(deg):
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
    Gemini V3: 使用 Pillow 的 anchor 参数实现绝对贴边的标注。
    """
    x, y = pos
    margin = 4
    
    # 颜色处理
    stroke_fill = (0, 0, 0, 255)
    
    anchor = 'mm' # 默认居中
    tx, ty = x, y

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

    draw.text((tx, ty), text, font=font, fill=color, anchor=anchor, stroke_width=2, stroke_fill=stroke_fill)


def draw_projected_line(draw, wcs, world_coords, width, height, color, label_func=None, font=None, text_color=None, is_dec=False):
    """
    Gemini V3: 
    1. 彻底解决漏标：使用“线段跨越边界”的几何检测法，捕捉所有进出屏幕的线。
    2. 解决错位：对所有捕捉到的边缘点强制吸附到 0/W/H。
    3. 解决极区圆环：如果没有检测到左右边缘穿越，则激活垂直中轴线标注。
    """
    try:
        pix_coords = wcs.wcs_world2pix(world_coords, 0)
    except:
        return

    # 分段处理 (断点检测)
    segments = []
    current_segment = []
    
    # 允许画出界的 Padding，保证线条连贯
    draw_padding = 100 # 增大 padding 确保能捕捉到穿越点
    
    for px, py in pix_coords:
        if np.isnan(px) or np.isnan(py):
            if current_segment: segments.append(current_segment)
            current_segment = []
            continue
            
        in_bounds = -draw_padding < px < width + draw_padding and -draw_padding < py < height + draw_padding
        
        if in_bounds:
            if current_segment:
                last_x, last_y = current_segment[-1]
                dist = math.hypot(px - last_x, py - last_y)
                if dist > max(width, height) / 4: # 断点检测
                    segments.append(current_segment)
                    current_segment = []
            current_segment.append((px, py))
        else:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
    
    if current_segment: segments.append(current_segment)
    
    label_candidates = [] # (x, y, align_type)

    for seg in segments:
        if len(seg) < 2: continue
        
        # 绘制线条
        draw.line(seg, fill=color, width=1)
        
        if not label_func: continue

        # --- Gemini V3 核心逻辑：线段跨越检测 (Edge Crossing) ---
        has_horizontal_crossing = False # 标记是否穿过了左右边缘
        
        # 遍历线段的每一步，检查是否穿过边界
        # 这种方法比检查端点距离更可靠，不会受 padding 影响
        
        # 为了性能，步长设为 1，或者简单遍历所有点
        for i in range(len(seg) - 1):
            p1 = seg[i]
            p2 = seg[i+1]
            x1, y1 = p1
            x2, y2 = p2
            
            # 左边缘检测 (x=0)
            if (x1 < 0 and x2 >= 0) or (x1 >= 0 and x2 < 0):
                # 计算交点 y
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

        # --- Gemini V3 核心逻辑：纬线中轴标注 (Fallback) ---
        # 如果是 Dec 线 (不管是直的还是圆的)，且没有穿过左右边缘 (即全在图内，或者是上下贯穿)，
        # 则在图像垂直中轴线附近找一个点补标。
        # 这解决了极区圆环没有标注的问题，同时也避免了普通纬线重复标注。
        if is_dec and not has_horizontal_crossing:
            # 提取可视区域内的点
            visible_points = [p for p in seg if 0 <= p[0] <= width and 0 <= p[1] <= height]
            
            if visible_points:
                # 找到离图像垂直中心线 (x = width/2) 最近的点
                center_cand = min(visible_points, key=lambda p: abs(p[0] - width/2))
                
                # 再次检查：如果是上下贯穿的线，可能会在顶部或底部被标过，这里再标中间也没问题
                # 但如果是圆环，这里是唯一的标注机会
                
                # 为了美观，如果是圆环下半部分，标在下方；上半部分标在上方
                # 简单起见，统一标在该点，align='center'
                label_candidates.append((center_cand[0], center_cand[1], 'center'))

    # 执行标注 (简单的空间去重)
    drawn_labels = []
    # 按照优先级排序：左右 > 上下 > 中间
    # 这样边缘标注优先
    priority_map = {'left': 0, 'right': 0, 'top': 1, 'bottom': 1, 'center': 2}
    label_candidates.sort(key=lambda item: priority_map.get(item[2], 2))

    for cx, cy, calign in label_candidates:
        # 检查是否太靠近已有标注
        is_cluttered = False
        for dx, dy in drawn_labels:
            if math.hypot(cx-dx, cy-dy) < 60: # 60px 避让半径
                is_cluttered = True
                break
        
        if not is_cluttered:
            draw_smart_label(draw, (cx, cy), label_func, font, width, height, text_color, align_type=calign)
            drawn_labels.append((cx, cy))


def draw_celestial_grid(draw, wcs, width, height):
    """
    球面网格绘制算法
    """
    
    # --- 1. 视场探测与步长计算 ---
    # 采样更多点以确保覆盖整个视场
    sample_grid_x = np.linspace(0, width, 10)
    sample_grid_y = np.linspace(0, height, 10)
    xx, yy = np.meshgrid(sample_grid_x, sample_grid_y)
    sample_pix = np.vstack([xx.ravel(), yy.ravel()]).T
    
    try:
        sample_world = wcs.wcs_pix2world(sample_pix, 0)
    except:
        return
        
    ra_samples = sample_world[:, 0]
    dec_samples = sample_world[:, 1]
    
    # 计算视场中心和 FOV
    c_world = wcs.pixel_to_world(width/2, height/2)
    center_ra = c_world.ra.deg
    
    # 计算 FOV 大小 (用于步长)
    e_world = wcs.pixel_to_world(width/2 + 10, height/2)
    pix_scale = c_world.separation(e_world).deg / 10.0
    fov_deg = pix_scale * max(width, height)
    
    # 动态步长
    step_ra = calculate_dynamic_step(fov_deg, is_ra=True)
    step_dec = calculate_dynamic_step(fov_deg, is_ra=False)
    
    # 极区步长修正 (依然保留是为了美观，防止极点处线条过密，但这不属于特判逻辑，而是密度控制)
    max_abs_dec = np.max(np.abs(dec_samples))
    if max_abs_dec > 80:
        step_ra = max(step_ra, 15.0)

    # --- 2. 统一计算绘制范围 (Unified Range Calculation) ---
    
    # A. Dec 范围 (确定画哪些纬线)
    min_dec, max_dec = np.min(dec_samples), np.max(dec_samples)
    d_start = math.floor(min_dec / step_dec) * step_dec
    d_end = math.ceil(max_dec / step_dec) * step_dec
    dec_lines = np.arange(d_start, d_end + step_dec + 0.001, step_dec)
    
    # B. RA 范围 (确定画哪些经线)
    # 使用相对角度法处理 0/360 跳变，计算视场覆盖的 RA 范围
    # 公式: ((ra - center + 180) % 360) - 180，将所有 RA 映射到 [-180, 180] 相对域
    ra_diffs = (ra_samples - center_ra + 180) % 360 - 180
    min_diff = np.min(ra_diffs)
    max_diff = np.max(ra_diffs)
    
    # 确定需要绘制的 RA 线列表
    # 将相对范围转换回绝对范围进行网格对齐
    ra_lines_raw = []
    # 稍微外扩一点范围以包含边缘的线
    search_min = min_diff - step_ra
    search_max = max_diff + step_ra
    
    # 遍历可能的偏移量
    # 这里的逻辑是：在 center_ra 附近搜索符合 step 的 grid
    # 这种方法比直接 min/max 更鲁棒，能处理跨越 0 度的情况
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

    # --- 4. 统一绘制循环 ---

    # === 画经线 (Meridians) ===
    # 策略：对于每一条在视场内的 RA 线，直接画从 -90 到 +90 的全长。
    # draw_projected_line 会负责裁剪掉屏幕外的部分。
    for ra_val in ra_lines:
        norm_ra = ra_val % 360
        
        # 路径：总是 -90 到 90
        path_dec = np.linspace(-90, 90, 300) # 点数给够，保证弯曲平滑
        path_ra = np.full_like(path_dec, norm_ra)
        
        world_coords = np.vstack([path_ra, path_dec]).T
        draw_projected_line(draw, wcs, world_coords, width, height, grid_color, 
                            label_func=format_ra_label(norm_ra), font=font, text_color=text_color, is_dec=False)

    # === 画纬线 (Parallels) ===
    # 策略：计算视场覆盖的 RA 跨度，大幅扩展(200%)后绘制。
    # 如果扩展后超过 360，则画全圆。
    for dec_val in dec_lines:
        if dec_val <= -90 or dec_val >= 90: continue
        
        # 计算绘制路径的 RA 范围
        span = max_diff - min_diff
        
        # 如果视场跨度很大(>120度)或者扩展后超过360，直接画 0-360 全圆
        if span > 120 or span * 3 > 360:
             path_ra = np.linspace(0, 360, 400)
        else:
            # 普通视场：左右各扩展 100% (即总宽度为 3 倍视场宽)
            # 这样保证大角度倾斜时线也能穿透屏幕
            pad = max(span, 10.0) # 至少扩展 10 度
            start_rel = min_diff - pad
            end_rel = max_diff + pad
            
            # 生成相对坐标路径
            rel_path = np.linspace(start_rel, end_rel, 400)
            # 转换回绝对坐标 (处理 0/360)
            # 这里不取模，直接传给 WCS，保持线条连续性 (Astropy WCS handle unbounded angles usually)
            # 但为了安全和 format_dec 统一，我们在构建 world_coords 时取模
            path_ra = center_ra + rel_path
            
        path_dec = np.full_like(path_ra, dec_val)
        # 注意：这里传入 path_ra % 360 可能会在 360/0 处产生断裂
        # 但 draw_projected_line 里的断点检测 (dist > width/4) 会正确处理这个问题，
        # 把它视为穿越屏幕的连续线条或者正确分段。
        # 对于非全圆的线，直接传原始值可能更好，但为了统一逻辑，我们依赖 V3 的分段绘制能力。
        world_coords = np.vstack([path_ra % 360, path_dec]).T
        
        draw_projected_line(draw, wcs, world_coords, width, height, grid_color,
                            label_func=format_dec_label(dec_val), font=font, text_color=text_color, is_dec=True)
        
    # 十字丝
    cx, cy = width/2, height/2
    len_cross = min(width, height) * 0.03
    draw.line([(cx-len_cross, cy), (cx+len_cross, cy)], fill=(255, 50, 50, 180), width=2)
    draw.line([(cx, cy-len_cross), (cx, cy+len_cross)], fill=(255, 50, 50, 180), width=2)

def draw_final_output(wcs_path, img_path, out_path, task_id):
    """主绘图入口"""
    if DF_NGC is None: return 0, None

    # 1. 加载 WCS
    with fits.open(wcs_path) as hdul:
        header = hdul[0].header
        wcs = WCS(header)
    
    # 2. 加载图像并修复 WCS 尺寸信息
    with Image.open(img_path) as im:
        im = im.convert("RGBA")
        w, h = im.size
        wcs.pixel_shape = (w, h)

        draw = ImageDraw.Draw(im)
        
        # 3. 绘制高级网格与标注
        draw_celestial_grid(draw, wcs, w, h)
        
        # 4. NGC 标注
        fp = wcs.calc_footprint()
        if fp is None:
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
            fp = wcs.wcs_pix2world(corners, 0)

        dec_min, dec_max = np.min(fp[:,1]), np.max(fp[:,1])
        c_sky = wcs.pixel_to_world(w/2, h/2)
        corner = wcs.pixel_to_world(0,0)
        radius = c_sky.separation(corner).deg
        
        limit_mag = np.clip(14.0 - (radius / 2.5), 3.5, 12.0)
        
        df = DF_NGC
        cands = df[(df['Dec_deg'] >= dec_min - 2) & (df['Dec_deg'] <= dec_max + 2) & (df['V-Mag'] <= limit_mag)]
        
        count = 0
        font = get_font(max(11, int(h/55)))
        
        if len(cands) > 0:
            pts_world = np.vstack((cands['RA_deg'], cands['Dec_deg'])).T
            try:
                pts_pix = wcs.wcs_world2pix(pts_world, 0)
            except: pts_pix = []
            
            for i, (_, row) in enumerate(cands.iterrows()):
                if i >= len(pts_pix): break
                px, py = pts_pix[i]
                if np.isnan(px) or not (0 <= px <= w and 0 <= py <= h): continue
                
                name = f"M{int(row['M'])}" if pd.notna(row['M']) else row['Name']
                maj = row['MajAx']
                r_pix = 15
                if maj > 0:
                    r_deg = (maj/2)/60
                    pix_scale = c_sky.separation(wcs.pixel_to_world(w/2+1, h/2)).deg
                    if pix_scale > 0: r_pix = r_deg / pix_scale
                
                r_pix = max(r_pix, 8)
                bbox = [px-r_pix, py-r_pix, px+r_pix, py+r_pix]
                draw.ellipse(bbox, outline="yellow", width=1)
                draw.text((px+r_pix*0.8, py+r_pix*0.8), name, font=font, fill="yellow", stroke_width=2, stroke_fill="black")
                count += 1

        im.save(out_path, "PNG")
        return count, wcs

# --- 4. 任务管道 ---

async def processing_pipeline(task_id, fpath, lat, lon):
    job = JOBS[task_id]
    temps = []
    
    try:
        utc_time, is_fb = get_utc_time(fpath, lat, lon)
        job["obs_time_utc"] = utc_time.to_datetime()

        # Astrometry
        job.update({"status": "solving", "log": "Astrometry 解析中..."})
        cmd = ["solve-field", str(fpath)] + SOLVE_OPTS
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        
        wcs_f = fpath.with_suffix(".wcs")
        if not wcs_f.exists(): raise RuntimeError("解析失败: 星点不足")
        temps.append(wcs_f)
        temps.extend(fpath.parent.glob(f"{fpath.stem}.*"))

        # Constellation Lines
        job.update({"status": "plotting", "log": "绘制连线..."})
        ppm = fpath.with_suffix(".ppm")
        base_png = fpath.parent / f"{task_id}_base.png"
        temps.extend([ppm, base_png])
        
        await asyncio.to_thread(lambda: Image.open(fpath).convert("RGB").save(ppm))
        plot_cmd = ["plot-constellations", "-w", str(wcs_f), "-i", str(ppm), "-o", str(base_png), "-C", "-B", "-N"]
        proc_plot = await asyncio.create_subprocess_exec(*plot_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc_plot.communicate()

        # Annotation
        job.update({"status": "annotating", "log": "标注天体与网格..."})
        final_png = STATIC_DIR / f"{task_id}_annotated.png"
        src = base_png if base_png.exists() else fpath
        
        cnt, wcs_obj = await asyncio.to_thread(draw_final_output, wcs_f, src, final_png, task_id)

        # Data Calculation
        w, h = wcs_obj.pixel_shape if wcs_obj.pixel_shape else (0,0)
        if w==0: 
             with Image.open(fpath) as im: w, h = im.size
             
        c_sky = wcs_obj.pixel_to_world(w/2, h/2)
        # FOV (Great Circle)
        fov_w = wcs_obj.pixel_to_world(0, h/2).separation(wcs_obj.pixel_to_world(w, h/2)).deg
        fov_h = wcs_obj.pixel_to_world(w/2, 0).separation(wcs_obj.pixel_to_world(w/2, h)).deg
        
        # AltAz
        loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=10*u.m)
        altaz = c_sky.transform_to(AltAz(obstime=utc_time, location=loc))

        job["data"] = {
            "fov": f"{fov_w:.1f}° x {fov_h:.1f}°",
            "ra": f"{c_sky.ra.to_string(unit=u.hour, sep='h', precision=1)}", # 格式化为 hh:mm:ss
            "dec": f"{c_sky.dec.to_string(unit=u.deg, sep='°', precision=1)}",
            "az": f"{altaz.az.deg:.1f}°",
            "alt": f"{altaz.alt.deg:.1f}°",
            "time": utc_time.strftime("%Y-%m-%d %H:%M UTC") + (" (Est.)" if is_fb else "")
        }
        job.update({"status": "done", "log": f"完成: {cnt} objects", "result_url": f"/static/{final_png.name}"})

    except Exception as e:
        logger.exception("Task Failed")
        job.update({"status": "failed", "log": str(e)})
    finally:
        for p in temps:
            if p.exists() and p != fpath: 
                try: os.remove(p)
                except: pass

@app.on_event("startup")
async def startup():
    UPLOADS_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(exist_ok=True)
    load_catalog()

@app.post("/upload")
async def upload(bg: BackgroundTasks, file: UploadFile=File(...), lat:str=Form(None), lon:str=Form(None)):
    if not lat or not lon: return HTMLResponse("<div class='text-red-400'>Error: GPS Required</div>")
    tid = str(uuid.uuid4())
    fp = UPLOADS_DIR / f"{tid}{Path(file.filename).suffix}"
    with open(fp, "wb") as f: shutil.copyfileobj(file.file, f)
    
    JOBS[tid] = {"status": "pending", "log": "Queued"}
    bg.add_task(processing_pipeline, tid, fp, float(lat), float(lon))
    return HTMLResponse(f"<div hx-get='/status/{tid}' hx-trigger='load delay:1s' hx-swap='outerHTML' class='text-blue-300 animate-pulse'>Starting...</div>")

@app.get("/status/{tid}")
async def status(tid: str):
    job = JOBS.get(tid)
    if not job: return HTMLResponse("404")
    if job['status'] == 'done':
        d = job['data']
        return HTMLResponse(f"""
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
        """)
    elif job['status'] == 'failed':
        return HTMLResponse(f"<div class='text-red-400 border-red-600 border p-4 rounded'>Failed: {job['log']} <br><a href='/' class='underline'>Retry</a></div>")
    return HTMLResponse(f"<div hx-get='/status/{tid}' hx-trigger='load delay:1s' hx-swap='outerHTML' class='text-blue-200'>{job.get('log','Processing...')}</div>")

@app.get("/", response_class=HTMLResponse)
async def index(req: Request):
    return templates.TemplateResponse("index.html", {"request": req})