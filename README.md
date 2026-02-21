# AstroLocal

**AstroLocal** 是一个基于 **Linux/WSL (Windows Subsystem for Linux)** 的本地天文盲解（Plate Solving）与标注工具。

它将 **Astrometry.net** 引擎封装在一个现代化的 Web 界面中。按下面的操作部署，可为常用的摄影镜头焦段（80mm - 400mm 焦段）实现精准解析。无需上传图片到云端排队，利用本地算力实现解析，并自动叠加 **Messier/NGC/IC** 深空天体标注与星座连线。

![Screenshot Placeholder](./8c2cdf55-681c-4e99-a62f-840eafad1ee6_annotated.png)

## 主要特性

*   **本地解析**：基于 WSL 运行，无需排队，解析速度更高。
*   **深空天体标注**：
    *   自动下载并使用 **OpenNGC** 数据库（包含 NGC, IC, Messier）。
    *   **动态星等过滤**：视场大时只标亮星，视场小时标暗星，避免文字重叠。
    *   **智能标注**：优先显示 M 编号（如 M31），支持黄色虚线框选范围。
*   **星座连线**：自动绘制星座连线与边界。
*   **现代化界面**：
    *   基于 **FastAPI + HTMX + TailwindCSS**。
    *   **响应式设计**：支持电脑端和手机端（局域网访问）。
    *   **实时进度**：显示详细的解析步骤（提取星点 -> 匹配索引 -> 绘图）。

## 环境要求

*   **操作系统**: Windows 10/11 (已启用 WSL2) 或 Linux (Ubuntu/Debian)。
*   **Python**: 3.10+ (推荐使用 `uv` 进行包管理)。
*   **系统依赖**: `astrometry.net` 引擎。

## 安装指南

### 1. 安装系统级依赖 (WSL/Ubuntu)

终端中运行：

```bash
sudo apt update
sudo apt install astrometry.net astrometry-data-tycho2
# 安装绘图工具依赖
sudo apt install libnetpbm10 netpbm
# 可选：如果 plot-constellations 报错缺失数据
sudo apt install astrometry.net-data-ngc
```

### 2. 下载索引文件（必要）

Astrometry 需要索引文件才能工作。针对常见的 80mm-400mm 焦段（视场约 5°-24°），推荐下载 **4100 系列** 索引。

```bash
cd /usr/share/astrometry
# 下载适合广角到中长焦的索引 (Index 4107 - 4116)
# 注意：需要 sudo 权限
sudo wget -c http://data.astrometry.net/4100/index-4107.fits
sudo wget -c http://data.astrometry.net/4100/index-4108.fits
sudo wget -c http://data.astrometry.net/4100/index-4109.fits
sudo wget -c http://data.astrometry.net/4100/index-4110.fits
sudo wget -c http://data.astrometry.net/4100/index-4111.fits
sudo wget -c http://data.astrometry.net/4100/index-4112.fits
sudo wget -c http://data.astrometry.net/4100/index-4113.fits
sudo wget -c http://data.astrometry.net/4100/index-4114.fits
sudo wget -c http://data.astrometry.net/4100/index-4115.fits
sudo wget -c http://data.astrometry.net/4100/index-4116.fits
```

当然你可以采用任何你喜欢的方式下载并拷贝fits文件到对应目录。这里的索引文件总大小不会超过 1GB.

如有需要，也可以下载 4200 系列和 4100 系列的其他索引，以支持更多样的视场。

### 3. 克隆项目与安装 Python 依赖

本项目推荐利用 uv 进行包管理。

```bash
# 进入项目目录
cd ~/my_astro_web

# 如果没有 uv，先安装: pip install uv
# 初始化环境并安装依赖
uv init
uv add fastapi uvicorn python-multipart jinja2 pandas numpy astropy pillow requests scipy
```

## 运行方法

在项目根目录下运行：

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

*   **电脑访问**: 浏览器打开 `http://localhost:8000`
*   **手机访问**: 确保手机和电脑在同一局域网，访问 `http://<WSL宿主机IP>:8000`
    *   *注*: WSL IP 可通过 `hostname -I` 查看。如果无法访问，可能需要配置 Windows 防火墙或端口转发。

用手机查看的方便性在于可以直接拍摄相机屏幕实现即刻解析。

## 项目结构

```text
.
├── main.py              # 核心后端逻辑 (FastAPI + 绘图)
├── templates/
│   └── index.html       # 前端页面 (Tailwind + HTMX)
├── static/              # 存放解析结果图片
├── uploads/             # 存放上传的原始图片
├── NGC.csv              # OpenNGC 数据库 (首次运行自动下载)
├── pyproject.toml       # uv 依赖配置
└── README.md            # 说明文档
```

## 参数调整

### 1. 镜头调整

如果你需要修改解析参数（例如更换了镜头），请编辑 `main.py` 中的 `SOLVE_OPTS`：

```python
SOLVE_OPTS = [
    "--downsample", "2",           # 降采样 (加速)
    "--scale-units", "degwidth",   # 单位: 视场宽度(度)
    "--scale-low", "3",            # 最小视场 (400mm+)
    "--scale-high", "30",          # 最大视场 (80mm-)
    "--no-plots",                  # 不生成冗余分析图
    "--overwrite", 
    "--cpulimit", "60"             # 超时时间
]
```

并记得增补索引文件。

### 2. 标注信息

`main.py` 里面有一行

```python
limit_mag = 14.0 - (fov_radius_deg / 3.0)
```

这决定了视场和显示的最暗星等的对应关系。如果你需要更丰富的标注，可以自由调整，或者写死为一个比较大的值。

## （可能的）常见问题

**Q: 解析一直显示 Failed，日志提示 "Did not solve"？**
A:
1.  检查是否下载了正确的索引文件 (`/usr/share/astrometry` 下是否有 `index-41xx.fits`)。
2.  检查图片星点是否清晰（拖线严重或失焦会导致失败）。
3.  尝试放宽 `SOLVE_OPTS` 中的 `--scale-low` 和 `--scale-high` 范围。

**Q: OpenNGC 数据库下载失败？**
A:
程序启动时会自动从 GitHub 下载 `NGC.csv`。如果网络不通，请手动下载 [OpenNGC CSV](https://github.com/mattiaverga/OpenNGC/raw/master/database_files/NGC.csv)，并重命名为 `NGC.csv` 放入项目根目录。

**Q: 标注图上没有深空天体？**
A:
可能是视场内确实没有亮于极限星等的天体。程序会根据视场大小动态计算极限星等。你可以修改 `draw_annotations` 函数中的 `limit_mag` 逻辑来强制显示更暗的天体。参考前文。

## License

MIT License.

OpenNGC Database is licensed under CC-BY-SA 4.0.

Gemini 3 Pro 参与了本项目的构建过程。
