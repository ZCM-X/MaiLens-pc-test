# MaiLens-pc-test

MaiLens-pc-test 是一个独立的“手机采集 → Wi‑Fi → 电脑处理”实验项目。它不改动原来的 MaiLens iPhone 工程，先把手机的超广角视频和同一帧的姿态数据送到电脑，所有稳定、裁切和机台检测算法在电脑上迭代，验证后再移植回 iPhone。

## 当前链路

```text
iPhone 0.5× 相机
  ├─ JPEG 视频帧（默认 1280×720，目标 60 fps）
  └─ 同帧姿态（四元数、重力、角速度、时间戳）
             │ TCP / 局域网
             ▼
电脑 pc/pc_receiver.py（可实时处理）
  ├─ sessions/<时间>/frames/*.jpg
  ├─ sessions/<时间>/capture.jsonl
  ├─ sessions/<时间>/raw.mp4
  └─ sessions/<时间>/processed-live.mp4（实时模式）

同一路采集也能完全脱离电脑，直接录在手机上：
iPhone 本地录制（App 内“开始录制”）
  └─ MaiLensSessions/<时间>/
       ├─ video.mp4（原始 60 fps H.264）
       ├─ capture.jsonl（每帧时间戳 + 最近姿态）
       ├─ pose.jsonl（120 Hz 四元数/重力/角速度/用户加速度）
       └─ session.json（清单）
             │ 局域网“发到电脑”（端口 +1），或爱思助手 / “文件”App
             ▼
电脑 pc/session_server.py → phone_sessions/<时间>/
  └─ pc/import_phone_session.py → sessions/<时间>-phone/ → pc/process_session.py
```

视频和姿态被放在同一个 TCP 数据包里，电脑端不需要猜测两条流的对应关系。发送端使用有限缓冲，只保留最新待发送帧；电脑或网络变慢时会丢弃旧帧，不会把延迟越积越大。

## 机台锁定（pc/machine_lock.py）

这是对标参考视频里“机台在画面中间完全不动”的离线管线，也是移植回 iPhone 之前的验证台。它不依赖 YOLO 权重，直接在同一套鱼眼矫正后的画面上做几何测量：

```text
每一帧
  ├─ 鱼眼矫正（fov 106.4583 / k1 0.0893163 / k2 -0.0174637 / crop 0.74）
  ├─ 内屏：青色掩膜 H70–115 S>60 V>40 → 最大圆盘连通域 → fitEllipse
  ├─ 外键：紫色掩膜 H125–145 S>80 V>60 → 最多 12 个连通域质心 → 圆拟合
  └─ 8 个按键的 8 次谐波相位 → 机台横滚
        │
        ▼
  零相位平滑（对称高斯核 + 5 帧中值预滤，无时间滞后）
        │
        ▼
  单次仿射变换：内屏中心 → 画面正中，内屏长轴 → 固定像素直径
```

内屏的**长轴**当作距离信号（它在俯仰时不会像短轴那样被压缩），所以手机前后移动时画面会自动反向缩放，机台在屏幕上的大小保持不变；缩放只放大不缩小，永远不会露出黑边。

在 `IMG_8901.MOV`（2160×3840，801 帧）上的实测：

```text
机台中心偏离画面正中   均值 14.2 px   中位 12.0 px   p95 33.4 px   （1080 宽输出）
内屏在画面上的直径     中位 776 px = 画面宽度的 72%
自动补偿的缩放范围     1.29–1.65
取样窗口越出画面的帧   0 / 801
原始检测噪声           0.8 px（1080 下约 1.6 px）
```

检测噪声只有 0.8 px，所以平滑强度必须**很小**：`--smooth-sigma` 取 12 会让机台自己漂 95 px，取 3 才是既有锁定又不抖的正确工作点。

`--margin`（默认 1.25）会先按更宽的鱼眼视角（`crop / margin`）展开一张 1.25 倍的工作画布，锁完再从中间裁出成片。`IMG_8901.MOV` 里有 117 帧机台本来靠近原画幅边缘，直接平移会采到画外、`BORDER_REFLECT101` 补出镜像的“翅膀”；多这一圈余量之后越界帧变成 0，而且鱼眼在这一圈仍然有像素（黑边 0.00%）。如果哪天机台偏得更狠，`need` 会自动抬高缩放兜底，代价是那一小段画面会轻微推近。

```powershell
# 1:1 输出 2160×3840
.\.venv\Scripts\python.exe pc\machine_lock.py "C:\Users\93543\Downloads\IMG_8901.MOV" `
  --output "C:\Users\93543\Downloads\IMG_8901_机台锁定_v1.mp4" `
  --output-scale 1.0 --smooth-sigma 3

# 快速预览 + 12 格拼图 + 逐帧诊断数据
.\.venv\Scripts\python.exe pc\machine_lock.py "C:\Users\93543\Downloads\IMG_8901.MOV" `
  --output work\locked.mp4 --output-scale 0.5 `
  --contact work\locked_contact.jpg --trace work\locked.jsonl --draw
```

`--draw` 会在画面正中画绿色十字，并画一个红圈标出机台实际落点，两者重合就说明锁定成立；左上角 `lock err` 是这一帧的残差像素。`--trace` 里的 `resid_x / resid_y` 是同一件事的逐帧数值。

`--roll-gain`（默认 0）打开后会用按键环的相位抵消手机横滚，相位每 45° 一个周期，管线里先解缠再平滑。这一路信号噪声约 2.3°/帧，比中心弱，所以默认不开。

## 电脑端启动

在 Windows PowerShell 中（先 `cd` 到项目根目录，也就是放着 `.venv`、`pc`、`models` 的那一层；不在这一层会直接报「无法将`.\.venv\Scripts\python.exe`项识别为...」）：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe pc\pc_receiver.py --host 0.0.0.0 --port 8765 `
  --process-live --preview
```

采集端会尝试让 0.5× 摄像头以 60 fps 输出；App 界面显示的是实际发送帧率。实时处理默认使用 `--processing-scale 0.5`，先以半分辨率完成鱼眼矫正和云台，再放大到预览尺寸，以降低延迟；原始 JPEG 仍按完整分辨率保存。JPEG 编码、Wi‑Fi 或电脑处理跟不上时，发送端会丢弃旧帧保持低延迟，预览上会显示实际值，例如 `48.2/60 fps`。需要最高画质时可改为 `--processing-scale 1.0`，但帧率会下降。

电脑防火墙允许 Python 监听 TCP 8765。手机和电脑必须在同一个局域网，手机端填写电脑的局域网 IPv4 地址（例如 `192.168.1.23`），不能填写 `127.0.0.1`。

这条命令会在收到每个视频帧后立即处理并显示结果。按 `q` 或 `Ctrl+C` 停止，终端会打印会话目录，例如：

```text
sessions/20261001-153012
```

实时模式会同时保存 `raw.mp4` 和 `processed-live.mp4`。如果先只采集原始数据，之后再离线调参，可以用：

```powershell
.\.venv\Scripts\python.exe pc\process_session.py sessions\20261001-153012 --output processed.mp4 --preview
```

默认处理使用姿态四元数建立数字云台变换；它会把第一帧姿态作为锁定方向，并通过边缘反射填补裁切后的空白。`--crop 0.74` 控制保留中心视场，值越小预留的稳定余量越大。

可以先用单张照片验证“鱼眼反变换 + 机台平面拉正 + 居中”这一目标。工具会先按 EXIF 方向读取照片，再用镜头参数做逆鱼眼；`--screen-points` 是逆鱼眼中内屏椭圆的左、上、右、下四点，单应变换会把它拉成中央圆形：

```powershell
.\.venv\Scripts\python.exe tools\rectify_photo.py `
  --input "H:\IMG_8811.JPG" `
  --output "work\IMG_8811_reference.jpg" `
  --width 625 --height 559 --radius 200 `
  --screen-points "232,577 512,344 796,577 512,811"
```

这四点目前是照片实验的手工点，证明了目标几何关系。实时版本要由模型/椭圆拟合器每个关键帧产生这四点，再对单应矩阵做平滑；只有一个外框矩形无法恢复倾斜和前后移动造成的透视变化。

处理器默认载入当前 MaiLens 的鱼眼参数（中心 `0.501753869, 0.499423644`、`k1=0.0893163`、`k2=-0.0174637`、输出视场角 `106.4583°`），所以电脑生成的结果会先做鱼眼反变换，再做姿态稳定。参数可以直接用 `--center-x`、`--center-y`、`--k1`、`--k2` 和 `--fov` 覆盖。

如果要让实时模式同时做机台外框/内屏锁定，直接使用仓库内的 ONNX 模型。实时接收端和标注工具预标注默认都使用 v4；它们在 App 中处理的是手机送来的原始鱼眼帧，普通照片上看起来正确的框不代表鱼眼帧也能识别。电脑端优先用 OpenCV DNN 推理，不需要安装 PyTorch：

```powershell
.\.venv\Scripts\python.exe pc\pc_receiver.py --port 8765 `
  --process-live --preview --machine-lock --debug
```

如果要使用 `.pt` 等非 ONNX 模型，才需要额外安装 `ultralytics` 和对应的 PyTorch 运行环境。

实时检测默认每 12 帧运行一次，模型在原始鱼眼帧上检测，再映射到鱼眼矫正后的输出坐标。首次同时得到有效 `outer_buttons` 和 `inner_screen` 后，处理器会在外圈按键与内屏组成的机台区域内取特征点，用 LK 光流和 RANSAC `findHomography` 估计每帧的平面运动，把整台机台平面反向映射到一个固定的中央目标；这个变换同时补偿横移、横滚、倾斜和中等幅度的前后移动，八个按键与内屏保持同一相对位置，不再把检测框中心和面积当作唯一稳定信号。特征点暂时不足时保持上一张可靠的单应矩阵，超过丢失时限才回到搜索状态。

调试画面用黄色框标外圈按键区域、绿色框标内屏，白色十字是输出中心；`processed.jsonl` / `debug.jsonl` 会额外记录 `plane_lock`、内点数、内点比例和重投影误差。`--lock-fill` 调整首次锁定时内屏在画面短边中的占比，默认 `0.71`，对应近景校准参考中内屏约占短边 71% 的构图；数值越大，机台越近。这个比例依赖正确的 `inner_screen` 框，当前模型的外框/内屏定位仍需人工校正，暂时不要把错误预测当成最终输出验收。当前版本仍需要内屏在鱼眼有效视野内且保留可跟踪纹理；完全无纹理或被遮挡时会安全保持上一帧，不会用错误框跳走。

## 标注自己的机台数据

这里要训练两套模型，两个数据集不要混在一起：

1. 几何模型：`outer_buttons` + `inner_screen`，负责识别外圈按键区域、内屏以及两者的距离关系，用于机台居中和裁切。
2. 按键模型：`button` + `inner_screen`，负责在内屏坐标系中识别谱面按键；它的结果不参与机台居中。

当前内置模型会把部分机台外框和内屏框错，工具支持先用模型生成可修改的建议框，再由人确认修正。运行几何数据标注工具：

```powershell
.\.venv\Scripts\python.exe tools\annotate_geometry.py `
  --input sessions\你的会话目录 --output datasets\geometry-review `
  --preset geometry --every 6 --val-every 10 `
  --model models\frame-geometry-yolo11n-v5.onnx
```

`--model` 是可选项；打开图片后会用模型自动画出黄色 `outer_buttons [AI]` 和绿色 `inner_screen [AI]` 建议框。先检查位置，再用 `1`/`2` 选类别并拖出正确矩形，新的框会替换该类别的 AI 框；`x` 删除当前类别，`s` 保存，`n`/空格切下一张时会自动保存当前 AI 建议和人工修改，`p` 上一张，`r` 清除此图，`q` 退出。中文目录和文件名可以直接读写；重新打开同一 `--output` 会跳到第一个未标注帧。首次校正建议用独立的 `--output` 目录，避免覆盖之前的人工数据。

iPhone 的 `.HEIC/.HEIF` 照片也可以直接读取。照片目录建议每张都看，所以把 `--every` 设为 `1`，例如：

```powershell
.\.venv\Scripts\python.exe tools\annotate_geometry.py `
  --input "C:\Users\93543\Downloads\maimoller训练" `
  --output datasets\maimoller-geometry-review --preset geometry --every 1 `
  --model models\frame-geometry-yolo11n-v5.onnx
```

几何模型按截图中的方式标：`outer_buttons` 只画一个整体框，覆盖外圈 8 个实体按键和它们所在的环形区域；不要拆成 8 个小框。`inner_screen` 沿圆形游戏屏幕边缘画一个整体矩形。若静态照片上的框看似正确、实时推流中仍错，优先用手机推流会话目录作 `--input` 并按 `--every 12` 抽帧校正；App 的检测器直接吃这些原始鱼眼帧，训练数据也要包含这种输入。

按键模型单独使用另一份输出目录：

```powershell
.\.venv\Scripts\python.exe tools\annotate_geometry.py `
  --input "C:\Users\93543\Downloads\maimoller训练" `
  --output datasets\maimoller-buttons --preset buttons `
  --every 1 --val-every 10 --max-instances 16
```

按键模型中 `1=button`，`2=inner_screen`；`button` 可以在同一张图重复框选多个谱面按键，按 `z` 撤销最后一个。若要专门训练滑条，把 `--preset buttons` 改成 `--preset slides`，类别会变成 `slide + inner_screen`。

两套模型的标注语义分别是：

- 几何模型：`outer_buttons`、`inner_screen`，外圈按键区域只标一个整体框。
- 按键模型：`button`、`inner_screen`，每个谱面按键一个框；内屏框在两套模型中都保留，作为坐标参考。

不要为了每张图都凑两个框而猜测。某个目标被遮挡或出画时，只标清楚可见的那个；两类都看不清就跳过样本。工具输出标准 YOLO `images/{train,val}`、`labels/{train,val}` 和 `dataset.yaml`。建议先标至少 100 张，覆盖远近、左右偏移、倾斜、遮挡和曝光变化，再按 Ultralytics YOLO 文档训练并导出 ONNX。

本仓库提供了一个可复现的训练包装器。它会先检查 YOLO 标注，再为 Windows 生成绝对数据根目录的运行时 YAML，避免 Ultralytics 把 `path: .` 误解为当前终端目录：

```powershell
.\.venv\Scripts\python.exe tools\train_detector.py `
  --dataset datasets\maimoller-geometry\dataset.yaml `
  --weights ..\..\yolo11n.pt --device 0 `
  --project runs --name geometry-yolo11n `
  --export onnx
```

`maimoller-geometry` 是 51 张手标图的小数据集，`frame-geometry-yolo11n-v2.onnx` 就是它训出来的。手标图太少，`inner_screen` 在手机真实素材上经常只有 0.2-0.4 置信度，机台锁最吃的就是这个类别，所以又扩了一版：

```powershell
# 手标 51 张 + D:\桌面文件\训练2 的 97 张照片 + 22 段会话里抽 211 帧鱼眼图
.\.venv\Scripts\python.exe tools\build_geometry_dataset.py `
  --source-dataset datasets\maimoller-geometry --source-sessions sessions `
  --photo-dir "D:\桌面文件\训练2" `
  --model models\frame-geometry-yolo11n-v2.onnx `
  --output datasets\maimoller-geometry-v3 --samples-per-session 12 --max-side 1600

.\.venv\Scripts\python.exe tools\train_detector.py `
  --dataset datasets\maimoller-geometry-v3\dataset.yaml `
  --weights ..\..\yolo11n.pt --device 0 --epochs 100 --imgsz 640 --batch 16 `
  --project runs --name geometry-v3 --export onnx
```

新模型是 `models/frame-geometry-yolo11n-v4.onnx`（354 张训练图，验证集 5 张手标：mAP50 0.97，outer 0.995 / inner 0.945）。同一批真实素材上，`inner_screen` 置信度：矫正片段 0.24→0.76（35/40→40/40），手机原始鱼眼帧 0.78→0.87（47/48→48/48），训练照片 0.65→0.95。手标验证集只有 5 张，所以这个数字只说明标注闭环没崩，真正的验收还是看实拍锁定效果。

针对“标注工具里框正确、手机实时输入却框偏”的差异，v5 从 v4 权重继续微调，训练数据合并了 `training2-geometry-review-20261002` 与 20 张本次真实手机鱼眼帧（18 train / 2 val）。ONNX 使用 PC 接收端同一套 OpenCV DNN 前后处理，在这 20 张手机帧上外框平均 IoU 从 v4 的 0.259 升到 0.920，内屏从 0.356 升到 0.929；该检查含训练帧，主要证明输入域适配，不能当作独立测试成绩。v5 文件为 `models/frame-geometry-yolo11n-v5.onnx`，PC 实时接收和收到录制后的处理现已默认选它，v4 仍保留可手动回退。

这轮 v5 的训练可在项目根目录复现：

```powershell
.\.venv\Scripts\python.exe tools\train_detector.py `
  --dataset datasets\live-phone-finetune-20261002\dataset.yaml `
  --weights Training\machine-detector.pt --device 0 --epochs 80 `
  --imgsz 640 --batch 8 --patience 15 --project runs `
  --name geometry-live-phone-v5 --disable-amp --export onnx
```

如果要把模型用在离线处理上：

如果要把模型用在离线处理上（下面都用新模型 v4）：

```powershell
.\.venv\Scripts\python.exe pc\process_session.py sessions\20261001-153012 `
  --model models\frame-geometry-yolo11n-v5.onnx --output processed-machine.mp4 --debug
```

检测器锁定只使用几何模型的 `outer_buttons` 和 `inner_screen`；按键模型的结果不会把机台锁到某个谱面元素。当前电脑算法保留了调试输出：`debug.jsonl` 中有姿态延迟、中心、缩放和检测框，便于先在电脑上调曲线。

## 手机端本地录制

App 里的“本地录制”和推流互相独立：不连电脑、不开 Wi‑Fi 也能录。它复用预览那一路 0.5× 采集，但直接写原始文件，不经 JPEG、不经网络，所以网络丢帧和压缩都不会污染素材：

- `video.mp4`：H.264，默认 720×1280@60，约 11 Mbps。
- `capture.jsonl`：每帧的 `frame_id`、`timestamp`（mach 时间）、宽高、该帧匹配到的姿态和 `sensor_delta_ms`。
- `pose.jsonl`：120 Hz 姿态全量日志，字段和推流协议里的 `pose` 完全一致，另带 `user_acceleration`（去掉重力后的加速度，做前后移动补偿用）。
- `session.json`：帧数、姿态样本数、时长、丢帧、时钟说明。

每次录制按时间写进手机上的 `MaiLensSessions/<yyyyMMdd-HHmmss>/`，拿到电脑上有四种办法，前两种就在 App 里：

- “发到电脑”（推荐）：走局域网直接推到电脑，不用线、不用第三方工具。电脑上先起接收端，端口固定是推流端口 +1（默认 8766）。`--process` 收到就用保存的 `video.mp4` 加姿态日志直接出稳定结果（`phone_sessions/<名字>/processed.mp4`），中间不需要抽帧；`--import` 是给标注准备的，会额外生成 `sessions/<名字>-phone/frames/`；两个都不加就只收原始文件。只要手机正连着电脑推流，录完会自动发送这一段；没在推流就手动点“发到电脑”。
- “导出”：存到“文件”或分享出去。
- 手机“文件 → 我的 iPhone → MaiLensRemoteCapture”里按时间找文件夹。
- 用爱思助手直接拷整个文件夹。

电脑这边三种用法，按需要挑：

```powershell
# 所有命令都在项目根目录运行

# 电脑端接收 + 直接处理（另开一个终端，手机点“发到电脑”之前先跑起来）
.\.venv\Scripts\python.exe pc\session_server.py --process --debug

# 默认自动使用 models\frame-geometry-yolo11n-v5.onnx；需要时可用 --model 覆盖

# 也可以事后处理收到的会话：直接读 video.mp4 + pose.jsonl，不抽帧
.\.venv\Scripts\python.exe pc\process_session.py phone_sessions\20261002-153000 `
  --model models\frame-geometry-yolo11n-v5.onnx --output processed-phone.mp4 --debug

# 只有要标注 JPEG 帧时才抽帧
.\.venv\Scripts\python.exe pc\import_phone_session.py "D:\phone\20261002-153000"
```

没有 `.venv` 时，把 `.\.venv\Scripts\python.exe` 换成 `py` 也能跑（只要本机 Python 装了 `opencv-python` 和 `numpy`）。

`process_session.py` 两种输入都吃：有 `frames/*.jpg` 的老会话照旧逐张读；手机上只有 `video.mp4` 时就直接解码视频，按帧号和 `capture.jsonl` 对齐，缺姿态的帧再从 120 Hz 的 `pose.jsonl` 里就近取；录像比日志长（中途被杀掉）就按最后一帧的时钟外推，整段照样跑完。`import_phone_session.py` 只在需要 JPEG 帧做标注时才用，它把视频解成 `frames/*.jpg` 并给 `capture.jsonl` 补上 `frame_path`。

如果 `session_server.py` 报 `bad magic` 或提示 `MLCP`，那是手机 App 里的端口被填成了上传端口：改回推流端口（默认 8765），上传端口由 App 自己用「推流端口 +1」算出来，不用手填。

## iPhone 端构建

这是一个独立的最小采集 App。电脑上安装 XcodeGen 后：

```bash
xcodegen generate --spec project.yml
xcodebuild -project MaiLensRemoteCapture.xcodeproj \
  -scheme MaiLensRemoteCapture \
  -destination 'generic/platform=iOS' \
  -configuration Release \
  -archivePath build/MaiLensRemoteCapture.xcarchive \
  CODE_SIGNING_ALLOWED=NO archive
```

App 内填写电脑 IPv4 和端口，点“连接电脑”，再点“开始发送”。连接状态、发送帧率、丢帧数和最近一次姿态时间戳都会显示出来。想直接录素材就点“开始录制”，它会自己开相机，不依赖电脑。推流仍然是 JPEG/TCP（为了低延迟调试），本地录制走 VideoToolbox H.264 全质量落盘。

仓库还附带 `codemagic.yaml`，会生成 `MaiLensRemoteCapture-unsigned.ipa`。它不签名，拿到 IPA 后可以继续用你的第三方工具签名。

## 协议

每个包都是大端序：

```text
4 bytes  magic = MLCP
1 byte   version = 1
1 byte   packet type (1 = frame, 2 = hello)
4 bytes  metadata JSON length
4 bytes  JPEG payload length
N bytes  UTF-8 metadata JSON
M bytes  JPEG bytes
```

帧 metadata 至少包含 `frame_id`、`timestamp`、`width`、`height` 和 `pose`。`pose` 里有 `timestamp`、`quaternion(x,y,z,w)`、`gravity`、`rotation_rate`、`user_acceleration`（去掉重力后的加速度，做前后移动补偿时用）。接收端按包内时间戳写入 `capture.jsonl`，所以后续处理可以复现每帧的姿态补偿。
