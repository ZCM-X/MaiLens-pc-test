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
```

视频和姿态被放在同一个 TCP 数据包里，电脑端不需要猜测两条流的对应关系。发送端使用有限缓冲，只保留最新待发送帧；电脑或网络变慢时会丢弃旧帧，不会把延迟越积越大。

## 电脑端启动

在 Windows PowerShell 中：

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

如果要让实时模式同时做机台外框/内屏锁定，直接使用仓库内的 ONNX 模型。当前实时默认使用 v2，因为它在现有视频上能连续检测内屏；v3 先保留作重新训练实验。电脑端优先用 OpenCV DNN 推理，不需要安装 PyTorch：

```powershell
.\.venv\Scripts\python.exe pc\pc_receiver.py --port 8765 `
  --process-live --preview --machine-lock --debug
```

如果要使用 `.pt` 等非 ONNX 模型，才需要额外安装 `ultralytics` 和对应的 PyTorch 运行环境。

实时检测默认每 12 帧运行一次，模型在原始鱼眼帧上检测，再映射到鱼眼矫正后的输出坐标。首次得到有效 `inner_screen` 后，处理器会在内屏纹理内取特征点，用 LK 光流和 RANSAC `findHomography` 估计每帧的平面运动，把当前内屏反向映射到一个固定的中央矩形；这个变换同时补偿横移、横滚、倾斜和中等幅度的前后移动，不再把检测框中心和面积当作唯一稳定信号。特征点暂时不足时保持上一张可靠的单应矩阵，超过丢失时限才回到搜索状态。外框仍用于验证和调试，不会把锁定锚点拉到机台上沿。

调试画面用黄色框标外圈按键区域、绿色框标内屏，白色十字是输出中心；`processed.jsonl` / `debug.jsonl` 会额外记录 `plane_lock`、内点数、内点比例和重投影误差。`--lock-fill 0.64` 调整首次锁定时内屏在画面中的大小。当前版本仍需要内屏在鱼眼有效视野内且保留可跟踪纹理；完全无纹理或被遮挡时会安全保持上一帧，不会用错误框跳走。

## 标注自己的机台数据

这里要训练两套模型，两个数据集不要混在一起：

1. 几何模型：`outer_buttons` + `inner_screen`，负责识别外圈按键区域、内屏以及两者的距离关系，用于机台居中和裁切。
2. 按键模型：`button` + `inner_screen`，负责在内屏坐标系中识别谱面按键；它的结果不参与机台居中。

当前内置模型经常框到内屏而不是外圈按键区域，因此建议先用自己的鱼眼照片标这两套数据，再训练新的模型。运行标注工具：

```powershell
.\.venv\Scripts\python.exe tools\annotate_geometry.py `
  --input sessions\你的会话目录 --output datasets\geometry `
  --preset geometry --every 6 --val-every 10
```

`--input` 可以是实时会话目录、视频、单张图或图片目录。视频默认每 6 帧取一帧，避免连续相似帧占满数据；`--max-frames 120` 可限制本次数量。窗口里用 `1`、`2` 切换类别，拖动矩形，按 `s` 保存，`n`/空格下一张，`p` 上一张，`x` 删除当前类别框，`r` 清除此图，`q` 退出。下一次以相同 `--output` 打开会继续已有标注。

iPhone 的 `.HEIC/.HEIF` 照片也可以直接读取。照片目录建议每张都看，所以把 `--every` 设为 `1`，例如：

```powershell
.\.venv\Scripts\python.exe tools\annotate_geometry.py `
  --input "C:\Users\93543\Downloads\maimoller训练" `
  --output datasets\maimoller-geometry --preset geometry --every 1
```

几何模型按截图中的方式标：`outer_buttons` 只画一个整体框，覆盖外圈 8 个实体按键和它们所在的环形区域；不要拆成 8 个小框。`inner_screen` 沿圆形游戏屏幕边缘画一个整体矩形。

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

当前 `models/frame-geometry-yolo11n-v3.onnx` 就是用 `maimoller-geometry` 数据集训练并导出的版本。验证集只有 5 张图，指标只能说明标注闭环和推理类别正常，后续还要用不同距离、角度和遮挡的手机视频验收。

如果要把已有的 `frame-geometry-yolo11n-v2.onnx` 用在离线处理上：

```powershell
.\.venv\Scripts\python.exe pc\process_session.py sessions\20261001-153012 `
  --model models\frame-geometry-yolo11n-v2.onnx --output processed-machine.mp4 --debug
```

检测器锁定只使用几何模型的 `outer_buttons` 和 `inner_screen`；按键模型的结果不会把机台锁到某个谱面元素。当前电脑算法保留了调试输出：`debug.jsonl` 中有姿态延迟、中心、缩放和检测框，便于先在电脑上调曲线。

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

App 内填写电脑 IPv4 和端口，点“连接电脑”，再点“开始发送”。连接状态、发送帧率、丢帧数和最近一次姿态时间戳都会显示出来。这个实验版本先用 JPEG/TCP 确认算法和同步关系；链路稳定后再把视频编码替换为 VideoToolbox H.264/HEVC。

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

帧 metadata 至少包含 `frame_id`、`timestamp`、`width`、`height` 和 `pose`。`pose` 里有 `timestamp`、`quaternion(x,y,z,w)`、`gravity`、`rotation_rate`。接收端按包内时间戳写入 `capture.jsonl`，所以后续处理可以复现每帧的姿态补偿。
