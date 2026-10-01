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

处理器默认载入当前 MaiLens 的鱼眼参数（中心 `0.501753869, 0.499423644`、`k1=0.0893163`、`k2=-0.0174637`、输出视场角 `106.4583°`），所以电脑生成的结果会先做鱼眼反变换，再做姿态稳定。参数可以直接用 `--center-x`、`--center-y`、`--k1`、`--k2` 和 `--fov` 覆盖。

如果要让实时模式同时做机台外框/内屏锁定，直接使用仓库内的 ONNX 模型。电脑端优先用 OpenCV DNN 推理，不需要安装 PyTorch：

```powershell
.\.venv\Scripts\python.exe pc\pc_receiver.py --port 8765 `
  --process-live --preview --machine-lock --debug
```

如果要使用 `.pt` 等非 ONNX 模型，才需要额外安装 `ultralytics` 和对应的 PyTorch 运行环境。

实时检测默认每 4 帧运行一次，模型在原始鱼眼帧上检测，再映射到鱼眼矫正后的输出坐标；中间帧使用稀疏光流跟踪。锁定器会分别保留外框与内屏：外框中心负责把整台机台送到画面中心，内屏尺寸只负责裁切缩放。调试画面用黄色框标外框、绿色框标内屏，白色十字是输出中心；`--lock-fill 0.64` 可以调整内屏在画面中的大小。

## 标注自己的机台数据

当前内置模型经常框到内屏而不是机台主体，因此建议先用自己的鱼眼画面标一些真实样本，再训练新的模型。运行标注工具：

```powershell
.\.venv\Scripts\python.exe tools\annotate_geometry.py `
  --input sessions\你的会话目录 --output datasets\geometry `
  --every 6 --val-every 10
```

`--input` 可以是实时会话目录、视频、单张图或图片目录。视频默认每 6 帧取一帧，避免连续相似帧占满数据；`--max-frames 120` 可限制本次数量。窗口里拖动矩形，按 `1` 标完整机台可见外框，按 `2` 标实际游戏内屏；按 `s` 保存，`n`/空格下一张，`p` 上一张，`x` 删除当前类别框，`r` 清除此图，`q` 退出。下一次以相同 `--output` 打开会继续已有标注。

如果要同时标谱面的八个按键，打开第三类：

```powershell
.\.venv\Scripts\python.exe tools\annotate_geometry.py `
  --input sessions\你的会话目录 --output datasets\geometry-buttons `
  --every 6 --val-every 10 --with-buttons
```

启用后按 `3`，在同一帧依次框出最多 8 个按键；按 `z` 撤销最后一个按键，`x` 清空全部按键。按键少于 8 个时只标清楚可见的按键，不要为了凑数乱框。

训练时只标这两个目标：

- `outer_frame`：整台机台/机柜在画面里可见的完整外轮廓，包括屏幕外的边框和机身；不要把地面、背景或旁边设备框进去。
- `inner_screen`：显示游戏内容的屏幕矩形，沿屏幕玻璃/显示区域的边缘框；不要标八个判定星星、按钮、反光或 UI 元素。
- `button`：每个谱面按键的可见区域，一个按键一个框，最多 8 个；它只用于后续按键/谱面识别，不参与机台居中或裁切缩放。

不要为了每张图都凑两个框而猜测。某个目标被遮挡或出画时，只标清楚可见的那个；两类都看不清就跳过样本。工具输出标准 YOLO `images/{train,val}`、`labels/{train,val}` 和 `dataset.yaml`。建议先标至少 100 张，覆盖远近、左右偏移、倾斜、遮挡和曝光变化，再按 Ultralytics YOLO 文档训练并导出 ONNX。

如果要把已有的 `frame-geometry-yolo11n-v2.onnx` 用在离线处理上：

```powershell
.\.venv\Scripts\python.exe pc\process_session.py sessions\20261001-153012 `
  --model models\frame-geometry-yolo11n-v2.onnx --output processed-machine.mp4 --debug
```

检测器锁定只使用 `outer_frame` 和 `inner_screen` 两类几何；即使模型还输出 `button`，也不会把按键当成机台外框。当前电脑算法保留了调试输出：`debug.jsonl` 中有姿态延迟、中心、缩放和检测框，便于先在电脑上调曲线。

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
