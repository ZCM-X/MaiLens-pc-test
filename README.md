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
  ├─ 内屏：青色掩膜 H70–115 S>60 V>40 → 最大圆盘连通域 → 抗离群 fitEllipse
  ├─ 外键：紫色掩膜 H125–145 S>80 V>60 → 连通域质心按“内屏自身坐标”的
  │        0.85–1.95 倍半径带过滤（挡掉立绘碎片和别的机台）→ 圆拟合
  └─ 双环联合拟合：一个仿射同时满足“内屏是圆 + 外键与它同心”
        │
        ▼
  两条时间尺度分别平滑：中心跟得紧（--smooth-sigma，默认 1），
  压扁和缩放放得慢（--shape-sigma，默认 14）
        │
        ▼
  单次仿射变换：内屏中心 → 画面正中，内屏直径 → 固定像素直径
```

内屏的**长轴**当作距离信号（它在俯仰时不会像短轴那样被压缩），所以手机前后移动时画面会自动反向缩放，机台在屏幕上的大小保持不变；缩放只放大不缩小，永远不会露出黑边。

### 为什么位置和形状要用两条时间尺度

两件事经常被混在一起说，但它们对滤波的要求正好相反：

* **机台在画面里的位置**必须跟着测量走。位置的偏差就是“机台没定在中间”，任何平滑都会立刻变成看得见的漂移。所以 `--smooth-sigma` 默认只有 0.5，平滑掉的只是逐帧独立的检测抖动。想更狠可以取 0：这时机台会**逐像素**钉在正中，代价是检测噪声全部由背景承担。
* **机台的压扁和远近**变化得极慢，却带着很大的检测噪声。这里用零相位高斯（对称核，无时间滞后）压到 `--shape-sigma` 默认 14，机台就不会一帧一帧地“呼吸”和摇头。残差只有测量本身那一部分。

早期版本两者共用 `--smooth-sigma 3`，位置因此每帧抖动约 9 px、横滚每帧抖 0.29°；拆开之后（1080 宽输出、`IMG_8901.MOV` 801 帧）：

```text
                              旧（单椭圆 + σ3）   新（双环 + 分开平滑）
机台中心偏离正中（中位）        12.0 px             2.9 px
每帧横滚抖动                    0.285°              0.03°
外键半径离散度（中位）          12 %                3 %
四条边距 |L-R| / |T-B|（中位）  59 / 88 px          12 / 19 px
```

`--smooth-sigma` 扫一遍（1080 宽输出，机台偏离正中的中位数）：

```text
0     0.00 px   机台逐像素钉死，背景承担全部检测抖动
0.5   2.93 px   默认
1     6.93 px
3    22.34 px   背景最稳，机台会自己晃
```

取样窗口越出画面的帧 0 / 801，自动补偿的缩放范围 1.29–1.65。`--margin`（默认 1.4）会先按更宽的鱼眼视角（`crop / margin`）展开一张 1.4 倍的工作画布，锁完再从中间裁出成片；`need` 会在机台本来就贴近画幅边缘时自动抬高缩放兜底。

### `--ring-weight`：按钮到底要多圆

双环联合拟合把 72 个内屏轮廓点和 8 个按键质心放进同一个仿射里一起解，但这两组数据本身给的答案是**不一致**的：在这段素材里内屏椭圆的短长轴之比中位 0.954，外键质心拟合出来是 0.918，而且外键只有 8 个点，椭圆朝向的标准差高达 36°。

`--ring-weight`（默认 0.5）就是这两者的权重，实测：

```text
权重    外键半径离散度   四条边距 |L-R| / |T-B|   内屏轮廓离散度
0.15    3.7 %           13 / 17 px              7.6 %
0.50    5.0 %           16 / 25 px              8.0 %   ← 默认
2.00    9.1 %           32 / 73 px              3.0 %
```

调小 → 外键更圆、边距更接近相等，代价是内屏会被拉得略微不圆；调大则相反。想要博主那种“四条边距相等”的观感就往小了调。

`--margin`（默认 1.4）会先按更宽的鱼眼视角（`crop / margin`）展开一张 1.4 倍的工作画布，锁完再从中间裁出成片。`IMG_8901.MOV` 里有 117 帧机台本来靠近原画幅边缘，直接平移会采到画外、`BORDER_REFLECT101` 补出镜像的“翅膀”；多这一圈余量之后越界帧变成 0，而且鱼眼在这一圈仍然有像素（margin 1.4 和 1.6 的黑边都是 0.00%）。如果哪天机台偏得更狠，`need` 会自动抬高缩放兜底，代价是那一小段画面会轻微推近。

## 视角拉正（--rectify）

上一步只做了平移和缩放，机台在画面里还是斜的。这一步把它拉成正视：内屏本来就是机台正面上的一**个圆**，它在图像里是一个椭圆，长短轴之比就是倾斜角的余弦。把这个椭圆还原成正圆，倾斜就没了，而且外圈按键会同时变成同心圆——也就是博主说的“左右、上下四条边距相等”，区别是这里靠几何直接解出来，不用在像素边距上反复试。

关于要不要连透视一起解：如果内屏和外键真是平面上的一对同心圆，它们的图像就是一对圆锥曲线，两者的凸组合（铅笔）里应当有唯一一个秩 1 成员，而那个成员就是机台平面的**消失线**。用 `work/_perspective.py` 在 `IMG_8901.MOV` 上重算，结果并不干净：

```text
rank-1 residual σ2/σ1        2.3 … 8（早先一次用未过滤按键得到的 3e-7 是退化拟合）
推出来的消失线朝向            92° … 152°，逐帧乱跳
```

秩 1 残差只有个位数就说明这对锥线并不共享那条极限成员。最可能的原因是：画面里看到的紫色并不是按键在机台平面上的投影，而是按键侧面那圈**裙边**——它本来就是偏出机台平面的三维面，图像不再是平面上的圆锥曲线。所以 `--rectify` 只还原内屏椭圆（这部分几何是干净的），不去碰外键锥线。

```text
原始内屏 短轴/长轴    0.837 – 0.980（中位 0.912，对应最大约 33° 的倾斜）
拉正后内屏            所有帧都是正圆
额外自动缩放          余量 1.4 时 ≤12%，只有 23/300 帧触发
```

`--rectify 0` 回到纯平移缩放的相似变换，`0..1` 之间是连续过渡（混合的是半轴的**倒数**，线性部分全程保持正定，不会在中间塌掉）。

```powershell
# 1:1 输出 2160×3840
.\.venv\Scripts\python.exe pc\machine_lock.py "C:\Users\93543\Downloads\IMG_8901.MOV" `
  --output "C:\Users\93543\Downloads\IMG_8901_双环锁定_v5.mp4" `
  --output-scale 1.0

# 快速预览 + 12 格拼图 + 逐帧诊断数据
.\.venv\Scripts\python.exe pc\machine_lock.py "C:\Users\93543\Downloads\IMG_8901.MOV" `
  --output work\locked.mp4 --output-scale 0.5 `
  --contact work\locked_contact.jpg --trace work\locked.jsonl --draw
```

跑完会在结尾打印这一条的质量分：`cabinet off centre`（机台偏了多少像素）、`button ring cv`（外键有多圆）、`four-side gap |L-R| / |T-B|`（博主那套四条边距）。第二次跑同一段素材会直接读 `IMG_8901.MOV.measure.json` 缓存，只要 `--measure-scale / --margin / --ring-weight` 没变就跳过重新测量。

`--draw` 会在画面正中画绿色十字，并画一个红圈标出机台实际落点，两者重合就说明锁定成立；左上角 `lock err` 是这一帧的残差像素。`--trace` 里的 `resid_x / resid_y` 是同一件事的逐帧数值。

`--roll-gain`（默认 0）打开后会用按键环的相位抵消手机横滚，相位每 45° 一个周期，管线里先解缠再平滑。这一路信号噪声约 2.3°/帧，比中心弱，所以默认不开。

`--slide-limit`（默认 0.06）决定机台最多可以离开画面中心多远，单位是交付画面宽度的比例。手机甩到鱼眼边缘时裁切框会需要画面外的像素，旧的做法直接放大整幅图，观感就是机台朝人扑过来；现在裁切框先平移（最多 6% 宽）把这段顶过去，尺寸和到机台的距离保持不变，只有平移盖不住的余量才触发缩放。缩放判断会向前看半秒，所以是提前慢慢缩出去、而不是到边缘突然一顿。设成 0 就回到只缩放的行为。

## 量八个装饰框（tools/measure_button_frames.py）

外圈八个装饰框是同一个零件，量它可以拿到机台的尺寸参照，也可以验证"模型框到底歪在哪"。工具用极坐标剖面找每个框的内缘和外缘：紫色框的**蓝减绿**很高，中间的灰环和框外那条白色高光是中性（蓝减绿约为零），所以从外往里扫就能同时找到两条边界，不用在一个有强光照渐变的画面上定阈值。标注工具画的红色数字和游戏里的绿色打击特效都压在框上，会让蓝减绿测试误判，所以只有中性的采样才算边界。

```powershell
.\.venv\Scripts\python.exe tools\measure_button_frames.py "F:\iloader_work\截图.png" `
  --output "C:\Users\93543\Downloads\按键装饰框测量.png" --screen-radius 197.4
```

不给 `--centre` 会自动搜索机台中心（先粗扫一圈，再把八个框心拟合到圆上）。`--screen-radius` 是内屏半径（屏幕那圈白点），给了就顺带输出各尺寸相对内屏的比值。

输出会按你标的 1–8 编号（1 右上、顺时针、8 正上）列每个框的角度跨度、切向宽、内缘/外缘半径、径向高和面积，然后把每一列的离散度拆成两部分：**跟着角度转一圈的那一项（tilt）**是照片自己的透视/倾斜，**扣掉它剩下的（leftover）**才是八个框真正的大小差异。在同一张正面照上实测，宽度 5.75% 的离散里 4.35% 是 tilt，剩下 0.80%；面积 14.09% 里 11.65% 是 tilt，剩下 1.65%。也就是八个框实际一样大，看到的差别基本都是拍摄角度。

自动搜中心只在这张截图那种干净画面上可靠。手机原图上机台亮着灯，按键外面那圈发光**也是紫色**，蓝减绿并不落回中性，工具就找不到外缘（`IMG_8839` 上只认出 4 个框）。要量原图请给 `--centre`，或者直接看下面 `--ring-round` 的逐帧读数。

## 拉回正面（`--ring-round`）

八个装饰框是同一个零件重复八次，所以**正对**的机台上，它们到屏幕中心的距离八个方向都相等，屏幕边到按键圈的四条间距也相等。这就是“没有畸变”的可测量定义：把每个槽位的 **框心半径 / 内屏半径** 量出来，绕圈的任何起伏都是残留畸变。内屏是正圆**不能**保证这一点——博主说的正是这个。

`--rectify` 先把内屏椭圆拉成正圆（去掉倾斜）。剩下的偏差是**跟方向有关**的：屏幕圆了，外面那圈不圆；或者圆了但和内屏不同心。八个槽位刚好把它量出来：

```text
每一帧
  ├─ 内屏：青色掩膜 → 椭圆 → --rectify 拉成正圆
  ├─ 外键：紫色掩膜 → 八个槽位的框心半径 / 内屏半径
  ├─ ρ(θ)：用 cos/sin 谐波到三阶最小二乘拟合绕圈的半径（八点七个系数）
  ├─ ρ(θ) 再沿时间做零相位高斯（--ring-round-sigma，默认 6 帧）
  └─ 重映射 r_in = r_out · (1 + ramp(r_out) · (ρ(θ) / ρ₀ − 1))
```

`ramp` 在屏幕边是 0、在按键圈是 1：屏幕边一个像素都不动，每个方向的按键圈都落到同一个 `ρ₀`，两者的间距自然四条相等。修正量夹在 ±18%，量崩的帧直接退回不做（系数不是有限值就跳过）。

`ρ₀` 默认 1.23，来自 `H:\IMG_8839(20261002-154029).JPG` 那张正对参考图——而且是用**同一套检测器**量的：青色游戏区当内屏、按键颜色的质心当框心。换一个分母就差 9%（对着屏幕那圈白点手量是 1.365），所以基准必须和检测器配套，不能混用。

`IMG_8901.MOV` 801 帧、1080 宽输出：

```text
                          关闭        打开
八个槽位半径离散度         2.71%       1.89%
四条间距 |L-R|             26.4px      13.7px
        |T-B|              35.7px      13.5px
```

```powershell
.\.venv\Scripts\python.exe pc\machine_lock.py "C:\Users\93543\Downloads\IMG_8901.MOV" `
  --output "C:\Users\93543\Downloads\IMG_8901_环形矫正_v9.mp4" --output-scale 1.0 `
  --ring-round 1.0
```

剩下的 1.89% 是**逐帧检测噪声**：八个框的质心各有几个像素的抖动，而三阶拟合七个系数只吃八个采样，平均不掉。想再往下压得先把框心量准，不是把拟合加阶。

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

### 四边间距标注（`--gaps`）

判断一张图是不是正对机台，看的是外圈按键和内屏之间的四条间距：正对时左右相等、上下相等，机台实体上四下都是 75 mm。加上 `--gaps` 会多出一层间距标注：

```powershell
.\.venv\Scripts\python.exe tools\annotate_geometry.py `
  --input "C:\Users\93543\Downloads\maimoller训练" `
  --output datasets\maimoller-gaps --preset geometry --every 1 --gaps
```

按 `g` 进入间距模式，然后按 左内屏→左外键→右内屏→右外键→上→上→下→下 的顺序点 8 个点（`z` 撤销一个点，`x` 清空）。已经有 `outer_buttons` / `inner_screen` 两个框时，直接按 `a` 从两个框推出来，不用手点。每张的读数存到 `<output>/gaps/frame-NNNNNN.json`，含四边像素、按目标归一的毫米值、四边离散度和 `dead_on` 判定；画面顶端会实时显示 `L … R … T … B …`，哪条边偏了哪条就变红。

归一化的做法是：先算四条边的平均像素，再整体缩放到目标值（默认 75），所以四个读数都接近 75 就是正对。注意这只是标注侧的口径，不等于机台锁定已经做到；锁定是否达标要用 `--trace` 里的 `four-side gap` 和交付视频上的实测来验收。

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
