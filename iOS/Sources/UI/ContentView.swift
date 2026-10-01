import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var capture = CaptureController()
    @AppStorage("pcReceiverHost") private var host = ""
    @AppStorage("pcReceiverPort") private var portText = "8765"

    private var port: UInt16 { UInt16(portText) ?? 8765 }

    var body: some View {
        ZStack {
            Color(red: 0.035, green: 0.048, blue: 0.055).ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header
                    CameraPreview(session: capture.session)
                        .aspectRatio(9.0 / 16.0, contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 22))
                        .overlay(alignment: .topLeading) {
                            Label(capture.isRunning ? "0.5× 正在采集" : "预览待机",
                                  systemImage: capture.isRunning ? "camera.fill" : "camera")
                                .font(.system(size: 11, weight: .semibold))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 7)
                                .background(.black.opacity(0.6), in: Capsule())
                                .padding(12)
                        }

                    receiverSettings
                    statusPanel

                    Button(action: toggleCapture) {
                        Label(capture.isRunning ? "停止发送" : "连接电脑并开始发送",
                              systemImage: capture.isRunning ? "stop.fill" : "dot.radiowaves.left.and.right")
                            .font(.system(size: 15, weight: .bold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 15)
                            .foregroundStyle(capture.isRunning ? .white : .black)
                            .background(capture.isRunning ? Color.red : Color.mint,
                                        in: RoundedRectangle(cornerRadius: 15))
                    }
                    .disabled(!capture.isRunning && host.trimmingCharacters(in: .whitespaces).isEmpty)

                    Text("视频帧和该帧最近的陀螺仪姿态装在同一个数据包里。电脑负责矫正与稳定计算；手机端只采集和发送。")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.white.opacity(0.5))
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(18)
            }
        }
        .preferredColorScheme(.dark)
        .onDisappear { capture.disconnect() }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("MAI LENS · PC LAB")
                    .font(.system(size: 12, weight: .black, design: .rounded))
                    .tracking(2)
                    .foregroundStyle(Color.mint)
                Text("无线采集")
                    .font(.system(size: 26, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
            }
            Spacer()
            Circle()
                .fill(capture.isRunning ? Color.green : Color.gray)
                .frame(width: 9, height: 9)
            Text(capture.isRunning ? "发送中" : "未连接")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.white.opacity(0.8))
        }
    }

    private var receiverSettings: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("电脑接收地址")
                .font(.system(size: 16, weight: .bold))
            HStack(spacing: 10) {
                TextField("例如 192.168.1.23", text: $host)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.numbersAndPunctuation)
                    .textFieldStyle(.roundedBorder)
                TextField("端口", text: $portText)
                    .keyboardType(.numberPad)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 86)
            }
            Text("手机和电脑需连接同一个 Wi‑Fi。先在电脑运行接收程序，再开始发送。")
                .font(.system(size: 11))
                .foregroundStyle(.white.opacity(0.48))
        }
        .padding(16)
        .background(.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 20))
    }

    private var statusPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("连接状态")
                    .foregroundStyle(.white.opacity(0.55))
                Spacer()
                Text(capture.connectionState)
                    .foregroundStyle(capture.connectionState == "已连接" ? Color.mint : .white)
            }
            HStack {
                Text("发送帧率")
                    .foregroundStyle(.white.opacity(0.55))
                Spacer()
                Text(String(format: "%.1f fps", capture.framesPerSecond))
                    .foregroundStyle(.white)
            }
            HStack {
                Text("网络丢帧")
                    .foregroundStyle(.white.opacity(0.55))
                Spacer()
                Text("\(capture.droppedFrames)")
                    .foregroundStyle(capture.droppedFrames == 0 ? Color.mint : Color.orange)
            }
            if let error = capture.errorMessage {
                Text(error)
                    .font(.system(size: 11))
                    .foregroundStyle(.orange)
            }
        }
        .font(.system(size: 12, weight: .semibold, design: .monospaced))
        .padding(16)
        .background(.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 20))
    }

    private func toggleCapture() {
        if capture.isRunning {
            capture.disconnect()
        } else {
            capture.connect(host: host, port: port)
        }
    }
}

private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        if let connection = view.previewLayer.connection,
           connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.previewLayer.session = session
    }
}

private final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

