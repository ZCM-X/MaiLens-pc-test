import SwiftUI
import AVFoundation
import UIKit

struct ContentView: View {
    @StateObject private var capture = CaptureController()
    @AppStorage("pcReceiverHost") private var host = ""
    @AppStorage("pcReceiverPort") private var portText = "8765"
    @State private var recordings: [SessionRecorder.Recording] = []
    @State private var shareItems: [Any] = []
    @State private var isSharing = false
    @State private var pendingDelete: SessionRecorder.Recording?

    private var port: UInt16 { UInt16(portText) ?? 8765 }

    var body: some View {
        ZStack {
            Color(red: 0.035, green: 0.048, blue: 0.055).ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header
                    preview
                    recordingPanel
                    receiverSettings
                    statusPanel
                    streamButton
                    recordingsPanel
                    footer
                }
                .padding(18)
            }
        }
        .preferredColorScheme(.dark)
        .sheet(isPresented: $isSharing) {
            ActivityView(items: shareItems)
        }
        .confirmationDialog("删除这段录制？",
                            isPresented: Binding(get: { pendingDelete != nil },
                                                 set: { if !$0 { pendingDelete = nil } }),
                            titleVisibility: .visible) {
            Button("删除", role: .destructive) {
                if let target = pendingDelete { delete(target) }
            }
            Button("取消", role: .cancel) { pendingDelete = nil }
        } message: {
            Text(pendingDelete.map { "\($0.name) 的录像、姿态日志和元数据会一起删掉。" } ?? "")
        }
        .onAppear { refreshRecordings() }
        .onChange(of: capture.libraryRevision) { _, _ in refreshRecordings() }
        .onChange(of: capture.isRecording) { _, _ in refreshRecordings() }
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

    private var preview: some View {
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
            .overlay(alignment: .topTrailing) {
                if capture.isRecording {
                    Label("REC", systemImage: "record.circle.fill")
                        .font(.system(size: 11, weight: .black))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .foregroundStyle(Color.red)
                        .background(.black.opacity(0.6), in: Capsule())
                        .padding(12)
                }
            }
    }

    private var recordingPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("本地录制")
                    .font(.system(size: 16, weight: .bold))
                Spacer()
                Text(capture.isRecording ? "写入中" : "待机")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(capture.isRecording ? Color.red : .white.opacity(0.5))
            }

            Button(action: toggleRecording) {
                Label(capture.isRecording ? "停止录制" : "开始录制",
                      systemImage: capture.isRecording ? "stop.circle.fill" : "record.circle")
                    .font(.system(size: 15, weight: .bold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 15)
                    .foregroundStyle(capture.isRecording ? .white : .black)
                    .background(capture.isRecording ? Color.red : Color.orange,
                                in: RoundedRectangle(cornerRadius: 15))
            }

            HStack(spacing: 10) {
                statTile("帧数", "\(capture.recordingStatus.frameCount)")
                statTile("时长", String(format: "%.1f s", capture.recordingStatus.duration))
                statTile("体积", ByteCountFormatter.string(fromByteCount: capture.recordingStatus.bytes,
                                                          countStyle: .file))
                statTile("丢帧", "\(capture.recordingStatus.droppedFrames)")
            }

            HStack(spacing: 8) {
                Text("姿态样本")
                    .foregroundStyle(.white.opacity(0.45))
                Spacer()
                Text("\(capture.recordingStatus.poseSampleCount)")
                    .foregroundStyle(.white)
                if capture.recordingStatus.width > 0 {
                    Text("·")
                        .foregroundStyle(.white.opacity(0.3))
                    Text("\(capture.recordingStatus.width)×\(capture.recordingStatus.height)")
                        .foregroundStyle(.white.opacity(0.7))
                }
            }
            .font(.system(size: 11, weight: .semibold, design: .monospaced))

            if let upload = uploadLine {
                Text(upload.text)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(upload.color)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
        .background(.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 20))
    }

    private var uploadLine: (text: String, color: Color)? {
        switch capture.uploadState {
        case .idle:
            return nil
        case .sending(let progress):
            return (String(format: "正在发送到电脑 %.0f%%", progress * 100), .mint)
        case .finished(let bytes):
            let size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
            return ("已把 \(size) 发到电脑，接着在电脑跑 import_phone_session.py。", .mint)
        case .failed(let message):
            return (message, .orange)
        }
    }

    private func statTile(_ title: String, _ value: String) -> some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.system(size: 13, weight: .bold, design: .monospaced))
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
            Text(title)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.white.opacity(0.5))
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 9)
        .background(.black.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))
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
            Text("手机和电脑需连接同一个 Wi‑Fi。端口填推流端口（默认 8765）；录制推送会自动用 8766，也就是推流端口 +1，不要把它填到这里。")
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

    private var streamButton: some View {
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
    }

    private var recordingsPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("已录会话")
                    .font(.system(size: 16, weight: .bold))
                Spacer()
                Text("\(recordings.count)")
                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.6))
            }

            if recordings.isEmpty {
                Text("还没有录制。每次录制按时间生成一个文件夹，里面是 video.mp4、姿态日志和元数据；可以用上面的「导出」按钮存到「文件」，也可以用爱思助手直接拷到电脑。")
                    .font(.system(size: 11))
                    .foregroundStyle(.white.opacity(0.48))
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                ForEach(recordings) { recording in
                    recordingRow(recording)
                }
            }
        }
        .padding(16)
        .background(.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 20))
    }

    private func recordingRow(_ recording: SessionRecorder.Recording) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(recording.name)
                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                    .foregroundStyle(.white)
                Spacer()
                Text(ByteCountFormatter.string(fromByteCount: recording.bytes, countStyle: .file))
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.6))
            }

            Text(detailLine(for: recording))
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(.white.opacity(0.45))

            HStack(spacing: 8) {
                Button { capture.sendToPC(recording, host: host, port: port) } label: {
                    Label("发到电脑", systemImage: "wifi")
                        .font(.system(size: 12, weight: .semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 9)
                        .foregroundStyle(Color.mint)
                        .background(Color.mint.opacity(0.18), in: RoundedRectangle(cornerRadius: 10))
                }
                Button { share(recording) } label: {
                    Label("导出", systemImage: "square.and.arrow.up")
                        .font(.system(size: 12, weight: .semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 9)
                        .foregroundStyle(.white.opacity(0.85))
                        .background(.white.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
                }
                Button { pendingDelete = recording } label: {
                    Label("删除", systemImage: "trash")
                        .font(.system(size: 12, weight: .semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 9)
                        .foregroundStyle(Color.red)
                        .background(Color.red.opacity(0.16), in: RoundedRectangle(cornerRadius: 10))
                }
            }
        }
        .padding(12)
        .background(.black.opacity(0.22), in: RoundedRectangle(cornerRadius: 12))
    }

    private var footer: some View {
        Text("推流用来实时调试；本地录制把原始 60fps 画面和 120Hz 姿态写进手机，不经过 JPEG。录完点“发到电脑”走局域网送到电脑，没连电脑也能先存着，之后整段拷过去离线跑稳定和机台锁定。")
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(.white.opacity(0.5))
            .fixedSize(horizontal: false, vertical: true)
    }

    private func detailLine(for recording: SessionRecorder.Recording) -> String {
        var parts = [String(format: "%.1f s", recording.duration), "\(recording.frameCount) 帧"]
        if recording.width > 0 {
            parts.append("\(recording.width)×\(recording.height)")
        }
        parts.append("姿态 \(recording.poseSampleCount)")
        return parts.joined(separator: "  ·  ")
    }

    private func toggleCapture() {
        if capture.isRunning {
            capture.disconnect()
        } else {
            capture.connect(host: host, port: port)
        }
    }

    private func toggleRecording() {
        if capture.isRecording {
            capture.stopRecording()
        } else {
            capture.startRecording()
        }
    }

    private func refreshRecordings() {
        recordings = SessionRecorder.recordings()
    }

    private func share(_ recording: SessionRecorder.Recording) {
        let files = recording.files
        guard !files.isEmpty else { return }
        shareItems = files
        isSharing = true
    }

    private func delete(_ recording: SessionRecorder.Recording) {
        try? SessionRecorder.delete(recording)
        pendingDelete = nil
        refreshRecordings()
    }
}

private struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
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
