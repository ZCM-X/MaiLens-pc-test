import AVFoundation
import Combine
import CoreImage
import CoreMedia
import CoreVideo
import Foundation

final class CaptureController: NSObject, ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var connectionState = "已断开"
    @Published private(set) var framesPerSecond = 0.0
    @Published private(set) var droppedFrames: UInt64 = 0
    @Published private(set) var errorMessage: String?
    @Published private(set) var isRecording = false
    @Published private(set) var recordingStatus = SessionRecorder.Status()
    /// Bumped whenever a finished session becomes visible on disk.
    @Published private(set) var libraryRevision = 0

    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.mailens.pc-test.camera")
    private let outputQueue = DispatchQueue(label: "com.mailens.pc-test.frames", qos: .userInitiated)
    private let output = AVCaptureVideoDataOutput()
    private let sender = RemoteFrameSender()
    private let motion = MotionPoseProvider()
    private let recorder = SessionRecorder()
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let targetFrameRate: Int32 = 60
    private var configured = false
    private var lastSentTimestamp: Double = 0
    private var lastStatsTimestamp = Date.timeIntervalSinceReferenceDate
    private var statsFrameCount = 0

    override init() {
        super.init()
        sender.onState = { [weak self] in self?.connectionState = $0 }
        sender.onStatistics = { [weak self] _, dropped in
            DispatchQueue.main.async { [weak self] in self?.droppedFrames = dropped }
        }
        motion.onSample = { [weak self] pose in self?.recorder.log(pose: pose) }
        recorder.onStatus = { [weak self] status in self?.recordingStatus = status }
        recorder.onFinished = { [weak self] _ in
            self?.isRecording = false
            self?.libraryRevision += 1
        }
    }

    func connect(host: String, port: UInt16) {
        guard !host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            errorMessage = "请输入电脑的局域网 IP 地址。"
            return
        }
        errorMessage = nil
        motion.start()
        sender.connect(host: host, port: port)
        startCamera()
    }

    func disconnect() {
        stopRecording()
        stopCamera()
        motion.stop()
        sender.disconnect()
        DispatchQueue.main.async { [weak self] in
            self?.framesPerSecond = 0
        }
    }

    /// Records the raw camera without needing the PC.  Starting it also starts
    /// the camera when the stream is off, so a take can be captured anywhere.
    func startRecording() {
        guard !isRecording else { return }
        errorMessage = nil
        motion.start()
        guard recorder.start(nominalFPS: Double(targetFrameRate)) != nil else {
            errorMessage = "无法创建录制会话，请检查手机剩余存储空间。"
            return
        }
        recordingStatus = SessionRecorder.Status()
        isRecording = true
        startCamera()
    }

    func stopRecording() {
        guard isRecording else { return }
        isRecording = false
        recorder.stop()
    }

    private func startCamera() {
        sessionQueue.async { [weak self] in
            guard let self, !self.session.isRunning else { return }
            do {
                if !self.configured { try self.configure() }
                self.session.startRunning()
                DispatchQueue.main.async { self.isRunning = true }
            } catch {
                self.sender.disconnect()
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                    self.isRunning = false
                }
            }
        }
    }

    private func stopCamera() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
            DispatchQueue.main.async { self.isRunning = false }
        }
    }

    private func configure() throws {
        guard let camera = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back) else {
            throw CaptureError.ultraWideUnavailable
        }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = session.canSetSessionPreset(.hd1280x720) ? .hd1280x720 : .high
        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input), session.canAddOutput(output) else { throw CaptureError.cannotConfigure }
        session.addInput(input)
        configureFrameRate(for: camera)
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: outputQueue)
        session.addOutput(output)
        if let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
        configured = true
    }

    private func configureFrameRate(for camera: AVCaptureDevice) {
        let candidates = camera.formats.filter { format in
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let is720p = dimensions.width == 1280 && dimensions.height == 720
            let supportsTarget = format.videoSupportedFrameRateRanges.contains {
                $0.minFrameRate <= Double(targetFrameRate) && $0.maxFrameRate >= Double(targetFrameRate)
            }
            return is720p && supportsTarget
        }
        guard let format = candidates.first else { return }

        do {
            try camera.lockForConfiguration()
            defer { camera.unlockForConfiguration() }
            camera.activeFormat = format
            let duration = CMTime(value: 1, timescale: targetFrameRate)
            camera.activeVideoMinFrameDuration = duration
            camera.activeVideoMaxFrameDuration = duration
        } catch {
            // Keep the camera's default frame rate if this format cannot be locked.
        }
    }

    private func handle(_ sampleBuffer: CMSampleBuffer) {
        let presentation = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let timestamp = presentation.seconds
        guard timestamp.isFinite, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let pose = motion.nearest(to: timestamp)
        recorder.append(sampleBuffer: sampleBuffer, pose: pose)

        guard timestamp - lastSentTimestamp >= 1.0 / Double(targetFrameRate) else { return }
        lastSentTimestamp = timestamp

        let inputImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scale = min(1.0, 1280.0 / Double(max(CVPixelBufferGetWidth(pixelBuffer), CVPixelBufferGetHeight(pixelBuffer))))
        let scaled = scale < 1 ? inputImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale)) : inputImage
        let compressionKey = CIImageRepresentationOption(
            rawValue: kCGImageDestinationLossyCompressionQuality as String
        )
        guard let jpeg = ciContext.jpegRepresentation(of: scaled,
                                                      colorSpace: CGColorSpaceCreateDeviceRGB(),
                                                      options: [compressionKey: 0.84]) else { return }
        let width = Int(scaled.extent.width.rounded())
        let height = Int(scaled.extent.height.rounded())
        sender.send(jpeg: jpeg, timestamp: timestamp, width: width, height: height, pose: pose)
        statsFrameCount += 1
        let now = Date.timeIntervalSinceReferenceDate
        if now - lastStatsTimestamp >= 1 {
            let elapsed = max(now - lastStatsTimestamp, 0.001)
            DispatchQueue.main.async { [weak self] in self?.framesPerSecond = Double(self?.statsFrameCount ?? 0) / elapsed }
            statsFrameCount = 0
            lastStatsTimestamp = now
        }
    }

    private enum CaptureError: LocalizedError {
        case ultraWideUnavailable
        case cannotConfigure
        var errorDescription: String? {
            switch self {
            case .ultraWideUnavailable: return "未找到 0.5× 超广角摄像头，请在 iPhone 真机运行。"
            case .cannotConfigure: return "无法配置相机视频采集。"
            }
        }
    }
}

extension CaptureController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        handle(sampleBuffer)
    }
}
