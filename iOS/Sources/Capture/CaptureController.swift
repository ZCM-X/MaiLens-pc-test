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

    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.mailens.pc-test.camera")
    private let outputQueue = DispatchQueue(label: "com.mailens.pc-test.frames", qos: .userInitiated)
    private let output = AVCaptureVideoDataOutput()
    private let sender = RemoteFrameSender()
    private let motion = MotionPoseProvider()
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private var configured = false
    private var lastSentTimestamp: Double = 0
    private var lastStatsTimestamp = Date.timeIntervalSinceReferenceDate
    private var statsFrameCount = 0

    override init() {
        super.init()
        sender.onState = { [weak self] in self?.connectionState = $0 }
        sender.onStatistics = { dropped in
            DispatchQueue.main.async { [weak self] in self?.droppedFrames = dropped }
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
        stopCamera()
        motion.stop()
        sender.disconnect()
        DispatchQueue.main.async { [weak self] in
            self?.framesPerSecond = 0
        }
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
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: outputQueue)
        session.addOutput(output)
        if let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
        configured = true
    }

    private func handle(_ sampleBuffer: CMSampleBuffer) {
        let presentation = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let timestamp = presentation.seconds
        guard timestamp.isFinite, timestamp - lastSentTimestamp >= 1.0 / 15.0,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastSentTimestamp = timestamp

        let inputImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scale = min(1.0, 1280.0 / Double(max(CVPixelBufferGetWidth(pixelBuffer), CVPixelBufferGetHeight(pixelBuffer))))
        let scaled = scale < 1 ? inputImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale)) : inputImage
        guard let jpeg = ciContext.jpegRepresentation(of: scaled,
                                                      colorSpace: CGColorSpaceCreateDeviceRGB(),
                                                      options: [.lossyCompressionQuality: 0.84]) else { return }
        let pose = motion.nearest(to: timestamp)
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
