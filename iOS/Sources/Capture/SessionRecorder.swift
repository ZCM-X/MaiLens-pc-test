import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

/// Records what the camera really delivers to the phone: one H.264 file plus
/// the full-rate sensor log on the same clock.
///
/// The live stream is deliberately lossy and rate limited, so it cannot be
/// re-processed offline.  This recorder keeps the raw material the PC side
/// needs to re-run the stabiliser and the machine lock on a real session.
final class SessionRecorder {
    struct Status {
        var frameCount = 0
        var poseSampleCount = 0
        var droppedFrames = 0
        var duration: Double = 0
        var bytes: Int64 = 0
        var width = 0
        var height = 0
    }

    struct Manifest: Codable {
        var source = "MaiLensRemoteCapture"
        var version = 1
        var created: String
        var clock = "mach_absolute_time"
        var wallClockStart: Double
        var video: String
        var metadata: String
        var poseLog: String
        var width: Int
        var height: Int
        var nominalFPS: Double
        var frameCount: Int
        var poseSampleCount: Int
        var duration: Double
        var droppedFrames: Int
        var wallClockDuration: Double

        enum CodingKeys: String, CodingKey {
            case source, version, created, clock, video, metadata, width, height, duration
            case wallClockStart = "wall_clock_start"
            case poseLog = "pose_log"
            case nominalFPS = "nominal_fps"
            case frameCount = "frame_count"
            case poseSampleCount = "pose_sample_count"
            case droppedFrames = "dropped_frames"
            case wallClockDuration = "wall_clock_duration_s"
        }
    }

    struct Recording: Identifiable, Hashable {
        let directory: URL
        let createdAt: Date
        let bytes: Int64
        let frameCount: Int
        let poseSampleCount: Int
        let duration: Double
        let width: Int
        let height: Int

        var id: URL { directory }
        var name: String { directory.lastPathComponent }

        /// Everything that belongs to the session, in the order the PC needs it.
        var files: [URL] {
            ["video.mp4", "capture.jsonl", "pose.jsonl", "session.json"].compactMap { name in
                let url = directory.appendingPathComponent(name)
                return FileManager.default.fileExists(atPath: url.path) ? url : nil
            }
        }

        init(directory: URL,
             createdAt: Date,
             bytes: Int64,
             frameCount: Int,
             poseSampleCount: Int,
             duration: Double,
             width: Int,
             height: Int) {
            self.directory = directory
            self.createdAt = createdAt
            self.bytes = bytes
            self.frameCount = frameCount
            self.poseSampleCount = poseSampleCount
            self.duration = duration
            self.width = width
            self.height = height
        }

        init(directory: URL, manifest: Manifest) {
            self.init(directory: directory,
                      createdAt: SessionRecorder.isoFormatter.date(from: manifest.created) ?? Date(),
                      bytes: SessionRecorder.size(of: directory.appendingPathComponent(manifest.video)),
                      frameCount: manifest.frameCount,
                      poseSampleCount: manifest.poseSampleCount,
                      duration: manifest.duration,
                      width: manifest.width,
                      height: manifest.height)
        }
    }

    static let folderName = "MaiLensSessions"

    /// Called on the main thread at most a few times per second.
    var onStatus: ((Status) -> Void)?
    /// Called on the main thread once the movie file is closed and complete.
    var onFinished: ((Recording) -> Void)?

    private let queue = DispatchQueue(label: "com.mailens.pc-test.recorder", qos: .utility)
    private let stateLock = NSLock()
    private let encoder = JSONEncoder()

    private var recording = false
    private var sessionDirectory: URL?
    private var videoURL: URL?
    private var writer: AVAssetWriter?
    private var writerInput: AVAssetWriterInput?
    private var captureHandle: FileHandle?
    private var poseHandle: FileHandle?
    private var status = Status()
    private var startedAt = Date()
    private var firstTimestamp: Double?
    private var lastTimestamp: Double?
    private var nominalFPS: Double = 60
    private var lastPublish = Date.distantPast
    /// Frames waiting for the writer queue.  Bounded so a saturated encoder
    /// cannot grow the queue until iOS kills the app mid-take.
    private var pendingFrames = 0
    private var backlogDrops = 0

    var isRecording: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return recording
    }

    // MARK: - Lifecycle

    /// Creates the session directory and opens the logs.  The movie writer is
    /// created on the first frame, because only then the buffer size is known.
    @discardableResult
    func start(nominalFPS: Double = 60) -> URL? {
        var directory: URL?
        queue.sync {
            directory = self.startLocked(nominalFPS: nominalFPS)
        }
        return directory
    }

    func stop() {
        queue.async { [weak self] in self?.stopLocked() }
    }

    // MARK: - Capture callbacks

    /// Keeps every frame the camera handed over, even when the JPEG stream is
    /// throttled down or the Wi-Fi link is not connected at all.
    func append(sampleBuffer: CMSampleBuffer, pose: PoseSnapshot?) {
        guard isRecording else { return }
        stateLock.lock()
        let accepted = pendingFrames < Self.maxPendingFrames
        if accepted {
            pendingFrames += 1
        } else {
            backlogDrops += 1
        }
        stateLock.unlock()
        guard accepted else { return }
        queue.async { [weak self] in
            guard let self else { return }
            self.appendLocked(sampleBuffer: sampleBuffer, pose: pose)
            self.stateLock.lock()
            self.pendingFrames -= 1
            self.stateLock.unlock()
        }
    }

    /// The full-rate motion log.  Called from the motion queue, so it only
    /// hands the sample to the writer queue.
    func log(pose: PoseSnapshot) {
        guard isRecording else { return }
        queue.async { [weak self] in
            guard let self, self.recording, let handle = self.poseHandle else { return }
            self.status.poseSampleCount += 1
            self.write(pose, to: handle)
            self.publishStatus()
        }
    }

    // MARK: - Writer queue

    private func startLocked(nominalFPS: Double) -> URL? {
        guard !recording else { return sessionDirectory }
        let root = Self.sessionsRoot()
        do {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        } catch {
            return nil
        }

        let base = Self.nameFormatter.string(from: Date())
        var directory = root.appendingPathComponent(base, isDirectory: true)
        var suffix = 1
        while FileManager.default.fileExists(atPath: directory.path) {
            directory = root.appendingPathComponent("\(base)-\(suffix)", isDirectory: true)
            suffix += 1
        }

        let captureURL = directory.appendingPathComponent("capture.jsonl")
        let poseURL = directory.appendingPathComponent("pose.jsonl")
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            _ = FileManager.default.createFile(atPath: captureURL.path, contents: nil)
            _ = FileManager.default.createFile(atPath: poseURL.path, contents: nil)
            captureHandle = try FileHandle(forWritingTo: captureURL)
            poseHandle = try FileHandle(forWritingTo: poseURL)
        } catch {
            captureHandle = nil
            poseHandle = nil
            return nil
        }

        sessionDirectory = directory
        videoURL = directory.appendingPathComponent("video.mp4")
        writer = nil
        writerInput = nil
        status = Status()
        startedAt = Date()
        firstTimestamp = nil
        lastTimestamp = nil
        lastPublish = .distantPast
        pendingFrames = 0
        backlogDrops = 0
        setRecording(true)
        return directory
    }

    private func appendLocked(sampleBuffer: CMSampleBuffer, pose: PoseSnapshot?) {
        guard recording, sessionDirectory != nil else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width > 0, height > 0, ensureWriter(width: width, height: height) else { return }
        guard let writer, let writerInput, writer.status == .writing else { return }

        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let seconds = timestamp.seconds
        guard seconds.isFinite else { return }
        if firstTimestamp == nil {
            writer.startSession(atSourceTime: timestamp)
            firstTimestamp = seconds
        }
        guard writerInput.isReadyForMoreMediaData, writerInput.append(sampleBuffer) else {
            status.droppedFrames += 1
            publishStatus(force: true)
            return
        }

        status.frameCount += 1
        status.width = width
        status.height = height
        lastTimestamp = seconds
        let record = FrameRecord(frameID: UInt64(status.frameCount),
                                 timestamp: seconds,
                                 width: width,
                                 height: height,
                                 pose: pose,
                                 sensorDeltaMS: pose.map { ($0.timestamp - seconds) * 1000 })
        write(record, to: captureHandle)
        publishStatus()
    }

    private func stopLocked() {
        guard recording else { return }
        setRecording(false)

        let directory = sessionDirectory
        let frames = status.frameCount
        let poses = status.poseSampleCount
        stateLock.lock()
        let dropped = status.droppedFrames + backlogDrops
        stateLock.unlock()
        let width = status.width
        let height = status.height
        let duration = durationLocked()
        let wallClock = Date().timeIntervalSince(startedAt)
        let started = startedAt
        let fps = nominalFPS
        pendingFrames = 0
        backlogDrops = 0

        try? captureHandle?.close()
        try? poseHandle?.close()
        captureHandle = nil
        poseHandle = nil
        sessionDirectory = nil
        videoURL = nil
        firstTimestamp = nil
        lastTimestamp = nil
        status = Status()

        let manifest = Manifest(created: Self.isoFormatter.string(from: started),
                                wallClockStart: started.timeIntervalSince1970,
                                video: "video.mp4",
                                metadata: "capture.jsonl",
                                poseLog: "pose.jsonl",
                                width: width,
                                height: height,
                                nominalFPS: fps,
                                frameCount: frames,
                                poseSampleCount: poses,
                                duration: duration,
                                droppedFrames: dropped,
                                wallClockDuration: wallClock)

        let finish: () -> Void = { [weak self] in
            guard let self, let directory else { return }
            Self.writeManifest(manifest, to: directory)
            let recording = Recording(directory: directory, manifest: manifest)
            DispatchQueue.main.async { self.onFinished?(recording) }
        }

        if let writer, let writerInput, writer.status == .writing {
            writerInput.markAsFinished()
            writer.finishWriting(completionHandler: finish)
        } else {
            writer?.cancelWriting()
            finish()
        }
        writer = nil
        writerInput = nil
    }

    private func ensureWriter(width: Int, height: Int) -> Bool {
        if writer != nil { return true }
        guard let videoURL else { return false }
        do {
            let writer = try AVAssetWriter(outputURL: videoURL, fileType: .mp4)
            // Fragments survive a crash or a kill mid-session, which matters
            // when a take is recorded for minutes at a time.
            writer.movieFragmentInterval = CMTime(seconds: 2, preferredTimescale: 600)
            let input = AVAssetWriterInput(mediaType: .video,
                                           outputSettings: Self.videoSettings(width: width,
                                                                              height: height,
                                                                              fps: nominalFPS))
            input.expectsMediaDataInRealTime = true
            guard writer.canAdd(input) else { return false }
            writer.add(input)
            guard writer.startWriting() else { return false }
            self.writer = writer
            self.writerInput = input
            return true
        } catch {
            return false
        }
    }

    private func durationLocked() -> Double {
        guard let first = firstTimestamp, let last = lastTimestamp else { return 0 }
        return max(0, last - first)
    }

    private func setRecording(_ value: Bool) {
        stateLock.lock()
        recording = value
        stateLock.unlock()
    }

    private func write<T: Encodable>(_ value: T, to handle: FileHandle?) {
        guard let handle, let data = try? encoder.encode(value) else { return }
        var line = data
        line.append(0x0A)
        try? handle.write(contentsOf: line)
    }

    private func publishStatus(force: Bool = false) {
        let now = Date()
        if !force, now.timeIntervalSince(lastPublish) < 0.4 { return }
        lastPublish = now
        var snapshot = status
        snapshot.duration = durationLocked()
        stateLock.lock()
        snapshot.droppedFrames += backlogDrops
        stateLock.unlock()
        if let videoURL {
            snapshot.bytes = Self.size(of: videoURL)
        }
        DispatchQueue.main.async { [weak self] in self?.onStatus?(snapshot) }
    }

    // MARK: - Files

    static func sessionsRoot() -> URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        return documents.appendingPathComponent(folderName, isDirectory: true)
    }

    static func recordings() -> [Recording] {
        let root = sessionsRoot()
        let manager = FileManager.default
        guard let entries = try? manager.contentsOfDirectory(at: root,
                                                            includingPropertiesForKeys: [.isDirectoryKey],
                                                            options: [.skipsHiddenFiles]) else { return [] }
        return entries
            .compactMap { entry -> Recording? in
                let values = try? entry.resourceValues(forKeys: [.isDirectoryKey])
                guard values?.isDirectory == true else { return nil }
                return loadRecording(at: entry)
            }
            .sorted { $0.createdAt > $1.createdAt }
    }

    static func loadRecording(at directory: URL) -> Recording? {
        let manifestURL = directory.appendingPathComponent("session.json")
        if let data = try? Data(contentsOf: manifestURL),
           let manifest = try? JSONDecoder().decode(Manifest.self, from: data) {
            return Recording(directory: directory, manifest: manifest)
        }
        // A session killed before the manifest was written still has usable
        // video and logs, so keep it visible instead of hiding the files.
        let video = directory.appendingPathComponent("video.mp4")
        guard FileManager.default.fileExists(atPath: video.path) else { return nil }
        let modified = (try? directory.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
        return Recording(directory: directory,
                         createdAt: modified ?? Date(),
                         bytes: size(of: video),
                         frameCount: 0,
                         poseSampleCount: 0,
                         duration: 0,
                         width: 0,
                         height: 0)
    }

    static func delete(_ recording: Recording) throws {
        try FileManager.default.removeItem(at: recording.directory)
    }

    static func size(of url: URL) -> Int64 {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes?[.size] as? NSNumber)?.int64Value ?? 0
    }

    static func writeManifest(_ manifest: Manifest, to directory: URL) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes, .sortedKeys]
        guard let data = try? encoder.encode(manifest) else { return }
        try? data.write(to: directory.appendingPathComponent("session.json"))
    }

    // MARK: - Helpers

    private static func videoSettings(width: Int, height: Int, fps: Double) -> [String: Any] {
        // ~0.2 bit per pixel per frame keeps the fisheye edges readable without
        // letting a long take eat the whole phone.
        let bitRate = min(max(Int(Double(width * height) * fps * 0.2), 4_000_000), 24_000_000)
        return [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitRate,
                AVVideoMaxKeyFrameIntervalKey: max(1, Int(fps.rounded())),
                AVVideoExpectedSourceFrameRateKey: max(1, Int(fps.rounded())),
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
                AVVideoAllowFrameReorderingKey: false,
            ],
        ]
    }

    private static let maxPendingFrames = 4

    private static let isoFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    private static let nameFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter
    }()

    private struct FrameRecord: Encodable {
        let frameID: UInt64
        let timestamp: Double
        let width: Int
        let height: Int
        let pose: PoseSnapshot?
        let sensorDeltaMS: Double?

        enum CodingKeys: String, CodingKey {
            case frameID = "frame_id"
            case timestamp, width, height, pose
            case sensorDeltaMS = "sensor_delta_ms"
        }
    }
}
