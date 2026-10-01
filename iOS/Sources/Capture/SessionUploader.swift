import Foundation
import Network

/// Pushes a recorded session to the PC over the LAN.
///
/// The frame stream keeps its own socket; this uses a second one (stream port
/// plus one) so a multi-minute upload cannot stall the live preview.  Plain TCP
/// also keeps App Transport Security out of the picture.
final class SessionUploader {
    enum State: Equatable {
        case idle
        case sending(progress: Double)
        case finished(bytes: Int64)
        case failed(String)

        var isBusy: Bool {
            if case .sending = self { return true }
            return false
        }
    }

    enum UploadError: LocalizedError {
        case nothingToSend
        case badPort
        case notConnected
        case timeout
        case cancelled
        case rejected(String)
        case transport(String)

        var errorDescription: String? {
            switch self {
            case .nothingToSend: return "这段录制里没有可发送的文件。"
            case .badPort: return "发送端口无效。"
            case .notConnected: return "连不上电脑的上传端口，先在电脑运行 pc/session_server.py。"
            case .timeout: return "发送超时，检查 Wi‑Fi 是否断开。"
            case .cancelled: return "已取消发送。"
            case .rejected(let reason): return "电脑拒绝接收：\(reason)"
            case .transport(let reason): return "网络错误：\(reason)"
            }
        }
    }

    /// Always delivered on the main thread.
    var onState: ((State) -> Void)?

    private static let magic = Data("MLSF".utf8)
    private static let version: UInt8 = 1
    private static let chunkSize = 256 * 1024
    private static let sendTimeout = DispatchTimeInterval.seconds(30)
    private static let openTimeout = DispatchTimeInterval.seconds(8)

    private let queue = DispatchQueue(label: "com.mailens.pc-test.upload", qos: .utility)
    /// Connection callbacks need their own queue: the sender blocks on
    /// semaphores while it streams, and the completions must still be able to
    /// run on a different queue.
    private let networkQueue = DispatchQueue(label: "com.mailens.pc-test.upload.network", qos: .utility)
    private var connection: NWConnection?
    private var cancelled = false

    func upload(_ recording: SessionRecorder.Recording, host: String, port: UInt16) {
        queue.async { [weak self] in
            guard let self else { return }
            self.cancelled = false
            self.publish(.sending(progress: 0))
            do {
                let bytes = try self.run(recording, host: host, port: port)
                self.publish(self.cancelled ? .failed(UploadError.cancelled.localizedDescription) : .finished(bytes: bytes))
            } catch {
                self.publish(.failed(error.localizedDescription))
            }
            self.connection?.cancel()
            self.connection = nil
        }
    }

    func cancel() {
        queue.async { [weak self] in
            self?.cancelled = true
            self?.connection?.cancel()
            self?.connection = nil
        }
    }

    // MARK: - Sender

    private func run(_ recording: SessionRecorder.Recording, host: String, port: UInt16) throws -> Int64 {
        let files = recording.files
        guard !files.isEmpty else { throw UploadError.nothingToSend }
        guard let nwPort = NWEndpoint.Port(rawValue: port) else { throw UploadError.badPort }

        let connection = NWConnection(host: NWEndpoint.Host(host), port: nwPort, using: .tcp)
        self.connection = connection
        let ready = DispatchSemaphore(value: 0)
        connection.stateUpdateHandler = { state in
            switch state {
            case .ready, .failed, .cancelled: ready.signal()
            default: break
            }
        }
        connection.start(queue: networkQueue)
        guard ready.wait(timeout: .now() + Self.openTimeout) == .success,
              connection.state == .ready else {
            throw UploadError.notConnected
        }

        let sessionName = Data(recording.name.utf8)
        var header = Self.magic
        header.append(Self.version)
        header.appendUInt16(UInt16(clamping: sessionName.count))
        header.append(sessionName)
        try send(header, on: connection)

        let total = files.reduce(Int64(0)) { $0 + SessionRecorder.size(of: $1) }
        var sent: Int64 = 0
        for url in files {
            let name = Data(url.lastPathComponent.utf8)
            let length = SessionRecorder.size(of: url)
            var record = Data([1])
            record.appendUInt16(UInt16(clamping: name.count))
            record.append(name)
            record.appendUInt64(UInt64(max(0, length)))
            try send(record, on: connection)
            try sendFile(url, length: length, on: connection, total: total, sent: &sent)
        }
        try send(Data([2]), on: connection)

        let reply = try receiveLine(on: connection)
        guard reply.hasPrefix("OK") else { throw UploadError.rejected(reply) }
        return total
    }

    private func sendFile(_ url: URL,
                          length: Int64,
                          on connection: NWConnection,
                          total: Int64,
                          sent: inout Int64) throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var remaining = length
        while remaining > 0 {
            if cancelled { throw UploadError.cancelled }
            let wanted = Int(min(Int64(Self.chunkSize), remaining))
            guard let chunk = try handle.read(upToCount: wanted), !chunk.isEmpty else { break }
            try send(chunk, on: connection)
            remaining -= Int64(chunk.count)
            sent += Int64(chunk.count)
            if total > 0 {
                publish(.sending(progress: min(1, Double(sent) / Double(total))))
            }
        }
    }

    private func send(_ data: Data, on connection: NWConnection) throws {
        let semaphore = DispatchSemaphore(value: 0)
        let result = SendResult()
        connection.send(content: data, completion: .contentProcessed { error in
            result.error = error
            semaphore.signal()
        })
        if semaphore.wait(timeout: .now() + Self.sendTimeout) == .timedOut {
            throw UploadError.timeout
        }
        if let error = result.error {
            throw UploadError.transport(error.localizedDescription)
        }
    }

    private func receiveLine(on connection: NWConnection) throws -> String {
        var buffer = Data()
        let deadline = Date().addingTimeInterval(15)
        while Date() < deadline {
            let semaphore = DispatchSemaphore(value: 0)
            let chunk = ReceiveResult()
            connection.receive(minimumIncompleteLength: 1, maximumLength: 512) { data, _, isComplete, error in
                chunk.data = data
                chunk.error = error
                chunk.isComplete = isComplete
                semaphore.signal()
            }
            if semaphore.wait(timeout: .now() + 15) == .timedOut { throw UploadError.timeout }
            if let error = chunk.error { throw UploadError.transport(error.localizedDescription) }
            if let data = chunk.data, !data.isEmpty {
                buffer.append(data)
                if let newline = buffer.firstIndex(of: 0x0A) {
                    return String(decoding: buffer[buffer.startIndex..<newline], as: UTF8.self)
                }
            }
            if chunk.isComplete { break }
        }
        guard !buffer.isEmpty else { throw UploadError.timeout }
        return String(decoding: buffer, as: UTF8.self)
    }

    private func publish(_ state: State) {
        DispatchQueue.main.async { [weak self] in self?.onState?(state) }
    }

    private final class SendResult {
        var error: Error?
    }

    private final class ReceiveResult {
        var data: Data?
        var error: Error?
        var isComplete = false
    }
}

private extension Data {
    mutating func appendUInt16(_ value: UInt16) {
        var bigEndian = value.bigEndian
        Swift.withUnsafeBytes(of: &bigEndian) { append(contentsOf: $0) }
    }

    mutating func appendUInt64(_ value: UInt64) {
        var bigEndian = value.bigEndian
        Swift.withUnsafeBytes(of: &bigEndian) { append(contentsOf: $0) }
    }
}
