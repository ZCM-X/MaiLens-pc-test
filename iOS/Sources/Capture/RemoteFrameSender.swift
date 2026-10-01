import Foundation
import Network

struct FrameMetadata: Codable {
    let frameID: UInt64
    let timestamp: Double
    let width: Int
    let height: Int
    let pose: PoseSnapshot?

    enum CodingKeys: String, CodingKey {
        case frameID = "frame_id"
        case timestamp, width, height, pose
    }
}

/// Length-prefixed TCP sender with a single in-flight packet and a one-frame
/// replaceable queue. This bounds latency when Wi-Fi cannot keep up.
final class RemoteFrameSender {
    private let queue = DispatchQueue(label: "com.mailens.pc-test.network", qos: .userInitiated)
    private var connection: NWConnection?
    private var pendingPacket: Data?
    private var isSending = false
    private var frameCounter: UInt64 = 0
    private var droppedCounter: UInt64 = 0

    var onState: ((String) -> Void)?
    var onStatistics: ((UInt64, UInt64) -> Void)?

    func connect(host: String, port: UInt16) {
        queue.async { [weak self] in
            guard let self else { return }
            self.connection?.cancel()
            self.pendingPacket = nil
            self.isSending = false
            guard let nwPort = NWEndpoint.Port(rawValue: port) else {
                self.publishState("端口无效")
                return
            }
            let connection = NWConnection(host: NWEndpoint.Host(host), port: nwPort, using: .tcp)
            self.connection = connection
            connection.stateUpdateHandler = { [weak self, weak connection] state in
                guard let self, self.connection === connection else { return }
                switch state {
                case .ready:
                    self.publishState("已连接")
                    let hello = (try? JSONSerialization.data(withJSONObject: ["client": "MaiLensRemoteCapture", "version": 1])) ?? Data()
                    self.enqueue(self.packet(type: 2, metadata: hello, payload: Data()))
                case .waiting(let error): self.publishState("等待网络：\(error.localizedDescription)")
                case .failed(let error): self.publishState("连接失败：\(error.localizedDescription)")
                case .cancelled: self.publishState("已断开")
                default: self.publishState("正在连接…")
                }
            }
            connection.start(queue: self.queue)
        }
    }

    func disconnect() {
        queue.async { [weak self] in
            guard let self else { return }
            self.pendingPacket = nil
            self.isSending = false
            self.connection?.cancel()
            self.connection = nil
            self.publishState("已断开")
        }
    }

    func send(jpeg: Data, timestamp: Double, width: Int, height: Int, pose: PoseSnapshot?) {
        queue.async { [weak self] in
            guard let self, self.connection != nil else { return }
            self.frameCounter &+= 1
            let metadata = FrameMetadata(frameID: self.frameCounter,
                                         timestamp: timestamp,
                                         width: width,
                                         height: height,
                                         pose: pose)
            guard let data = try? JSONEncoder().encode(metadata) else { return }
            self.enqueue(self.packet(type: 1, metadata: data, payload: jpeg))
            self.onStatistics?(self.frameCounter, self.droppedCounter)
        }
    }

    private func enqueue(_ packet: Data) {
        if isSending {
            if pendingPacket != nil { droppedCounter &+= 1 }
            pendingPacket = packet
            return
        }
        sendPacket(packet)
    }

    private func sendPacket(_ packet: Data) {
        guard let connection else { return }
        isSending = true
        connection.send(content: packet, completion: .contentProcessed { [weak self] error in
            guard let self else { return }
            self.queue.async {
                self.isSending = false
                if let error {
                    self.publishState("发送失败：\(error.localizedDescription)")
                    self.connection?.cancel()
                    self.connection = nil
                    self.pendingPacket = nil
                    return
                }
                guard let next = self.pendingPacket else { return }
                self.pendingPacket = nil
                self.sendPacket(next)
            }
        })
    }

    private func packet(type: UInt8, metadata: Data, payload: Data) -> Data {
        var result = Data("MLCP".utf8)
        result.append(1) // protocol version
        result.append(type)
        result.appendUInt32(UInt32(metadata.count))
        result.appendUInt32(UInt32(payload.count))
        result.append(metadata)
        result.append(payload)
        return result
    }

    private func publishState(_ value: String) {
        DispatchQueue.main.async { [weak self] in self?.onState?(value) }
    }
}

private extension Data {
    mutating func appendUInt32(_ value: UInt32) {
        var bigEndian = value.bigEndian
        Swift.withUnsafeBytes(of: &bigEndian) { append(contentsOf: $0) }
    }
}
