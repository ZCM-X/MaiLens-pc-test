import CoreMotion
import Foundation

struct PoseSnapshot: Codable {
    struct Quaternion: Codable {
        let x: Double
        let y: Double
        let z: Double
        let w: Double
    }

    struct Vector3: Codable {
        let x: Double
        let y: Double
        let z: Double
    }

    let timestamp: Double
    let quaternion: Quaternion
    let gravity: Vector3
    let rotationRate: Vector3
    /// Gravity already removed, so this is the movement the hand actually
    /// added.  Double integration drifts, but it is the only translation
    /// signal a single phone can give the front/back part of the lock.
    let userAcceleration: Vector3

    enum CodingKeys: String, CodingKey {
        case timestamp, quaternion, gravity
        case rotationRate = "rotation_rate"
        case userAcceleration = "user_acceleration"
    }

    init(_ motion: CMDeviceMotion) {
        let q = motion.attitude.quaternion
        timestamp = motion.timestamp
        quaternion = Quaternion(x: q.x, y: q.y, z: q.z, w: q.w)
        gravity = Vector3(x: motion.gravity.x, y: motion.gravity.y, z: motion.gravity.z)
        rotationRate = Vector3(x: motion.rotationRate.x,
                              y: motion.rotationRate.y,
                              z: motion.rotationRate.z)
        userAcceleration = Vector3(x: motion.userAcceleration.x,
                                   y: motion.userAcceleration.y,
                                   z: motion.userAcceleration.z)
    }
}

/// Keeps the high-rate sensor stream independent from camera/network work.
/// Camera frames select the closest sample by the host-monotonic timestamp.
final class MotionPoseProvider {
    private let manager = CMMotionManager()
    private let queue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "com.mailens.pc-test.motion"
        queue.qualityOfService = .userInteractive
        queue.maxConcurrentOperationCount = 1
        return queue
    }()
    private let lock = NSLock()
    private var samples: [PoseSnapshot] = []

    var isAvailable: Bool { manager.isDeviceMotionAvailable }

    /// Fired for every sample on the motion queue.  Frames only need
    /// `nearest(to:)`, but the offline session log wants the full 120 Hz stream
    /// so the PC can re-match it against a recorded video.
    var onSample: ((PoseSnapshot) -> Void)?

    func start() {
        guard manager.isDeviceMotionAvailable, !manager.isDeviceMotionActive else { return }
        manager.deviceMotionUpdateInterval = 1.0 / 120.0
        manager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: queue) { [weak self] motion, _ in
            guard let self, let motion else { return }
            let snapshot = PoseSnapshot(motion)
            self.append(snapshot)
            self.onSample?(snapshot)
        }
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
        lock.lock()
        samples.removeAll(keepingCapacity: true)
        lock.unlock()
    }

    func nearest(to timestamp: Double) -> PoseSnapshot? {
        lock.lock()
        defer { lock.unlock() }
        guard !samples.isEmpty else { return nil }
        return samples.min { abs($0.timestamp - timestamp) < abs($1.timestamp - timestamp) }
    }

    private func append(_ sample: PoseSnapshot) {
        lock.lock()
        samples.append(sample)
        if samples.count > 360 {
            samples.removeFirst(samples.count - 360)
        }
        lock.unlock()
    }
}
