import Foundation
import Networking

public protocol UserRepositoryProtocol: AnyObject {
    func load(id: String) async throws -> User
    func clear()
}

public final class UserRepository: UserRepositoryProtocol {
    private let client: APIClient
    private let cache = UserCache()

    public init(client: APIClient) {
        self.client = client
    }

    public func load(id: String) async throws -> User {
        if let cached = cache.get(id) {
            return cached
        }
        let data = try await client.get("/users/\(id)")
        let user = try JSONDecoder().decode(User.self, from: data)
        cache.put(user)
        return user
    }

    public func clear() {
        cache.clear()
    }
}

final class UserCache {
    private var storage: [String: User] = [:]

    func get(_ id: String) -> User? { storage[id] }

    func put(_ user: User) {
        storage[user.id] = user
    }

    func clear() {
        storage.removeAll()
    }

    func invalidate() { clear() }
}
