import Foundation

public protocol APIClient {
    func get(_ path: String) async throws -> Data
}

public final class URLSessionClient: APIClient {
    private let session: URLSession

    public init(session: URLSession = .shared) {
        self.session = session
    }

    public func get(_ path: String) async throws -> Data {
        let url = URL(string: "https://api.example.com\(path)")!
        let (data, _) = try await session.data(from: url)
        return data
    }
}
