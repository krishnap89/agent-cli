import Foundation
import CoreKit

@MainActor
final class UserListViewModel: ObservableObject {
    @Published private(set) var users: [User] = []
    @Published var selected: User?
    private let repository: UserRepositoryProtocol

    init(repository: UserRepositoryProtocol) {
        self.repository = repository
    }

    func refresh(id: String) {
        Task { [weak self] in
            guard let self else { return }
            if let user = try? await self.repository.load(id: id) {
                self.users = [user]
            }
        }
    }

    func reset() {
        repository.clear()
        users.removeAll()
    }

    func select(_ user: User) {
        selected = user
    }

    func select(id: String) {
        selected = users.first { $0.id == id }
    }
}
