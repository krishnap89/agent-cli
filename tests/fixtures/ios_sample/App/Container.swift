import CoreKit
import Networking

final class Container {
    static let shared = Container()

    lazy var repository: UserRepositoryProtocol = UserRepository(client: URLSessionClient())

    func makeViewModel() -> UserListViewModel {
        UserListViewModel(repository: repository)
    }
}
