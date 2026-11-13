import XCTest
import CoreKit
@testable import App

final class UserListViewModelTests: XCTestCase {
    @MainActor
    func testResetClearsRepository() {
        let repo = MockRepository()
        let vm = UserListViewModel(repository: repo)
        vm.reset()
        XCTAssertTrue(repo.cleared)
    }
}

final class MockRepository: UserRepositoryProtocol {
    var cleared = false

    func load(id: String) async throws -> User { User(id: id, name: "x") }

    func clear() { cleared = true }
}
