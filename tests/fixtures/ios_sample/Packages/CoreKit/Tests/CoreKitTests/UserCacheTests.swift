import XCTest
@testable import CoreKit

final class UserCacheTests: XCTestCase {
    func testPutGet() {
        let cache = UserCache()
        cache.put(User(id: "1", name: "a"))
        XCTAssertEqual(cache.get("1")?.name, "a")
    }
}
