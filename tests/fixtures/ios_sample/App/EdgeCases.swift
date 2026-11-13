import Foundation
import CoreKit

/* block /* nested */ comment: func fake() { fakeCall() } */
enum Formatter {
    static let raw = #"literal \(notCode()) and \#(realCode())"#
    static let multi = """
        func ghost() { haunt() }
        value: \(realCode())
        """

    static func realCode() -> String { "x" }

    struct Nested {
        func ping() -> Int { Deep().pong() + 1 }

        struct Deep {
            func pong() -> Int { 0 }
        }
    }
}

func closures(_ items: [Int], repo: UserRepositoryProtocol) {
    let transform: (Int) -> Int = { $0 * 2 }
    _ = transform(3)
    let mapped = items.map { value in value + 1 }
    _ = mapped.filter { $0 > 1 }
    let nested = Formatter.Nested()
    _ = nested.ping()
    if items.isEmpty {
        repo.clear()
    }
    #if DEBUG
    debugLog("closures")
    #endif
}

func debugLog(_ message: String) { print(message) }

// Unknown receiver: two project types have invalidate() -> ambiguous.
func invalidateFirst(_ things: [AnyObject]) {
    (things.first as AnyObject?)?.invalidate()
}
