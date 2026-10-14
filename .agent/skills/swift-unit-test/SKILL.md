---
name: swift-unit-test
description: How to write Swift unit tests (XCTest, Swift Testing, async/await)
triggers:
  - swift test
  - swift unit test
  - xctest
  - swift testing
  - swift mock
  - ios test
  - swiftui test
---

## 1. Libraries

| Purpose         | Library                                        |
|-----------------|------------------------------------------------|
| Framework       | XCTest (default) or Swift Testing (`@Test`)    |
| Assertions      | XCTest assertions / `#expect` (Swift Testing) |
| Mocking         | Protocols + hand-written fakes                 |
| Async           | `async`/`await` support built in               |
| Temp files      | `FileManager.default.temporaryDirectory`       |

Do **not** introduce third-party mocking frameworks (Cuckoo, Mockingbird,
Sourcery mocks) unless the project already uses them. Prefer protocol-based
fakes — they are idiomatic Swift and need no code generation.

## 2. File layout

- Tests live in the test target: `Tests/<ModuleName>Tests/` for SPM,
  `<Project>Tests/` for Xcode projects.
- One test file per type: `FooTests.swift` tests `Foo.swift`.
- Shared test helpers go in `Tests/<ModuleName>Tests/TestHelpers/` or a
  dedicated `TestSupport` module — never in the production target.

## 3. Naming

- **Classes**: `FooTests`, `FooServiceTests` — PascalCase, `Tests` suffix.
- **Methods**: `test` prefix + descriptive camelCase:

```swift
func testCalculateTax_returnsZeroForExemptItems() throws { … }
```

- With Swift Testing, use `@Test` with a display name:

```swift
@Test("Calculate tax returns zero for exempt items")
func calculateTaxExemptItems() throws { … }
```

- No `given_when_then` or `should` prefixes unless project convention.

## 4. Structure

### XCTest lifecycle

```swift
final class TaxCalculatorTests: XCTestCase {
    private var sut: TaxCalculator!
    private var mockLoader: MockRateLoader!

    override func setUp() {
        super.setUp()
        mockLoader = MockRateLoader()
        sut = TaxCalculator(loader: mockLoader)
    }

    override func tearDown() {
        sut = nil
        mockLoader = nil
        super.tearDown()
    }
}
```

### Swift Testing lifecycle

```swift
struct TaxCalculatorTests {
    let sut: TaxCalculator
    let mockLoader: MockRateLoader

    init() {
        mockLoader = MockRateLoader()
        sut = TaxCalculator(loader: mockLoader)
    }
}
```

### Arrange / Act / Assert

```swift
func testApplyDiscount_capsAt50Percent() {
    let order = Order(items: [Item(name: "A", price: 200)])

    let result = sut.applyDiscount(order, percent: 75)

    XCTAssertEqual(result.total, 100)
}
```

Separate the three phases with blank lines. No labeling comments.

## 5. Mocking with protocols

### Define a protocol

```swift
protocol OrderRepository {
    func findById(_ id: Int) -> Order?
    func save(_ order: Order) throws
}
```

### Write a fake

```swift
final class MockOrderRepository: OrderRepository {
    var findByIdResult: Order?
    private(set) var savedOrders: [Order] = []

    func findById(_ id: Int) -> Order? {
        findByIdResult
    }

    func save(_ order: Order) throws {
        savedOrders.append(order)
    }
}
```

- Expose inputs as `private(set)` stored properties for verification.
- Expose outputs as settable properties for stubbing.
- Name fakes `Mock<Protocol>` or `Stub<Protocol>`.

### Async mocking

```swift
protocol DataService {
    func fetch() async throws -> [Item]
}

final class MockDataService: DataService {
    var fetchResult: Result<[Item], Error> = .success([])

    func fetch() async throws -> [Item] {
        try fetchResult.get()
    }
}
```

## 6. Async tests

### XCTest

```swift
func testSync_fetchesAndPersists() async throws {
    mockAPI.fetchResult = .success([Item(name: "x")])

    try await sut.sync()

    XCTAssertEqual(mockDB.insertedItems.count, 1)
}
```

### Swift Testing

```swift
@Test("Sync fetches and persists")
func syncFetchesAndPersists() async throws {
    mockAPI.fetchResult = .success([Item(name: "x")])

    try await sut.sync()

    #expect(mockDB.insertedItems.count == 1)
}
```

- Mark test methods `async throws` — no completion handler wrapping.
- For `@MainActor`-isolated code, mark the test `@MainActor` too.
- Use `Task` and `TaskGroup` in tests only when testing concurrency itself.

## 7. Assertions

### XCTest

```swift
XCTAssertEqual(expected, actual)
XCTAssertTrue(result.isValid)
XCTAssertFalse(list.isEmpty)
XCTAssertNil(result)
XCTAssertNotNil(result)
XCTAssertThrowsError(try sut.process(nil)) { error in
    XCTAssertTrue(error is ValidationError)
}
XCTAssertNoThrow(try sut.validate(input))
```

### Swift Testing

```swift
#expect(result.isValid)
#expect(!list.isEmpty)
#expect(result == expected)
#expect(result == nil)
#expect(result != nil)
#expect(throws: ValidationError.self) {
    try sut.process(nil)
}
```

For collection assertions:

```swift
// XCTest
XCTAssertEqual(items.count, 3)
XCTAssertTrue(items.allSatisfy(\.isActive))
XCTAssertEqual(items.map(\.name), ["a", "b"])

// Swift Testing
#expect(items.count == 3)
#expect(items.allSatisfy(\.isActive))
#expect(items.map(\.name) == ["a", "b"])
```

## 8. Parameterized tests (Swift Testing)

```swift
@Test("Tax rate by region", arguments: [
    ("US", 0.08),
    ("EU", 0.20),
    ("exempt", 0.0),
])
func taxRate(region: String, expected: Double) {
    let rate = TaxCalculator.rate(for: region)
    #expect(rate == expected)
}
```

Use parameterized tests for table-driven cases instead of loops.

## 9. iOS / SwiftUI specific

- Use `ViewInspector` only if the project already depends on it.
- Prefer testing ViewModels and logic layers over views directly.
- ViewModel tests: inject dependencies via init, use protocol-based fakes.

```swift
@MainActor
func testViewModel_loadsItems() async {
    let mockService = MockItemService()
    mockService.fetchResult = .success([Item(name: "test")])
    let vm = ItemListViewModel(service: mockService)

    await vm.loadItems()

    XCTAssertEqual(vm.items.count, 1)
    XCTAssertFalse(vm.isLoading)
}
```

- For Combine publishers, use `values` (AsyncSequence) or `sink` with expectations:

```swift
func testPublisher_emitsValues() {
    let expectation = expectation(description: "receives value")
    var received: [String] = []

    sut.$items
        .dropFirst()
        .sink { items in
            received = items
            expectation.fulfill()
        }
        .store(in: &cancellables)

    sut.loadItems()
    wait(for: [expectation], timeout: 1.0)
    XCTAssertEqual(received, ["a", "b"])
}
```

## 10. Don'ts

- **No force-unwrapping in tests** — use `XCTUnwrap` or `try #require` (Swift
  Testing) instead of `!`.
- **No third-party mocks by default** — use protocol fakes.
- **No `sleep` or `Thread.sleep`** — use `async`/`await` or `XCTestExpectation`.
- **No testing private methods** — test through the public API.
- **No test logic in production code** — no `#if DEBUG` test hooks unless the
  alternative is impossible to test.
- **No `@testable import` abuse** — use it for internal access, not to test
  private implementation details.
- **No disabled tests** — delete them or fix them; `XCTSkipIf` / `XCTSkipUnless`
  is for conditional platform skips, not TODO placeholders.
