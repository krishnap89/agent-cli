---
name: kotlin-unit-test
description: How to write Kotlin/JVM unit tests (JUnit 5, MockK, coroutines)
triggers:
  - kotlin test
  - kotlin unit test
  - junit5
  - mockk
  - kotest
  - kotlin mock
  - android test
---

## 1. Libraries

| Purpose         | Library                                  |
|-----------------|------------------------------------------|
| Framework       | JUnit 5 (`org.junit.jupiter`)            |
| Assertions      | JUnit 5 assertions + kotlin.test helpers |
| Mocking         | MockK (`io.mockk`)                       |
| Coroutines      | `kotlinx-coroutines-test` (`runTest`)    |
| Temp files      | `@TempDir` (JUnit 5)                     |

Do **not** introduce Kotest matchers, AssertJ, Hamcrest, or Mockito unless the
project already uses them.

## 2. File layout

- Tests mirror `src/main/kotlin/` under `src/test/kotlin/` with matching packages.
- One test file per class: `FooTest.kt` tests `Foo.kt`.
- Shared test fixtures go in `src/test/kotlin/<package>/testutil/` — never in the
  production source set.

## 3. Naming

- **Classes**: `FooTest`, `FooServiceTest` — PascalCase, `Test` suffix.
- **Methods**: backtick descriptive names for readability:

```kotlin
@Test
fun `calculate tax returns zero for exempt items`() { … }
```

- Alternatively `camelCase` is fine if the project already uses it:

```kotlin
@Test
fun calculateTax_returnsZeroForExemptItems() { … }
```

- No `given_when_then` or `should` prefixes unless the project already does.

## 4. Structure

### Lifecycle

```kotlin
@TempDir
lateinit var tempDir: Path

private lateinit var sut: TaxCalculator

@BeforeEach
fun setUp() {
    sut = TaxCalculator(RateLoader(tempDir))
}
```

Use `@BeforeEach` / `@AfterEach`, not `@Before` / `@After` (those are JUnit 4).

### Arrange / Act / Assert

```kotlin
@Test
fun `apply discount caps at 50 percent`() {
    val order = Order(items = listOf(Item("A", price = 200)))

    val result = sut.applyDiscount(order, percent = 75)

    assertEquals(100, result.total)
}
```

Separate the three phases with blank lines. No labeling comments.

## 5. Mocking (MockK)

### Basic mock

```kotlin
private val repo: OrderRepository = mockk()

@BeforeEach
fun setUp() {
    every { repo.findById(any()) } returns Order(id = 1)
}
```

### Verify

```kotlin
verify(exactly = 1) { repo.save(match { it.status == Status.COMPLETE }) }
```

### Relaxed mocks

Use `mockk(relaxed = true)` only when the mock has many unused methods.
Prefer explicit stubs — they document what the test depends on.

### Coroutine mocking

```kotlin
coEvery { service.fetchRemote(any()) } returns RemoteData("ok")
coVerify { service.fetchRemote("key-1") }
```

## 6. Coroutine tests

```kotlin
@Test
fun `sync fetches and persists`() = runTest {
    coEvery { api.fetch() } returns listOf(Item("x"))

    sut.sync()

    coVerify { db.insertAll(match { it.size == 1 }) }
}
```

- Always use `runTest` from `kotlinx-coroutines-test`, never `runBlocking`.
- Inject `StandardTestDispatcher` or `UnconfinedTestDispatcher` via constructor
  when the class under test launches coroutines.

## 7. Assertions

Prefer Kotlin stdlib + JUnit 5:

```kotlin
assertEquals(expected, actual)
assertTrue(result.isValid)
assertFalse(list.isEmpty())
assertNull(result)
assertNotNull(result)
assertThrows<IllegalArgumentException> { sut.process(null) }
assertContains(result.tags, "urgent")
```

For collection assertions:

```kotlin
assertEquals(3, items.size)
assertTrue(items.all { it.isActive })
assertEquals(listOf("a", "b"), items.map { it.name })
```

No `shouldBe`, `assertThat`, or custom matchers unless project convention.

## 8. Android-specific

- Use `Robolectric` only when testing Android framework classes (Context, Intent).
- Use `@RunWith(AndroidJUnit4::class)` for instrumented tests.
- Prefer extracting logic into plain Kotlin classes testable without Android.
- ViewModel tests: use `Dispatchers.setMain(UnconfinedTestDispatcher())` in
  `@BeforeEach` and `Dispatchers.resetMain()` in `@AfterEach`.

```kotlin
@BeforeEach
fun setUp() {
    Dispatchers.setMain(UnconfinedTestDispatcher())
}

@AfterEach
fun tearDown() {
    Dispatchers.resetMain()
}
```

## 9. Don'ts

- **No JUnit 4** — use `org.junit.jupiter.api`, not `org.junit`.
- **No Mockito** — use MockK (idiomatic Kotlin mocking).
- **No `runBlocking` in tests** — use `runTest`.
- **No `Thread.sleep`** — use `advanceTimeBy` / `advanceUntilIdle` with test dispatchers.
- **No test logic in production code** — no `@VisibleForTesting` escape hatches
  unless the alternative is reflection.
- **No ignoring tests** — delete them or fix them; `@Disabled` is a last resort
  with a linked issue.
