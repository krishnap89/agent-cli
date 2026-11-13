// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "CoreKit",
    platforms: [.iOS(.v16)],
    products: [.library(name: "CoreKit", targets: ["CoreKit"])],
    dependencies: [.package(url: "https://github.com/hmlongco/Factory", from: "2.3.0")],
    targets: [
        .target(name: "Networking"),
        .target(name: "CoreKit", dependencies: ["Networking", .product(name: "Factory", package: "Factory")]),
        // .target(name: "Ghost"),  commented out: must be ignored
        .testTarget(name: "CoreKitTests", dependencies: ["CoreKit"]),
    ]
)
