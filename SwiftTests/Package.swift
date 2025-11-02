// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "EmbeddingGemmaSwiftTest",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "EmbeddingGemmaSwiftTest", targets: ["EmbeddingGemmaSwiftTest"]),
    ],
    dependencies: [
        .package(path: "../mlx-swift-examples"),
    ],
    targets: [
        .executableTarget(
            name: "EmbeddingGemmaSwiftTest",
            dependencies: [
                .product(name: "MLXEmbedders", package: "mlx-swift-examples"),
            ]
        ),
    ]
)
