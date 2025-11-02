import MLXEmbedders
import Foundation
import MLX
import MLXFast
import MLXLinalg
import Tokenizers

enum TestError: Error {
    case noTokens
    case missingEmbeddings
}

@main
struct EmbeddingGemmaSwiftTest {
    static func main() async {
        let sentences = [
            "task: sentence similarity | query: Nothing really matters.",
            "task: sentence similarity | query: The dog is barking.",
            "task: sentence similarity | query: The dog is barking."
        ]

        do {
            let configuration = ModelConfiguration(id: "mlx-community/embeddinggemma-300m-4bit")
            let container = try await loadModelContainer(configuration: configuration)

            let embeddings = try await container.perform { model, tokenizer, _ -> MLXArray in
                let eosId = tokenizer.eosTokenId ?? 0
                let encodedSequences = sentences.map { sentence in
                    tokenizer.encode(text: sentence, addSpecialTokens: true)
                }

                guard let maxLength = encodedSequences.map(\.count).max(), maxLength > 0 else {
                    throw TestError.noTokens
                }

                let paddedInputs = encodedSequences.map { sequence -> MLXArray in
                    let paddingCount = max(0, maxLength - sequence.count)
                    let padded = sequence + Array(repeating: eosId, count: paddingCount)
                    return MLXArray(padded.map(Int32.init))
                }
                let inputIds = stacked(paddedInputs, axis: 0).asType(.int32)

                let attentionMask = (inputIds .!= MLXArray(Int32(eosId))).asType(.float16)

                let output = model(
                    inputIds,
                    positionIds: nil,
                    tokenTypeIds: nil,
                    attentionMask: attentionMask
                )

                guard let pooled = output.pooledOutput else {
                    throw TestError.missingEmbeddings
                }

                return pooled
            }

            let normalizedEmbeddings = embeddings
            let similarity = normalizedEmbeddings.matmul(normalizedEmbeddings.transposed(1, 0))

            print("Similarity matrix between texts:")
            print(similarity)
        } catch {
            print("Error: \(error)")
        }
    }
}
