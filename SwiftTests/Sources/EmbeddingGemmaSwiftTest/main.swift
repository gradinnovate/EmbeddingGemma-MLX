import MLXEmbedders
import Foundation
import MLX
import MLXFast
import MLXLinalg
import Tokenizers

enum TestError: Error {
    case noTokens
    case missingEmbeddings
    case unsupportedTokenizer
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

            let embeddings = try await container.perform { model, tokenizer, pooler -> MLXArray in
                let padId: Int
                if let gemma = model as? EmbeddingGemmaModel {
                    padId = gemma.config.padTokenId ?? gemma.config.eosTokenId ?? tokenizer.eosTokenId ?? 0
                } else {
                    padId = tokenizer.eosTokenId ?? 0
                }
                let encodedSequences = sentences.map { sentence in
                    tokenizer.encode(text: sentence, addSpecialTokens: true)
                }

                guard let maxLength = encodedSequences.map(\.count).max(), maxLength > 0 else {
                    throw TestError.noTokens
                }

                var paddedRows = [MLXArray]()
                var maskRows = [MLXArray]()

                for sequence in encodedSequences {
                    var row = sequence
                    if sequence.count < maxLength {
                        row += Array(repeating: padId, count: maxLength - sequence.count)
                    }
                    paddedRows.append(MLXArray(row.map(Int32.init)))

                    var maskValues = Array(repeating: Int32(0), count: maxLength)
                    for idx in 0..<sequence.count {
                        maskValues[idx] = 1
                    }
                    maskRows.append(MLXArray(maskValues))
                }

                let inputIds = stacked(paddedRows, axis: 0).asType(.int32)
                let attentionMask = stacked(maskRows, axis: 0)

                let output = model(
                    inputIds,
                    positionIds: nil,
                    tokenTypeIds: nil,
                    attentionMask: attentionMask
                )
                //print("Model output obtained. \(output)")

                return pooler(
                    output,
                    mask: attentionMask,
                    normalize: true,
                    applyLayerNorm: false
                )
            }

            let normalizedEmbeddings = embeddings.asType(.float32)
            let similarity = normalizedEmbeddings.matmul(normalizedEmbeddings.transposed(1, 0)).asType(.float32)

            print("Similarity matrix between texts:")
            print(similarity)
        } catch {
            print("Error: \(error)")
        }
    }
}
