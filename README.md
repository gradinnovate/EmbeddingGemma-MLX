# EmbeddingGemma-MLX

This repository ports Google’s EmbeddingGemma text encoder to the Swift MLX stack by extending the `mlx-swift-examples` project. It provides a Swift implementation of the `embeddinggemma-300m-4bit` model, a sample executable that exercises the embedder, and parity scripts that verify the implementation against the original Python reference.

## Repository Layout
- `mlx-swift-examples/` – fork of Apple’s MLX Swift examples with `EmbeddingGemmaModel` wired into the embedders library.
- `models/embeddinggemma-300m-4bit/` – locally cached Hugging Face weights and tokenizer files (download requires agreeing to Google’s Gemma license).
- `SwiftTests/` – Swift Package that demonstrates loading the embedder and computing cosine similarities.


## Prerequisites
- Access to the gated [`mlx-community/embeddinggemma-300m-4bit`](https://huggingface.co/mlx-community/embeddinggemma-300m-4bit) repository on Hugging Face (accept Google’s Gemma license before downloading).

## Getting Started
1. **Clone and resolve dependencies**
   ```bash
   git clone https://github.com/gradinnovate/EmbeddingGemma-MLX.git
   ```
2. **Download model weights (required once)**
   ```bash
   huggingface-cli download mlx-community/embeddinggemma-300m-4bit \
     --local-dir models/embeddinggemma-300m-4bit
   ```

## Running the Swift Sample
The `SwiftTests` package links against the modified `mlx-swift-examples` library to exercise `EmbeddingGemmaModel`.

```bash
cd SwiftTests
swift run EmbeddingGemmaSwiftTest
```

The executable tokenizes a trio of test sentences, generates normalized embeddings, and prints the cosine similarity matrix. You can adapt `SwiftTests/Sources/EmbeddingGemmaSwiftTest/main.swift` to integrate the embedder into your own MLX applications.


## Development Notes
- `mlx-swift-examples/Libraries/Embedders/EmbeddingGemma.swift` contains the production implementation.
- The embedder is registered under the `gemma3_text` model type; any configuration JSON that references this `model_type` now resolves to `EmbeddingGemmaModel`.


## 🙏 Support

If you find this fork helpful for your EmbeddingGemma development, consider supporting the work:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-yellow.svg)](https://buymeacoffee.com/gradinnovate)

## 📄 License

This project maintains the same license as the original MLX Swift Examples repository.
