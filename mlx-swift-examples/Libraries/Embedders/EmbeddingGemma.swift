// Copyright © 2025 GradInnovate Inc.

import Foundation
import MLX
import MLXFast
import MLXLinalg
import MLXLMCommon
import MLXNN

public struct EmbeddingGemmaConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let kvHeads: Int
    public let ropeGlobalBaseFreq: Float
    public let ropeLocalBaseFreq: Float
    public let ropeTraditional: Bool
    public let queryPreAttnScalar: Float
    public let slidingWindow: Int
    public let slidingWindowPattern: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeGlobalBaseFreq = "rope_global_base_freq"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "_sliding_window_pattern"
    }

    public init(
        modelType: String,
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        headDim: Int,
        rmsNormEps: Float,
        vocabularySize: Int,
        kvHeads: Int,
        ropeGlobalBaseFreq: Float,
        ropeLocalBaseFreq: Float,
        ropeTraditional: Bool,
        queryPreAttnScalar: Float,
        slidingWindow: Int,
        slidingWindowPattern: Int
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.ropeGlobalBaseFreq = ropeGlobalBaseFreq
        self.ropeLocalBaseFreq = ropeLocalBaseFreq
        self.ropeTraditional = ropeTraditional
        self.queryPreAttnScalar = queryPreAttnScalar
        self.slidingWindow = slidingWindow
        self.slidingWindowPattern = slidingWindowPattern
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 4
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
        self.vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262_144
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
        self.ropeGlobalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeGlobalBaseFreq) ?? 1_000_000.0
        self.ropeLocalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
        self.ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.queryPreAttnScalar =
            try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        self.slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6
    }
}

private class EmbeddingGemmaAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm

    let rope: RoPE

    init(_ config: EmbeddingGemmaConfiguration, layerIdx: Int) {
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(config.queryPreAttnScalar, -0.5)

        let dim = config.hiddenSize
        self._queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        let isSlidingLayer = (layerIdx + 1) % config.slidingWindowPattern != 0
        let baseFreq = isSlidingLayer ? config.ropeLocalBaseFreq : config.ropeGlobalBaseFreq
        self.rope = RoPE(
            dimensions: headDim,
            traditional: config.ropeTraditional,
            base: baseFreq
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x)
        var keys = keyProj(x)
        var values = valueProj(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        queries = queryNorm(queries)
        keys = keyNorm(keys)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        var finalMask = mask
        if case .array(let maskArray) = mask {
            let keySeqLen = keys.shape[2]
            if maskArray.shape.last ?? keySeqLen != keySeqLen {
                let sliced = maskArray[.ellipsis, (-keySeqLen)...]
                finalMask = .array(sliced)
            }
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: finalMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return outputProj(output)
    }
}

private class EmbeddingGemmaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

private class EmbeddingGemmaBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: EmbeddingGemmaAttention
    @ModuleInfo(key: "mlp") var mlp: EmbeddingGemmaMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

    init(_ config: EmbeddingGemmaConfiguration, layerIdx: Int) {
        self._attention.wrappedValue = EmbeddingGemmaAttention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = EmbeddingGemmaMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let inputNorm = inputLayerNorm(x)
        let attnOut = attention(inputNorm, mask: mask, cache: cache)
        let attnNorm = postAttentionLayerNorm(attnOut)
        let residual = Gemma.clipResidual(x, attnNorm)
        let mlpInput = preFeedforwardLayerNorm(residual)
        let mlpOut = mlp(mlpInput)
        let mlpNorm = postFeedforwardLayerNorm(mlpOut)
        return Gemma.clipResidual(residual, mlpNorm)
    }
}

private class EmbeddingGemmaEncoder: Module {
    let config: EmbeddingGemmaConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [EmbeddingGemmaBlock]
    @ModuleInfo var norm: Gemma.RMSNorm

    init(_ config: EmbeddingGemmaConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            EmbeddingGemmaBlock(config, layerIdx: layerIdx)
        }
        self._norm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var hidden: MLXArray
        if let embeddings = inputEmbeddings {
            hidden = embeddings
        } else {
            hidden = embedTokens(inputs)
        }
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        hidden = hidden * scale.asType(hidden.dtype)

        var layerCache = cache
        if layerCache == nil {
            layerCache = Array(repeating: nil, count: layers.count)
        }

        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        var slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        if mask == nil {
            let j = config.slidingWindowPattern
            var globalMaskCaches: [KVCache]? = nil

            if j > 0,
                let caches = layerCache,
                j <= caches.count,
                let globalCache = caches[j - 1]
            {
                globalMaskCaches = [globalCache]
            }

            fullMask = createAttentionMask(h: hidden, cache: globalMaskCaches)

            let allCaches = layerCache?.compactMap { $0 } ?? []
            let slidingCaches = allCaches.isEmpty ? nil : allCaches
            slidingWindowMask = createAttentionMask(h: hidden, cache: slidingCaches)
        }

        for (index, layer) in layers.enumerated() {
            let isGlobalLayer =
                index % config.slidingWindowPattern == config.slidingWindowPattern - 1

            let localMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                localMask = mask
            } else if isGlobalLayer {
                localMask = fullMask
            } else {
                localMask = slidingWindowMask
            }

            let cacheEntry: KVCache? = layerCache?[index] ?? nil
            hidden = layer(hidden, mask: localMask, cache: cacheEntry)
        }
        return norm(hidden)
    }
}

public class EmbeddingGemmaModel: Module, EmbeddingModel {
    @ModuleInfo(key: "model") private var encoder: EmbeddingGemmaEncoder
    @ModuleInfo(key: "dense") private var dense: [Linear]

    public let vocabularySize: Int
    public let config: EmbeddingGemmaConfiguration

    public init(_ config: EmbeddingGemmaConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabularySize
        self._encoder.wrappedValue = EmbeddingGemmaEncoder(config)
        self._dense.wrappedValue = [
            Linear(config.hiddenSize, config.hiddenSize * 4, bias: false),
            Linear(config.hiddenSize * 4, config.hiddenSize, bias: false)
        ]
        super.init()
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        positionIds: MLXArray?,
        tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) -> EmbeddingModelOutput {
        let (batch, sequenceLength) = (inputIds.dim(0), inputIds.dim(1))
        var mask = attentionMask
        if mask == nil {
            mask = MLXArray.ones([batch, sequenceLength], dtype: .float32)
        }

        guard let validMask = mask else {
            fatalError("Failed to construct attention mask.")
        }

        let extendedMask = makeExtendedAttentionMask(validMask)
        let maskMode = MLXFast.ScaledDotProductAttentionMaskMode.array(extendedMask)
        var hiddenStates = encoder(
            inputIds,
            mask: maskMode,
            cache: nil,
            inputEmbeddings: nil
        )
        hiddenStates = dense[0](hiddenStates)
        hiddenStates = dense[1](hiddenStates)

        let pooled = meanPool(hiddenStates, attentionMask: validMask)
        let normalized = normalizeEmbeddings(pooled)

        return EmbeddingModelOutput(
            hiddenStates: hiddenStates,
            pooledOutput: normalized
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        let blockDensePattern = try! NSRegularExpression(
            pattern: #"model\.layers\.(\d+)\.mlp\.(\d+)_Dense\.(\w+)"#
        )

        let rootDensePattern = try! NSRegularExpression(
            pattern: #"^dense\.(\d+)\.(\w+)"#
        )

        for (key, value) in weights {
            var newKey = key

            if let match = blockDensePattern.firstMatch(
                in: key,
                options: [],
                range: NSRange(location: 0, length: key.utf16.count)
            ) {
                let layerIndex = String(key[Range(match.range(at: 1), in: key)!])
                let denseIndex = String(key[Range(match.range(at: 2), in: key)!])
                let component = String(key[Range(match.range(at: 3), in: key)!])

                let denseName: String
                switch denseIndex {
                case "0": denseName = "gate_proj"
                case "1": denseName = "up_proj"
                case "2": denseName = "down_proj"
                default: denseName = "dense_\(denseIndex)"
                }

                newKey = "model.layers.\(layerIndex).mlp.\(denseName).\(component)"
            } else if let match = rootDensePattern.firstMatch(
                in: key,
                options: [],
                range: NSRange(location: 0, length: key.utf16.count)
            ) {
                let denseIndex = String(key[Range(match.range(at: 1), in: key)!])
                let component = String(key[Range(match.range(at: 2), in: key)!])
                newKey = "dense.\(denseIndex).\(component)"
            }

            print("sanitize: \(key) -> \(newKey)")
            sanitized[newKey] = value
        }

        return sanitized
    }
}

private func meanPool(_ tokenEmbeddings: MLXArray, attentionMask: MLXArray) -> MLXArray {
    var mask = attentionMask
    if mask.dtype != tokenEmbeddings.dtype {
        mask = mask.asType(tokenEmbeddings.dtype)
    }

    let expandedMask = mask.expandedDimensions(axes: [-1])
    let sumEmbeddings = sum(tokenEmbeddings * expandedMask, axis: 1)
    let sumMask = sum(expandedMask, axis: 1) + MLXArray(1e-9, dtype: tokenEmbeddings.dtype)
    return sumEmbeddings / sumMask
}

private func normalizeEmbeddings(_ embeddings: MLXArray, epsilon: Float = 1e-9) -> MLXArray {
    let normValues = MLXLinalg.norm(embeddings, ord: 2, axis: -1, keepDims: true)
    return embeddings / (normValues + MLXArray(epsilon, dtype: embeddings.dtype))
}

private func makeExtendedAttentionMask(_ attentionMask: MLXArray) -> MLXArray {
    switch attentionMask.ndim {
    case 3:
        return attentionMask.expandedDimensions(axes: [1])
    case 2:
        let seqLength = attentionMask.dim(1)
        var extended = attentionMask.expandedDimensions(axes: [1, 2])
        extended = tiled(extended, repetitions: [1, 1, seqLength, 1])
        return extended
    default:
        fatalError("Unsupported attention mask shape \(attentionMask.shape)")
    }
}
