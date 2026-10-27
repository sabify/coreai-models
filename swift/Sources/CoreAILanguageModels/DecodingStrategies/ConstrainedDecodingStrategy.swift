#if canImport(CoreAI)
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Tokenizers

/// Grammar-constrained decoding strategy using xgrammar.
///
/// Conforms to `DecodingStrategy` so it can be used anywhere a decoding strategy is expected.
/// Streams `GenerationResult` tokens as they are generated, enabling incremental text display.
///
/// Uses xgrammar bitmask enforcement to ensure generated text conforms to a JSON schema.
/// Each position: (1) work out the grammar bitmask for it, while the model reads the
/// token sampled before it; (2) take a token the bitmask forces without a model step,
/// and stop when it allows only stop tokens; (3) otherwise read every token the model
/// has not seen in one step, mask its logits and sample; (4) accept the token in the
/// grammar matcher to advance the grammar state. The tokens chosen are those a step per
/// token would choose: a forced token is the only one any sampler could pick.
///
/// The engine is not reset between requests, so its token history keeps the prefix a new
/// request shares with the last, and an engine that keeps grammar sessions
/// (``ConstrainedSessionCaching``) compiles a schema once.
public struct ConstrainedDecodingStrategy: DecodingStrategy {
    /// The JSON schema that constrains generation output.
    private let jsonSchema: String

    /// Vocabulary size override. If nil, derived from tokenizer at generation time.
    private let vocabSizeOverride: Int?

    /// Most whitespace characters allowed between two JSON tokens; nil when unlimited.
    private let maxWhitespaceCount: Int?

    /// Initialize with a JSON schema string.
    ///
    /// - Parameters:
    ///   - jsonSchema: A valid JSON schema string that constrains output structure
    ///   - vocabSize: Optional vocabulary size override. If nil, derived from tokenizer.
    ///   - maxWhitespaceCount: Most whitespace characters allowed between two JSON tokens (default:
    ///     ``ConstrainedGenerationSession/defaultMaxWhitespaceCount``), or `nil` for no limit.
    public init(
        jsonSchema: String, vocabSize: Int? = nil,
        maxWhitespaceCount: Int? = ConstrainedGenerationSession.defaultMaxWhitespaceCount
    ) {
        self.jsonSchema = jsonSchema
        self.vocabSizeOverride = vocabSize
        self.maxWhitespaceCount = maxWhitespaceCount
    }

    // MARK: - DecodingStrategy conformance

    public func decode(
        from input: Input,
        tokenizer: any Tokenizer,
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        options: InferenceOptions,
        stopSequences: StopSequences
    ) async throws -> ConstrainedDecodedSequence {
        CLILogger.log(
            "Starting constrained decoding strategy with schema",
            component: "ConstrainedDecodingStrategy"
        )

        let cache = (inferenceEngine as? any ConstrainedSessionCaching)?.constrainedSessions
        let session: ConstrainedSessionHandle
        if let cached = cache?.take(schema: jsonSchema, maxWhitespaceCount: maxWhitespaceCount) {
            session = cached
        } else {
            session = ConstrainedSessionHandle(
                session: try Self.createSession(
                    jsonSchema: jsonSchema,
                    vocabSizeOverride: vocabSizeOverride,
                    tokenizer: tokenizer,
                    stopSequences: stopSequences,
                    maxWhitespaceCount: maxWhitespaceCount
                ),
                tokenizer: tokenizer)
        }
        let inputTokens =
            try PromptUtils
            .maybeApplyTokenizerChatTemplate(input, tokenizer: tokenizer)
            .map(Int32.init)
        let maxTokens = options.maxTokens ?? 512

        return ConstrainedDecodedSequence(
            prepared: ConstrainedDecodedSequence.Prepared(
                session: session,
                cache: cache,
                inputTokens: inputTokens,
                maxTokens: maxTokens
            ),
            tokenizer: tokenizer,
            inferenceEngine: inferenceEngine,
            samplingConfiguration: samplingConfiguration,
            stopSequences: stopSequences
        )
    }

    // MARK: - Private helpers

    /// Create a constrained generation session with stop token extraction.
    fileprivate static func createSession(
        jsonSchema: String,
        vocabSizeOverride: Int?,
        tokenizer: any Tokenizer,
        stopSequences: StopSequences,
        maxWhitespaceCount: Int?
    ) throws -> ConstrainedGenerationSession {
        guard let vocabSize = vocabSizeOverride ?? Self.deriveVocabSize(from: tokenizer) else {
            throw InferenceRuntimeError.invalidArgument(
                "Cannot determine vocabulary size from tokenizer. "
                    + "Pass vocabSize explicitly via CoreAIRunner or ModelBundle metadata."
            )
        }

        let singleTokenStops = stopSequences.sequences.filter { $0.count == 1 }.map { $0[0] }
        if stopSequences.sequences.contains(where: { $0.count > 1 }) {
            CLILogger.log(
                "Warning: Multi-token stop sequences not supported by xgrammar, using single-token stops only",
                component: "ConstrainedDecodingStrategy")
        }
        let stopTokenIds: [Int32]? = singleTokenStops.isEmpty ? nil : singleTokenStops

        let session = try ConstrainedGenerationSession(
            jsonSchema: jsonSchema,
            tokenizer: tokenizer,
            vocabSize: vocabSize,
            stopTokenIds: stopTokenIds,
            maxWhitespaceCount: maxWhitespaceCount
        )
        CLILogger.log(
            "Constrained session created (vocabSize=\(vocabSize), stopTokenIds=\(stopTokenIds ?? []))",
            component: "ConstrainedDecodingStrategy")
        return session
    }

    /// Read `tokens` the engine has not processed in one step and return the
    /// logits after the last of them.
    fileprivate static func read(
        _ tokens: [Int32],
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration
    ) async throws -> [LogitsScalarType] {
        for try await output in try await inferenceEngine.generate(
            with: tokens,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: InferenceOptions(maxTokens: 1, includeLogits: true)
        ) {
            if let logits = output.logits { return logits }
            break
        }
        throw ConstrainedGenerationError.generationFailed("No logits returned from engine")
    }

    fileprivate static func computeTextDelta(
        generatedTokens: [Int32],
        previousDecodedText: inout String,
        tokenizer: any Tokenizer,
        tokenStep: Int
    ) -> String {
        let decodeSpan = InstrumentsProfiler.beginDecode(step: tokenStep)
        let fullDecodedText = tokenizer.decode(tokens: generatedTokens.map { Int($0) })
        decodeSpan.end()

        let common = fullDecodedText.commonPrefix(with: previousDecodedText)
        let delta = String(fullDecodedText.dropFirst(common.count))

        if delta.unicodeScalars.contains(where: { $0 == "\u{FFFD}" }) {
            return ""
        }

        previousDecodedText = fullDecodedText
        return delta
    }

    // MARK: - Vocabulary size derivation

    /// Derive vocabulary size from a tokenizer using binary search.
    static func deriveVocabSize(from tokenizer: any Tokenizer) -> Int? {
        var low = 0
        var high = 524_288

        // Binary search for the last valid token ID
        while low < high {
            let mid = (low + high) / 2
            if tokenizer.convertIdToToken(mid) != nil {
                low = mid + 1
            } else {
                high = mid
            }
        }

        if low == 0 {
            CLILogger.log(
                "Warning: Could not determine vocab size from tokenizer — grammar mask may be wrong",
                component: "ConstrainedDecodingStrategy")
            return nil
        }
        return low
    }
}

// MARK: - ConstrainedDecodedSequence

extension ConstrainedDecodingStrategy {
    /// Async sequence of `GenerationResult` produced by `decode()`.
    public struct ConstrainedDecodedSequence: AsyncSequence {
        public typealias Element = GenerationResult
        public typealias Failure = Error

        fileprivate let prepared: Prepared
        let tokenizer: any Tokenizer
        let inferenceEngine: any InferenceEngine
        let samplingConfiguration: SamplingConfiguration
        let stopSequences: StopSequences

        public func makeAsyncIterator() -> Iterator {
            Iterator(
                prepared: prepared,
                tokenizer: tokenizer,
                inferenceEngine: inferenceEngine,
                samplingConfiguration: samplingConfiguration,
                stopSequences: stopSequences
            )
        }
    }
}

extension ConstrainedDecodingStrategy.ConstrainedDecodedSequence {
    /// The grammar session, where it goes back to, the tokenized prompt and the token budget.
    fileprivate final class Prepared {
        let session: ConstrainedSessionHandle
        let cache: ConstrainedSessionCache?
        let inputTokens: [Int32]
        let maxTokens: Int

        init(session: ConstrainedSessionHandle, cache: ConstrainedSessionCache?, inputTokens: [Int32], maxTokens: Int) {
            self.session = session
            self.cache = cache
            self.inputTokens = inputTokens
            self.maxTokens = maxTokens
        }
    }
}

extension ConstrainedDecodingStrategy.ConstrainedDecodedSequence {
    public final class Iterator: AsyncIteratorProtocol {
        public typealias Element = GenerationResult
        public typealias Failure = Error

        /// More allowed tokens than this and the position is neither forced nor
        /// stop-only, which is all the count is asked for.
        private static let allowedTokenLimit = 8

        private let tokenizer: any Tokenizer
        private let inferenceEngine: any InferenceEngine
        private let samplingConfiguration: SamplingConfiguration
        private let stopSequences: StopSequences
        private let session: ConstrainedSessionHandle
        private let cache: ConstrainedSessionCache?
        private let maxTokens: Int

        private var inputTokens: [Int32]
        /// How many of `inputTokens` the engine has read.
        private var readCount = 0
        /// Whether the last token came from the model's logits: the model reads it
        /// whatever comes next, so the read can run beside the grammar.
        private var lastWasSampled = false
        private var generatedTokens: [Int32] = []
        private var previousDecodedText: String = ""
        private var tokenStep: Int = 0
        private var finished: Bool = false

        fileprivate init(
            prepared: ConstrainedDecodingStrategy.ConstrainedDecodedSequence.Prepared,
            tokenizer: any Tokenizer,
            inferenceEngine: any InferenceEngine,
            samplingConfiguration: SamplingConfiguration,
            stopSequences: StopSequences
        ) {
            self.session = prepared.session
            self.cache = prepared.cache
            self.inputTokens = prepared.inputTokens
            self.maxTokens = prepared.maxTokens
            self.tokenizer = tokenizer
            self.inferenceEngine = inferenceEngine
            self.samplingConfiguration = samplingConfiguration
            self.stopSequences = stopSequences
        }

        public func next() async throws -> GenerationResult? {
            if finished {
                return nil
            }
            do {
                guard let result = try await nextToken() else {
                    finish()
                    return nil
                }
                return result
            } catch {
                finish()
                throw error
            }
        }

        /// The next token: forced by the grammar, or sampled from the model's logits.
        /// Every engine step started here ends before this returns.
        private func nextToken() async throws -> GenerationResult? {
            guard tokenStep < maxTokens, !session.isTerminated else { return nil }
            try Task.checkCancellation()

            let bitmask: [Int32]?
            var logits: [LogitsScalarType]?
            if lastWasSampled {
                let (tokens, engine, sampling) = (inputTokens, inferenceEngine, samplingConfiguration)
                async let read = ConstrainedDecodingStrategy.read(
                    tokens, inferenceEngine: engine, samplingConfiguration: sampling)
                bitmask = session.nextTokenBitmask()
                logits = try await read
                readCount = inputTokens.count
            } else {
                bitmask = session.nextTokenBitmask()
            }
            guard let bitmask else { return nil }

            let vocabularySize = session.vocabularySize
            if let allowed = ConstrainedGenerationSession.allowedTokens(
                in: bitmask, vocabularySize: vocabularySize, limit: Self.allowedTokenLimit)
            {
                if !allowed.isEmpty, allowed.allSatisfy({ stopSequences.matches(recentTokens: [$0]) }) {
                    return nil
                }
                if allowed.count == 1, let forced = allowed.first {
                    return accept(forced, logits: nil, sampled: false)
                }
            }

            if readCount < inputTokens.count {
                logits = try await ConstrainedDecodingStrategy.read(
                    inputTokens, inferenceEngine: inferenceEngine, samplingConfiguration: samplingConfiguration)
                readCount = inputTokens.count
            }
            guard let raw = logits else {
                throw ConstrainedGenerationError.generationFailed("No logits returned from engine")
            }
            var masked = raw
            if samplingConfiguration.needsRepetitionPenalty,
                let penalty = samplingConfiguration.repetitionPenalty
            {
                let window =
                    samplingConfiguration.repetitionPenaltyWindow.map { Swift.min($0, generatedTokens.count) }
                    ?? generatedTokens.count
                RepetitionPenaltyProcessor.apply(
                    to: &masked,
                    recentTokenIds: generatedTokens.suffix(window),
                    penalty: Float(penalty)
                )
            }
            ConstrainedGenerationSession.apply(bitmask, to: &masked, vocabularySize: vocabularySize)
            let token = samplingConfiguration.sampleToken(from: &masked, step: generatedTokens.count)
            return accept(token, logits: raw, sampled: true)
        }

        /// Advance the grammar by `token` and hand it out; nil once generation ends.
        private func accept(_ token: Int32, logits: [LogitsScalarType]?, sampled: Bool) -> GenerationResult? {
            guard session.acceptToken(token) else { return nil }
            let terminatedAfterAccept = session.isTerminated
            if stopSequences.matches(recentTokens: [token]) {
                return nil
            }
            inputTokens.append(token)
            generatedTokens.append(token)
            tokenStep += 1
            lastWasSampled = sampled
            if terminatedAfterAccept {
                return nil
            }
            let delta = ConstrainedDecodingStrategy.computeTextDelta(
                generatedTokens: generatedTokens,
                previousDecodedText: &previousDecodedText,
                tokenizer: tokenizer,
                tokenStep: tokenStep
            )
            return GenerationResult(text: delta, tokenId: token, rawLogits: logits)
        }

        /// Ends generation and gives the grammar session back for the next request.
        private func finish() {
            guard !finished else { return }
            finished = true
            cache?.store(session)
        }
    }
}
#endif
