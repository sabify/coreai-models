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
/// grammar matcher to advance the grammar state.
///
/// When the grammar forces a run of text — a key's name and its closing quote — and the
/// engine can return the logits after each of several tokens (``LogitRowReading``), the
/// run is tokenized as a draft and read with the token before it in one step. Where the
/// grammar forces nothing, the draft is what followed the last few tokens where they
/// stood before, in the prompt or the answer (``CopyDraft``): words a field copies from
/// the capture, a name the answer already wrote. Under greedy sampling each drafted
/// token is kept only where the model, masked, picks it itself; at the first it would
/// not, its own pick is taken. The tokens chosen are those a step per token would choose.
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
                maxTokens: maxTokens,
                stop: options.stop
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

// MARK: - Copy drafts

/// Guesses the next tokens from where the last few stood before: an answer that
/// copies the prompt's words, or repeats its own, continues as it did there. A guess
/// costs nothing the step would not spend — it fills positions the step reads anyway
/// — and is kept only where the model picks it itself.
enum CopyDraft {
    /// The longest run of last tokens looked for; shorter runs are tried after it.
    static let longestMatch = 3

    /// The tokens that followed the latest earlier stand of `history`'s last tokens,
    /// at most `room` of them, the longest match first. A single token is matched
    /// only where it stands once before: a common one ("the", a quote) says little
    /// about what comes next.
    static func tokens(after history: [Int32], room: Int) -> [Int32] {
        guard room > 0, history.count > 1 else { return [] }
        for length in stride(from: Swift.min(longestMatch, history.count - 1), through: 1, by: -1) {
            let suffix = history[(history.count - length)...]
            var latest: Int?
            var stands = 0
            for start in stride(from: history.count - length - 1, through: 0, by: -1)
            where history[start..<(start + length)].elementsEqual(suffix) {
                stands += 1
                if latest == nil { latest = start }
                if length > 1 || stands > 1 { break }
            }
            guard let start = latest, length > 1 || stands == 1 else { continue }
            let from = start + length
            let following = history[from..<Swift.min(from + room, history.count)]
            if !following.isEmpty { return Array(following) }
        }
        return []
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
        let stop: GenerationStop.Ticket?

        init(
            session: ConstrainedSessionHandle, cache: ConstrainedSessionCache?, inputTokens: [Int32], maxTokens: Int,
            stop: GenerationStop.Ticket?
        ) {
            self.session = session
            self.cache = cache
            self.inputTokens = inputTokens
            self.maxTokens = maxTokens
            self.stop = stop
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
        private let stop: GenerationStop.Ticket?
        /// Reads a draft of forced tokens in one step; nil unless the engine can and
        /// sampling is greedy, where a drafted token is kept only if it is the pick.
        private let drafter: (any LogitRowReading)?

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
        /// Tokens a draft read already decided, handed out one per call.
        private var decided: [GenerationResult] = []
        /// Generation ended inside a draft read; it ends once `decided` is out.
        private var endsAfterDecided = false

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
            self.stop = prepared.stop
            self.tokenizer = tokenizer
            self.inferenceEngine = inferenceEngine
            self.samplingConfiguration = samplingConfiguration
            self.stopSequences = stopSequences
            self.drafter =
                samplingConfiguration.isGreedy && !samplingConfiguration.needsRepetitionPenalty
                ? inferenceEngine as? any LogitRowReading : nil
        }

        public func next() async throws -> GenerationResult? {
            if !decided.isEmpty {
                return decided.removeFirst()
            }
            if finished || endsAfterDecided {
                finish()
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
            if stop?.isStopped == true { throw GenerationStopped() }

            if let drafter, let draft = draft(for: drafter) {
                return try await readDraft(draft, with: drafter)
            }

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

        /// A draft to read with the token just picked, when the model has read
        /// everything but it: the text the grammar forces next or, where it forces
        /// none, the tokens that followed the last ones where they stood before
        /// (``CopyDraft``). Either fits the step that reads it.
        private func draft(for drafter: any LogitRowReading) -> [Int32]? {
            guard lastWasSampled, readCount == inputTokens.count - 1 else { return nil }
            let room = drafter.tokensPerStep(after: readCount) - 1
            guard room > 0 else { return nil }
            if let forced = session.findJumpForwardString(), !forced.isEmpty {
                let tokens = tokenizer.encode(text: forced, addSpecialTokens: false).prefix(room).map(Int32.init)
                return tokens.isEmpty ? nil : Array(tokens)
            }
            let copied = CopyDraft.tokens(after: inputTokens, room: room)
            return copied.isEmpty ? nil : copied
        }

        /// Reads the token just picked and `draft` in one step, then keeps each
        /// drafted token the model picks itself under the grammar's mask and, at the
        /// first it would not, the model's own pick. After a fully kept draft, the
        /// last row gives the next token too.
        private func readDraft(_ draft: [Int32], with drafter: any LogitRowReading) async throws -> GenerationResult? {
            let tokens = inputTokens + draft
            async let read = drafter.readLogitRows(tokens, rowCount: draft.count + 1)
            var bitmask = session.nextTokenBitmask()
            let rows = try await read
            let firstDrafted = inputTokens.count
            var keptDraft = 0
            var results: [GenerationResult] = []
            for (index, row) in rows.enumerated() {
                guard tokenStep < maxTokens, let mask = bitmask else { break }
                let vocabularySize = session.vocabularySize
                if let allowed = ConstrainedGenerationSession.allowedTokens(
                    in: mask, vocabularySize: vocabularySize, limit: Self.allowedTokenLimit),
                    !allowed.isEmpty, allowed.allSatisfy({ stopSequences.matches(recentTokens: [$0]) })
                {
                    endsAfterDecided = true
                    break
                }
                var masked = row
                ConstrainedGenerationSession.apply(mask, to: &masked, vocabularySize: vocabularySize)
                let token = samplingConfiguration.sampleToken(from: &masked, step: generatedTokens.count)
                let isDrafted = index < draft.count && token == draft[index]
                guard let result = accept(token, logits: row, sampled: !isDrafted) else {
                    endsAfterDecided = true
                    break
                }
                results.append(result)
                guard isDrafted else { break }
                keptDraft += 1
                bitmask = session.nextTokenBitmask()
            }
            // The engine read the whole draft; the cursor keeps the part that was taken.
            readCount = firstDrafted + keptDraft
            guard !results.isEmpty else {
                endsAfterDecided = true
                return nil
            }
            decided = Array(results.dropFirst())
            return results.first
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
