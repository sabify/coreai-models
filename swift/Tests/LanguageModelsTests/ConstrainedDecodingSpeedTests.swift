// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

#if canImport(CoreAI)
import Foundation
import Synchronization
import Testing
import TestUtilities
import Tokenizers

@testable import CoreAILanguageModels

// MARK: - Bitmask helpers

@Suite("Constrained decoding bitmask helpers")
struct ConstrainedBitmaskHelperTests {
    /// The plain per-token loop the fast one must match.
    private func reference(_ bitmask: [Int32], _ logits: [Float], vocabularySize: Int) -> [Float] {
        var out = logits
        for token in 0..<min(logits.count, vocabularySize) {
            let (word, bit) = token.quotientAndRemainder(dividingBy: 32)
            if word >= bitmask.count || bitmask[word] & (1 << bit) == 0 { out[token] = -.infinity }
        }
        return out
    }

    @Test("The fast mask blocks exactly what the plain loop blocks")
    func applyMatchesReference() {
        var generator = SystemRandomNumberGenerator()
        for vocabularySize in [1, 31, 32, 33, 100, 151_936] {
            let words = (vocabularySize + 31) / 32
            for pattern in 0..<4 {
                let bitmask: [Int32] = (0..<words).map { _ in
                    switch pattern {
                    case 0: 0
                    case 1: -1
                    default: Int32(truncatingIfNeeded: generator.next() as UInt64)
                    }
                }
                let logits = (0..<vocabularySize).map { Float($0 % 97) }
                var fast = logits
                ConstrainedGenerationSession.apply(bitmask, to: &fast, vocabularySize: vocabularySize)
                #expect(fast == reference(bitmask, logits, vocabularySize: vocabularySize), "\(vocabularySize)/\(pattern)")
            }
        }
    }

    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    @Test("Float16 logits are blocked at the most negative finite value")
    func applyFloat16() {
        let bitmask: [Int32] = [0b1010]
        var logits: [Float16] = [1, 2, 3, 4, 5]
        ConstrainedGenerationSession.apply(bitmask, to: &logits, vocabularySize: 5)
        let blocked = -Float16.greatestFiniteMagnitude
        #expect(logits == [blocked, 2, blocked, 4, blocked])
    }
    #endif

    @Test("Allowed tokens are listed up to the limit, and nil beyond it")
    func allowedTokensUpToLimit() {
        // Tokens 3, 40 and 70 of a 71-token vocabulary; bit 7 of the last word is past its end.
        let bitmask: [Int32] = [1 << 3, 1 << 8, (1 << 6) | (1 << 7)]
        #expect(ConstrainedGenerationSession.allowedTokens(in: bitmask, vocabularySize: 71, limit: 8) == [3, 40, 70])
        #expect(ConstrainedGenerationSession.allowedTokens(in: bitmask, vocabularySize: 71, limit: 2) == nil)
        #expect(ConstrainedGenerationSession.allowedTokens(in: [0, 0, 0], vocabularySize: 71, limit: 8) == [])
        #expect(ConstrainedGenerationSession.allowedTokens(in: [1 << 5, 0, 0], vocabularySize: 71, limit: 1) == [5])
    }
}

// MARK: - Decoding loop

/// A schema-shaped toy: structure, a few words and numbers.
private let toyVocabulary = [
    "{", "}", "[", "]", ":", ",", "\"", " ", "\n", "\t",
    "name", "age", "email", "city", "true", "false", "null",
    "John", "Jane", "Doe", "Smith", "user", "admin", "guest", "status", "active",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20", "25",
]

private let personSchema = """
    {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
     "required": ["name", "age"], "additionalProperties": false}
    """

/// `{"name":"John","age":25}` in the toy vocabulary. The grammar forces the opening
/// brace, `name` and the quote after it, and `age` and the quote after it: five of
/// fifteen tokens, in three runs.
private let expectedAnswer: [Int32] = [0, 6, 10, 6, 4, 6, 17, 6, 5, 6, 11, 6, 4, 38, 1]
private let prompt: [Int32] = [26, 27, 28]

/// An engine whose logits after each read favour the token the script puts next,
/// recording how many tokens each read was handed.
private final class ScriptedLogitsEngine: InferenceEngine, ConstrainedSessionCaching, @unchecked Sendable {
    struct Config: Codable, InferenceConfiguration {
        var maxContextLength: Int
    }

    let config = Config(maxContextLength: 4096)
    let constrainedSessions = ConstrainedSessionCache()
    private let script: [Int32]
    private let vocabularySize: Int
    private let log = Mutex<(reads: [Int], resets: Int)>(([], 0))

    var supportsLogits: Bool { true }
    var reads: [Int] { log.withLock { $0.reads } }
    var resets: Int { log.withLock { $0.resets } }

    init(script: [Int32], vocabularySize: Int) {
        self.script = script
        self.vocabularySize = vocabularySize
    }

    func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> Output {
        log.withLock { $0.reads.append(input.count) }
        var logits = [LogitsScalarType](repeating: 0, count: vocabularySize)
        if input.count < script.count { logits[Int(script[input.count])] = 10 }
        return Output(logits: logits)
    }

    func reset(to tokenIndex: Int) async throws {
        log.withLock { $0.resets += 1 }
    }

    struct Output: InferenceOutputSequence {
        typealias Element = InferenceOutput
        typealias Failure = any Error

        let logits: [LogitsScalarType]
        var stopReason: StopReason? { nil }
        func setStopReason(_ reason: StopReason) {}

        func makeAsyncIterator() -> Iterator { Iterator(logits: logits) }

        struct Iterator: AsyncIteratorProtocol {
            var logits: [LogitsScalarType]?

            mutating func next() async throws -> InferenceOutput? {
                guard let logits else { return nil }
                self.logits = nil
                return InferenceOutput(tokenId: 0, logits: logits)
            }
        }
    }
}

@Suite("Constrained decoding loop")
struct ConstrainedDecodingLoopTests {
    private let tokenizer = MockTokenizer(
        vocab: Dictionary(uniqueKeysWithValues: toyVocabulary.enumerated().map { ($1, $0) }))

    private func decode(on engine: ScriptedLogitsEngine) async throws -> [Int32] {
        let strategy = ConstrainedDecodingStrategy(jsonSchema: personSchema, vocabSize: toyVocabulary.count)
        let stream = try await strategy.decode(
            from: .tokens(prompt.map(Int.init)),
            tokenizer: tokenizer,
            inferenceEngine: engine,
            samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: 64),
            stopSequences: StopSequences(sequences: [])
        )
        var tokens: [Int32] = []
        for try await result in stream { tokens.append(result.tokenId) }
        return tokens
    }

    @Test("Forced tokens are taken without a model step and read together, and the answer is the one a step per token gives")
    func forcedTokensSkipTheModel() async throws {
        let engine = ScriptedLogitsEngine(script: prompt + expectedAnswer, vocabularySize: toyVocabulary.count)
        let answer = try await decode(on: engine)
        #expect(answer == expectedAnswer)
        // A step per token reads once per position, sixteen in all with the end. Each
        // forced run is read with the token after it: three fewer reads, and the
        // prompt is read together with the forced brace.
        #expect(engine.reads.count == expectedAnswer.count + 1 - 3)
        #expect(engine.reads.first == prompt.count + 1)
        // `name` and its quote are read in one step, never one at a time.
        #expect(engine.reads.contains(prompt.count + 4))
        #expect(!engine.reads.contains(prompt.count + 3))
    }

    @Test("A request does not reset the engine, so its history can keep a shared prefix")
    func noResetBetweenRequests() async throws {
        let engine = ScriptedLogitsEngine(script: prompt + expectedAnswer, vocabularySize: toyVocabulary.count)
        _ = try await decode(on: engine)
        _ = try await decode(on: engine)
        #expect(engine.resets == 0)
    }

    @Test("The grammar session is kept on the engine and reused, reset, by the next request")
    func sessionIsReused() async throws {
        let engine = ScriptedLogitsEngine(script: prompt + expectedAnswer, vocabularySize: toyVocabulary.count)
        _ = try await decode(on: engine)
        let kept = try #require(engine.constrainedSessions.take(
            schema: personSchema, maxWhitespaceCount: ConstrainedGenerationSession.defaultMaxWhitespaceCount))
        engine.constrainedSessions.store(kept)

        let again = try await decode(on: engine)
        #expect(again == expectedAnswer)
        let keptAgain = try #require(engine.constrainedSessions.take(
            schema: personSchema, maxWhitespaceCount: ConstrainedGenerationSession.defaultMaxWhitespaceCount))
        #expect(keptAgain === kept)
    }

    @Test("The session cache holds a few schemas, dropping the least recently stored")
    func cacheIsBounded() throws {
        let cache = ConstrainedSessionCache()
        var handles: [ConstrainedSessionHandle] = []
        for index in 0...ConstrainedSessionCache.capacity {
            let schema = """
                {"type": "object", "properties": {"f\(index)": {"type": "string"}}, "required": ["f\(index)"]}
                """
            let session = try ConstrainedGenerationSession(
                jsonSchema: schema, vocabulary: toyVocabulary, vocabType: .raw)
            let handle = ConstrainedSessionHandle(session: session, tokenizer: tokenizer)
            handles.append(handle)
            cache.store(handle)
        }
        #expect(cache.take(schema: handles[0].schema, maxWhitespaceCount: handles[0].maxWhitespaceCount) == nil)
        #expect(cache.take(schema: handles[1].schema, maxWhitespaceCount: handles[1].maxWhitespaceCount) === handles[1])
    }
}
#endif
