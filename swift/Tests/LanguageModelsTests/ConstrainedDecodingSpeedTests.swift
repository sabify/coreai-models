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

// MARK: - Drafts of forced text

/// A character vocabulary: token `b` is the character with code `b`, as
/// `MockTokenizer` encodes text, plus one merged token for `": "`.
private let mergedColonToken: Int32 = 128
private let characterTokens: [String] = (0...Int(mergedColonToken)).map { id in
    if id == Int(mergedColonToken) { return "\": \"" }
    return (32..<127).contains(id) ? String(UnicodeScalar(UInt8(id))) : ""
}
private let characterVocabulary: [String: Int] = Dictionary(
    uniqueKeysWithValues: characterTokens.enumerated().filter { !$0.element.isEmpty }.map { ($0.element, $0.offset) })

/// Logits that favour, after each token, the token a script puts next, read one
/// step at a time or as rows; it counts both kinds of read.
private final class ScriptedRowsEngine: InferenceEngine, LogitRowReading, @unchecked Sendable {
    struct Config: Codable, InferenceConfiguration {
        var maxContextLength: Int
    }

    let config = Config(maxContextLength: 4096)
    private let script: [Int32]
    private let vocabularySize: Int
    private let log = Mutex<(steps: Int, rowReads: Int)>((0, 0))

    var supportsLogits: Bool { true }
    var steps: Int { log.withLock { $0.steps } }
    var rowReads: Int { log.withLock { $0.rowReads } }

    init(script: [Int32], vocabularySize: Int) {
        self.script = script
        self.vocabularySize = vocabularySize
    }

    private func row(after position: Int) -> [LogitsScalarType] {
        var logits = [LogitsScalarType](repeating: 0, count: vocabularySize)
        if position + 1 < script.count { logits[Int(script[position + 1])] = 10 }
        return logits
    }

    func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> ScriptedLogitsEngine.Output {
        log.withLock { $0.steps += 1 }
        return ScriptedLogitsEngine.Output(logits: row(after: input.count - 1))
    }

    func tokensPerStep(after processed: Int) -> Int { 8 - processed % 8 }

    func readLogitRows(_ input: [Int32], rowCount: Int) async throws -> [[LogitsScalarType]] {
        log.withLock {
            $0.steps += 1
            $0.rowReads += 1
        }
        return ((input.count - rowCount)..<input.count).map(row(after:))
    }

    func reset(to tokenIndex: Int) async throws {}
}

@Suite("Constrained decoding drafts")
struct ConstrainedDecodingDraftTests {
    private let tokenizer = MockTokenizer(vocab: characterVocabulary)
    private let schema = """
        {"type": "object", "properties": {"name": {"type": "string"}, "city": {"type": "string"}},
         "required": ["name", "city"], "additionalProperties": false}
        """

    private func decode(_ answer: [Int32]) async throws -> (tokens: [Int32], engine: ScriptedRowsEngine) {
        let prompt: [Int32] = [72, 105]
        let engine = ScriptedRowsEngine(script: prompt + answer, vocabularySize: characterTokens.count)
        let strategy = ConstrainedDecodingStrategy(jsonSchema: schema, vocabSize: characterTokens.count)
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
        return (tokens, engine)
    }

    private func characters(_ text: String) -> [Int32] { text.utf8.map(Int32.init) }

    @Test("Forced text after a picked token is drafted, read in one step and kept where the model picks it")
    func draftsAreReadInOneStep() async throws {
        let answer = characters(#"{"name": "Jo", "city": "Rome"}"#)
        let (tokens, engine) = try await decode(answer)
        #expect(tokens == answer)
        #expect(engine.rowReads > 0)
        // Fewer reads than tokens: each key's name and closing quote, which the
        // grammar forces once its opening quote is picked, is read in a draft.
        #expect(engine.steps < answer.count)
    }

    @Test("Where the model picks another token than the draft, its own pick is taken")
    func aRefusedDraftKeepsTheModelsToken() async throws {
        // The model writes the merged `": "` after `city`; the draft spells it out.
        let answer = characters(#"{"name": "Jo", "city"#) + [mergedColonToken] + characters(#"Rome"}"#)
        let (tokens, _) = try await decode(answer)
        #expect(tokens == answer)
    }
}
#endif
