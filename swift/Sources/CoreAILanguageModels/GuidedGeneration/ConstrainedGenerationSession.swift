#if canImport(CoreAI)
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Tokenizers

// MARK: - Constrained Generation Session

/// High-level API for constrained (guided) text generation using JSON schemas.
///
/// `ConstrainedGenerationSession` wraps the low-level xgrammar primitives into a
/// simple API that can be integrated into app code. It manages the grammar matcher
/// lifecycle, bitmask allocation, and logit masking.
///
/// Each session is tied to a specific JSON schema and vocabulary. It tracks
/// the generation state and produces token masks that enforce schema compliance.
public struct ConstrainedGenerationSession: ~Copyable {
    static let maxRollbackTokens = 64

    /// Most whitespace characters allowed between two JSON tokens unless a caller passes another
    /// limit. The limit ends a model that keeps choosing whitespace after the last value. It is far
    /// above what models write: pretty-printed output uses a newline and two spaces per nesting
    /// level, so 32 covers 15 levels, or 7 at four spaces per level. The session enforces the limit
    /// on the next-token mask, so it does not enlarge the compiled grammar.
    public static let defaultMaxWhitespaceCount = 32

    private let tokenizerInfo: TokenizerInfo
    private let compiler: GrammarCompiler
    private let compiledGrammar: CompiledGrammar
    private let matcher: GrammarMatcher
    public let vocabularySize: Int
    private let bitmaskSize: Int
    private var bitmaskBuffer: [Int32]
    // Tracks termination via all-zeros bitmask (xgrammar signals completion this way
    // when no EOS token is present, returning false from fillNextTokenBitmask).
    private var allTokensBlocked = false
    /// Enforces ``maxWhitespaceCount`` on the next-token mask; nil when unlimited.
    private var whitespaceGuard: WhitespaceGuard?

    /// The JSON schema this session enforces.
    public let schema: String

    /// Most whitespace characters allowed between two JSON tokens; nil when unlimited. With 0 the
    /// grammar allows none and writes the separators itself — `", "` between items and `": "` after
    /// a key, as Python's `json.dumps` does — so a model trained on that spacing never samples them.
    public let maxWhitespaceCount: Int?

    /// Whether the grammar has reached a terminal state (valid JSON complete).
    ///
    /// Returns true either when xgrammar explicitly marks the matcher as terminated,
    /// or when `fillNextTokenBitmask` returns false (all tokens blocked = JSON complete).
    public var isTerminated: Bool {
        matcher.isTerminated || allTokensBlocked
    }

    /// Memory used by the compiled grammar in bytes.
    public var compiledGrammarMemoryBytes: Int {
        compiledGrammar.memorySizeBytes
    }

    /// Initialize a constrained generation session with a JSON schema.
    ///
    /// - Parameters:
    ///   - jsonSchema: A valid JSON schema string
    ///   - vocabulary: The full tokenizer vocabulary (one entry per token ID)
    ///   - vocabType: The vocabulary encoding type (default: `.byteLevel`)
    ///   - stopTokenIds: Token IDs to treat as stop tokens (e.g., EOS).
    ///     xgrammar allows these only at grammar-terminal states (valid JSON complete).
    ///     Pass `nil` to rely on xgrammar defaults.
    ///   - maxWhitespaceCount: Most whitespace characters allowed between two JSON tokens (default:
    ///     ``defaultMaxWhitespaceCount``), 0 for the separators alone (see ``maxWhitespaceCount``), or
    ///     `nil` for no limit. Must not be negative.
    /// - Throws: `ConstrainedGenerationError` if the schema is invalid JSON or the limit is negative
    public init(
        jsonSchema: String,
        vocabulary: [String],
        vocabType: VocabularyType = .byteLevel,
        stopTokenIds: [Int32]? = nil,
        maxWhitespaceCount: Int? = ConstrainedGenerationSession.defaultMaxWhitespaceCount
    ) throws {
        try Self.validateJSONSchema(jsonSchema)
        try Self.validateMaxWhitespaceCount(maxWhitespaceCount)
        let tokenizerInfo = TokenizerInfo(
            vocabulary: vocabulary,
            vocabType: vocabType
        )
        try self.init(
            jsonSchema: jsonSchema, tokenizerInfo: tokenizerInfo, maxWhitespaceCount: maxWhitespaceCount,
            validated: true)
    }

    /// Initialize with a pre-built `TokenizerInfo` (useful when caching across sessions).
    ///
    /// - Parameters:
    ///   - jsonSchema: A valid JSON schema string
    ///   - tokenizerInfo: Pre-built tokenizer info (must include stop token IDs if needed)
    ///   - maxWhitespaceCount: Most whitespace characters allowed between two JSON tokens (default:
    ///     ``defaultMaxWhitespaceCount``), 0 for the separators alone (see ``maxWhitespaceCount``), or
    ///     `nil` for no limit. Must not be negative.
    /// - Throws: `ConstrainedGenerationError` if the schema is invalid JSON or the limit is negative,
    ///           `XGrammarError` if schema compilation fails
    public init(
        jsonSchema: String,
        tokenizerInfo: TokenizerInfo,
        maxWhitespaceCount: Int? = ConstrainedGenerationSession.defaultMaxWhitespaceCount
    ) throws {
        try Self.validateJSONSchema(jsonSchema)
        try Self.validateMaxWhitespaceCount(maxWhitespaceCount)
        try self.init(
            jsonSchema: jsonSchema, tokenizerInfo: tokenizerInfo, maxWhitespaceCount: maxWhitespaceCount,
            validated: true)
    }

    /// Internal init that skips validation (caller must validate first).
    private init(
        jsonSchema: String,
        tokenizerInfo: TokenizerInfo,
        maxWhitespaceCount: Int?,
        validated: Bool
    ) throws {
        self.schema = jsonSchema
        self.maxWhitespaceCount = maxWhitespaceCount
        self.tokenizerInfo = tokenizerInfo
        self.compiler = GrammarCompiler(tokenizerInfo: tokenizerInfo)
        // Without free whitespace, xgrammar's grammar writes its default separators, ", " and ": ".
        self.compiledGrammar = try compiler.compileJSONSchema(jsonSchema, anyWhitespace: maxWhitespaceCount != 0)
        self.whitespaceGuard = maxWhitespaceCount.flatMap { limit in
            limit > 0
                ? WhitespaceGuard(limit: limit, vocabulary: tokenizerInfo.vocabulary, vocabType: tokenizerInfo.vocabType)
                : nil
        }
        self.matcher = GrammarMatcher(compiledGrammar: compiledGrammar, maxRollbackTokens: Self.maxRollbackTokens)
        self.vocabularySize = tokenizerInfo.vocabularySize
        self.bitmaskSize = (vocabularySize + 31) / 32
        self.bitmaskBuffer = Array(repeating: 0, count: bitmaskSize)
    }

    /// Validate that a string is valid JSON.
    private static func validateMaxWhitespaceCount(_ limit: Int?) throws {
        if let limit, limit < 0 {
            throw ConstrainedGenerationError.invalidMaxWhitespaceCount(limit)
        }
    }

    private static func validateJSONSchema(_ schema: String) throws {
        guard let data = schema.data(using: .utf8),
            (try? JSONSerialization.jsonObject(with: data)) != nil
        else {
            throw ConstrainedGenerationError.invalidSchema(schema)
        }
    }

    // MARK: - Tokenizer Init

    /// Initialize from a `Tokenizer` (swift-transformers protocol).
    ///
    /// Extracts vocabulary automatically from the tokenizer.
    ///
    /// - Parameters:
    ///   - jsonSchema: A valid JSON schema string
    ///   - tokenizer: A swift-transformers Tokenizer for vocabulary extraction
    ///   - vocabSize: Size of the tokenizer vocabulary
    ///   - vocabType: The vocabulary encoding type (default: inferred from the vocabulary with
    ///     `VocabularyType.detect(vocabulary:)`)
    ///   - stopTokenIds: Token IDs to treat as stop tokens (e.g., EOS).
    ///     xgrammar allows these tokens only at grammar-terminal states (valid JSON complete),
    ///     blocking them mid-generation. Pass `nil` to rely on xgrammar defaults (not recommended).
    ///   - maxWhitespaceCount: Most whitespace characters allowed between two JSON tokens (default:
    ///     ``defaultMaxWhitespaceCount``), 0 for the separators alone (see ``maxWhitespaceCount``), or
    ///     `nil` for no limit. Must not be negative.
    /// - Throws: `ConstrainedGenerationError` if the schema is invalid JSON or the limit is negative
    public init(
        jsonSchema: String,
        tokenizer: any Tokenizer,
        vocabSize: Int,
        vocabType: VocabularyType? = nil,
        stopTokenIds: [Int32]? = nil,
        maxWhitespaceCount: Int? = ConstrainedGenerationSession.defaultMaxWhitespaceCount
    ) throws {
        // Extract vocabulary from tokenizer
        var vocabulary: [String] = []
        vocabulary.reserveCapacity(vocabSize)
        for i in 0..<vocabSize {
            // xgrammar handles empty strings for missing token IDs; many tokenizers
            // have gaps in their ID space, so nil here is expected, not an error.
            vocabulary.append(tokenizer.convertIdToToken(i) ?? "")
        }
        try self.init(
            jsonSchema: jsonSchema, vocabulary: vocabulary,
            vocabType: vocabType ?? VocabularyType.detect(vocabulary: vocabulary),
            stopTokenIds: stopTokenIds, maxWhitespaceCount: maxWhitespaceCount)
    }

    // MARK: - Raw Bitmask Access (for MaskProviding integration)

    /// Get the bitmask of allowed tokens for the next generation step.
    ///
    /// Each bit in the returned array represents whether a token is allowed (1) or not (0).
    /// Bit `i` in word `i/32` at position `i%32`.
    ///
    /// - Returns: The bitmask array, or nil if the grammar is terminated
    public mutating func nextTokenBitmask() -> [Int32]? {
        if isTerminated { return nil }

        let hasConstraints = bitmaskBuffer.withUnsafeMutableBufferPointer { buffer in
            let constrained = matcher.fillNextTokenBitmask(buffer.baseAddress!)
            if constrained { whitespaceGuard?.restrict(buffer.baseAddress!) }
            return constrained
        }
        // xgrammar may signal completion either by returning false, or by filling
        // an all-zeros bitmask (no tokens allowed). Both indicate the grammar is done.
        if !hasConstraints || Self.allowsNone(bitmaskBuffer) {
            allTokensBlocked = true
            return nil
        }
        return bitmaskBuffer
    }

    // MARK: - Logit Masking

    /// Apply the grammar mask to Float logits in-place (disallowed tokens → `-.infinity`).
    ///
    /// - Parameter logits: Raw logits from the model (modified in-place, length = vocab size)
    /// - Returns: `true` if mask was applied, `false` if terminated (logits unchanged)
    @discardableResult
    public mutating func applyMask(to logits: inout [Float]) -> Bool {
        guard let bitmask = nextTokenBitmask() else { return false }
        Self.apply(bitmask, to: &logits, vocabularySize: vocabularySize)
        return true
    }

    /// Accept a generated token and advance the grammar state.
    ///
    /// Call this after sampling a token from the masked logits.
    ///
    /// - Parameter tokenId: The token ID that was sampled
    /// - Returns: `true` if the token was accepted, `false` if rejected by the grammar
    @discardableResult
    public mutating func acceptToken(_ tokenId: Int32) -> Bool {
        let accepted = matcher.acceptToken(tokenId)
        if accepted { whitespaceGuard?.accept(tokenId) }
        return accepted
    }

    /// Reset the session to the initial state for reuse with the same schema.
    public mutating func reset() {
        matcher.reset()
        whitespaceGuard?.reset()
        allTokensBlocked = false
    }

    /// Rollback the grammar state by N tokens. Returns false if rollback failed
    /// (e.g., exceeds maxRollbackTokens budget).
    @discardableResult
    public mutating func rollback(_ numTokens: Int = 1) -> Bool {
        guard numTokens >= 0 else { return false }
        let rolledBack = matcher.rollback(numTokens)
        if rolledBack { whitespaceGuard?.rollback(numTokens) }
        return rolledBack
    }

    /// Find the longest deterministic string from the current grammar state.
    /// Does not change the matcher state. Returns nil if no jump-forward is possible.
    public func findJumpForwardString() -> String? {
        matcher.findJumpForwardString()
    }

    /// Result of filling a bitmask for the next token.
    public enum BitmaskResult: Equatable {
        /// Grammar is terminated or all tokens are blocked — generation should stop.
        case terminated
        /// All tokens are allowed — no mask needed, generate unconstrained.
        case unconstrained
        /// Bitmask was written — apply it to constrain sampling.
        case constrained
    }

    /// Fill the bitmask directly into a caller-provided buffer (e.g., a GPU-visible MTLBuffer).
    ///
    /// The caller must ensure the pointer has room for at least `(vocabularySize + 31) / 32`
    /// Int32 words.
    public mutating func fillBitmask(into pointer: UnsafeMutablePointer<Int32>) -> BitmaskResult {
        if isTerminated { return .terminated }

        let needsApplication = matcher.fillNextTokenBitmask(pointer)
        if !needsApplication {
            // xgrammar signals all tokens are allowed — no mask needed
            return .unconstrained
        }
        whitespaceGuard?.restrict(pointer)
        // Check for all-zeros (no tokens allowed — grammar done)
        for i in 0..<bitmaskSize {
            if pointer[i] != 0 { return .constrained }
        }
        allTokensBlocked = true
        return .terminated
    }
}

// MARK: - Float16 Masking

#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
extension ConstrainedGenerationSession {
    /// Apply the grammar mask to Float16 logits in-place, setting disallowed tokens to
    /// `-Float16.greatestFiniteMagnitude`.
    ///
    /// - Parameter logits: Raw logits from the model (modified in-place, length = vocab size)
    /// - Returns: `true` if mask was applied, `false` if terminated (logits unchanged)
    @discardableResult
    public mutating func applyMask(to logits: inout [Float16]) -> Bool {
        guard let bitmask = nextTokenBitmask() else { return false }
        Self.apply(bitmask, to: &logits, vocabularySize: vocabularySize)
        return true
    }

    /// Apply a bitmask from ``nextTokenBitmask()`` to Float16 logits in place,
    /// setting disallowed tokens to `-Float16.greatestFiniteMagnitude`: the mask
    /// can be worked out while the model computes the logits.
    public static func apply(_ bitmask: [Int32], to logits: inout [Float16], vocabularySize: Int) {
        bitmask.withUnsafeBufferPointer { words in
            logits.withUnsafeMutableBufferPointer { values in
                blockDisallowed(
                    words, in: values, vocabularySize: vocabularySize, blocked: -Float16.greatestFiniteMagnitude)
            }
        }
    }
}

private func blockDisallowed(
    _ words: UnsafeBufferPointer<Int32>, in logits: UnsafeMutableBufferPointer<Float16>, vocabularySize: Int,
    blocked: Float16
) {
    guard let mask = words.baseAddress, let values = logits.baseAddress else { return }
    let count = min(logits.count, vocabularySize)
    var word = 0
    while word < words.count {
        let first = word &* 32
        if first >= count { break }
        let allowed = UInt32(bitPattern: mask[word])
        if allowed != UInt32.max {
            let end = min(first &+ 32, count)
            if allowed == 0 {
                (values + first).update(repeating: blocked, count: end &- first)
            } else {
                var disallowed = ~allowed
                while disallowed != 0 {
                    let token = first &+ disallowed.trailingZeroBitCount
                    if token >= end { break }
                    values[token] = blocked
                    disallowed &= disallowed &- 1
                }
            }
        }
        word &+= 1
    }
}
#endif

// MARK: - Errors

public enum ConstrainedGenerationError: Error, LocalizedError {
    case invalidSchema(String)
    case invalidMaxWhitespaceCount(Int)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidSchema(let schema):
            return "Invalid JSON schema: \(schema.prefix(100))..."
        case .invalidMaxWhitespaceCount(let value):
            return "The whitespace limit must not be negative; 0 allows none, nil no limit (got \(value))"
        case .generationFailed(let reason):
            return "Constrained generation failed: \(reason)"
        }
    }
}

// MARK: - Convenience: Schema from File

extension ConstrainedGenerationSession {
    /// Create a session by loading a JSON schema from a file.
    ///
    /// - Parameters:
    ///   - schemaPath: Path to a JSON schema file
    ///   - vocabulary: The full tokenizer vocabulary
    ///   - vocabType: The vocabulary encoding type
    ///   - stopTokenIds: Token IDs to treat as stop tokens (e.g., EOS)
    /// - Throws: If the file can't be read or the schema is invalid
    public init(
        schemaPath: String,
        vocabulary: [String],
        vocabType: VocabularyType = .byteLevel,
        stopTokenIds: [Int32]? = nil,
        maxWhitespaceCount: Int? = ConstrainedGenerationSession.defaultMaxWhitespaceCount
    ) throws {
        let schema = try String(contentsOfFile: schemaPath, encoding: .utf8)
        try self.init(
            jsonSchema: schema, vocabulary: vocabulary, vocabType: vocabType,
            stopTokenIds: stopTokenIds, maxWhitespaceCount: maxWhitespaceCount)
    }
}

extension ConstrainedGenerationSession {
    /// Apply a bitmask from ``nextTokenBitmask()`` to Float logits in place
    /// (disallowed tokens → `-.infinity`): the mask can be worked out while the
    /// model computes the logits.
    ///
    /// A non-generic loop over raw buffers: the generic one this replaces spent
    /// most of a guided step looking up type metadata in an unoptimized build.
    public static func apply(_ bitmask: [Int32], to logits: inout [Float], vocabularySize: Int) {
        bitmask.withUnsafeBufferPointer { words in
            logits.withUnsafeMutableBufferPointer { values in
                blockDisallowed(words, in: values, vocabularySize: vocabularySize, blocked: -.infinity)
            }
        }
    }

    /// The tokens `bitmask` allows when there are at most `limit` of them, in
    /// increasing order; nil when there are more. A single allowed token is one
    /// the grammar forces, so no model step is needed to choose it.
    public static func allowedTokens(in bitmask: [Int32], vocabularySize: Int, limit: Int) -> [Int32]? {
        bitmask.withUnsafeBufferPointer { words -> [Int32]? in
            guard let mask = words.baseAddress else { return [] }
            var found: [Int32] = []
            var word = 0
            while word < words.count {
                let first = word &* 32
                if first >= vocabularySize { break }
                var allowed = UInt32(bitPattern: mask[word])
                let valid = vocabularySize &- first
                if valid < 32 { allowed &= (UInt32(1) << UInt32(valid)) &- 1 }
                while allowed != 0 {
                    if found.count == limit { return nil }
                    found.append(Int32(first &+ allowed.trailingZeroBitCount))
                    allowed &= allowed &- 1
                }
                word &+= 1
            }
            return found
        }
    }

    static func allowsNone(_ bitmask: [Int32]) -> Bool {
        bitmask.withUnsafeBufferPointer { words in
            guard let mask = words.baseAddress else { return true }
            var word = 0
            while word < words.count {
                if mask[word] != 0 { return false }
                word &+= 1
            }
            return true
        }
    }
}

private func blockDisallowed(
    _ words: UnsafeBufferPointer<Int32>, in logits: UnsafeMutableBufferPointer<Float>, vocabularySize: Int,
    blocked: Float
) {
    guard let mask = words.baseAddress, let values = logits.baseAddress else { return }
    let count = min(logits.count, vocabularySize)
    var word = 0
    while word < words.count {
        let first = word &* 32
        if first >= count { break }
        let allowed = UInt32(bitPattern: mask[word])
        if allowed != UInt32.max {
            let end = min(first &+ 32, count)
            if allowed == 0 {
                (values + first).update(repeating: blocked, count: end &- first)
            } else {
                var disallowed = ~allowed
                while disallowed != 0 {
                    let token = first &+ disallowed.trailingZeroBitCount
                    if token >= end { break }
                    values[token] = blocked
                    disallowed &= disallowed &- 1
                }
            }
        }
        word &+= 1
    }
}
#endif
