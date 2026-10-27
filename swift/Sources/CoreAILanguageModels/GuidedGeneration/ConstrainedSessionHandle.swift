#if canImport(CoreAI)
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Synchronization
import Tokenizers

/// Single-owner handle wrapping a `ConstrainedGenerationSession`.
///
/// Not thread-safe — ownership transfers between the engine cache and the
/// active generation task via the checkout/checkin pattern. Only one task
/// holds the handle at a time; concurrent access is a programming error.
package final class ConstrainedSessionHandle: @unchecked Sendable {
    /// Schema string is immutable after init — cached to avoid repeated access.
    let schema: String

    /// Whitespace limit the session enforces; nil when unlimited.
    let maxWhitespaceCount: Int?

    /// Tokenizer used for jump-forward string encoding.
    private let tokenizer: any Tokenizer

    private var _session: ConstrainedGenerationSession

    init(session: consuming ConstrainedGenerationSession, tokenizer: any Tokenizer) {
        self.schema = session.schema
        self.maxWhitespaceCount = session.maxWhitespaceCount
        self.tokenizer = tokenizer
        self._session = session
    }

    /// Whether this session was built for the same schema and whitespace limit, so a reset can reuse it.
    func matches(schema: String, maxWhitespaceCount: Int?) -> Bool {
        self.schema == schema && self.maxWhitespaceCount == maxWhitespaceCount
    }

    var isTerminated: Bool { _session.isTerminated }

    var vocabularySize: Int { _session.vocabularySize }

    /// The next position's bitmask, or nil once the grammar is done (see
    /// ``ConstrainedGenerationSession/nextTokenBitmask()``).
    func nextTokenBitmask() -> [Int32]? { _session.nextTokenBitmask() }

    func acceptToken(_ token: Int32) -> Bool {
        _session.acceptToken(token)
    }

    func fillBitmask(into ptr: UnsafeMutablePointer<Int32>) -> ConstrainedGenerationSession.BitmaskResult {
        _session.fillBitmask(into: ptr)
    }

    func reset() { _session.reset() }

    @discardableResult
    func rollback(_ numTokens: Int = 1) -> Bool { _session.rollback(numTokens) }

    func findJumpForwardString() -> String? {
        _session.findJumpForwardString()
    }

    /// Decode a sequence of token IDs back into a string using the stored tokenizer.
    func decodeTokens(_ tokens: [Int32]) -> String {
        tokenizer.decode(tokens: tokens.map(Int.init))
    }

    /// Tokenize a string for jump-forward injection. Uses the stored tokenizer
    /// without adding special tokens.
    func tokenizeForJumpForward(_ string: String) -> [Int32] {
        tokenizer.encode(text: string).map(Int32.init)
    }
}

/// Grammar sessions an engine keeps between requests, so a schema is compiled
/// once per loaded model rather than once per request. Building one rebuilds
/// the tokenizer's grammar view of the whole vocabulary and compiles the schema
/// against it, which took longer than reading the prompt.
package final class ConstrainedSessionCache: Sendable {
    /// Requests alternate between a few schemas (a reader's extraction and
    /// judgment), not dozens.
    static let capacity = 4

    private let entries = Mutex<[ConstrainedSessionHandle]>([])

    package init() {}

    /// A session for this schema and whitespace limit, reset to its start, or nil.
    package func take(schema: String, maxWhitespaceCount: Int?) -> ConstrainedSessionHandle? {
        let handle = entries.withLock { entries -> ConstrainedSessionHandle? in
            guard let index = entries.firstIndex(where: {
                $0.matches(schema: schema, maxWhitespaceCount: maxWhitespaceCount)
            }) else { return nil }
            return entries.remove(at: index)
        }
        handle?.reset()
        return handle
    }

    /// Keeps `handle` for the next request, dropping the least recently stored.
    package func store(_ handle: ConstrainedSessionHandle) {
        entries.withLock { entries in
            entries.removeAll { $0 === handle }
            entries.append(handle)
            if entries.count > Self.capacity { entries.removeFirst(entries.count - Self.capacity) }
        }
    }
}

/// An engine that keeps grammar sessions between requests.
package protocol ConstrainedSessionCaching: AnyObject {
    var constrainedSessions: ConstrainedSessionCache { get }
}

/// An engine that returns the logits after each of several new tokens from one
/// step, so a draft of the tokens the grammar forces is checked in one read.
package protocol LogitRowReading: AnyObject, Sendable {
    /// How many new tokens after `processed` one step reads on the rung a single
    /// new token is read with.
    func tokensPerStep(after processed: Int) -> Int

    /// Reads `input` from the prefix it shares with the engine's history and
    /// returns the logits after each of its last `rowCount` tokens, in order.
    func readLogitRows(_ input: [Int32], rowCount: Int) async throws -> [[LogitsScalarType]]
}
#endif
