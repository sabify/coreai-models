#if canImport(CoreAI)
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Synchronization

/// Ends the generations running on a model at their next token, without cancelling
/// the tasks that run them.
///
/// A caller whose request a newer one replaced frees the engine at once, and Foundation
/// Models, which can crash when cancellation interrupts a generation, sees an ordinary
/// error: the generation throws ``GenerationStopped``. A generation that starts after
/// the stop runs on.
public final class GenerationStop: Sendable {
    fileprivate let epoch = Atomic<Int>(0)

    public init() {}

    /// Ends every generation running now at its next token.
    public func stopRunning() {
        _ = epoch.wrappingAdd(1, ordering: .relaxed)
    }

    /// What a generation starting now checks before each token.
    package func ticket() -> Ticket {
        Ticket(stop: self, epoch: epoch.load(ordering: .relaxed))
    }

    package struct Ticket: Sendable {
        fileprivate let stop: GenerationStop
        fileprivate let epoch: Int

        /// Whether ``GenerationStop/stopRunning()`` was called since the ticket was taken.
        package var isStopped: Bool { stop.epoch.load(ordering: .relaxed) != epoch }
    }
}

/// The generation ended because its owner asked (``GenerationStop/stopRunning()``).
public struct GenerationStopped: Error, Sendable, Equatable {
    public init() {}
}
#endif
