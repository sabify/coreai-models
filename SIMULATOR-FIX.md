# Building for the iOS Simulator

This fork carries one change against `apple/coreai-models`: every source file
in `CoreAILanguageModels` and `CoreAIShared` is wrapped in
`#if canImport(CoreAI)`.

## Why

`CoreAI.framework` ships in the iPhoneOS and macOS SDKs and is **absent from
the iPhoneSimulator SDK** — no framework, no stub, no `.tbd`, nothing under
`System/Library/Frameworks`, `PrivateFrameworks` or `SubFrameworks`. Verified
on Xcode 27.0 (27A266a), all three SDKs at 27.0.

So a bare `import CoreAI` makes every simulator build fail:

```
error: Unable to resolve module dependency: 'CoreAI'
  (in target 'CoreAIShared' from project 'coreai-models')
```

This happens at *compile* time, inside the package, before any consumer's own
`#if` is consulted. No Xcode setting avoids it: SwiftPM has no device-versus-
simulator axis (`.when(platforms:)` treats `.iOS` as both), `platformFilters`
enumerate platforms rather than SDKs, and linker flags cannot answer a
"no such module" error.

The consequence for an app is severe: it cannot link `CoreAILM` at all without
losing the simulator, which is where its whole test suite runs.

Upstream: <https://github.com/apple/coreai-models/issues/49>, filed
2026-06-16, Apple Feedback FB23189921, still open.

## What the change does

Guarding only the files that `import CoreAI` is not enough — the files that do
not import it still reference types declared in the ones that do, so the
module's surface collapses and the build fails with ~128 "cannot find type"
errors instead. Every file in both modules is therefore compiled away
together.

On the simulator the modules exist and are empty. On device and macOS nothing
changes: the files compile exactly as upstream.

## What a consumer must do

Detect the framework, not the module. `canImport(CoreAILanguageModels)` is
**true** on the simulator with this fork — the module is present, just empty —
so a consumer that guards on it will still fail to find `CoreAILanguageModel`.

```swift
#if canImport(CoreAI)
import CoreAILanguageModels
// … the Core AI path …
#endif
```

## Verified

| target | result |
| --- | --- |
| `-sdk iphonesimulator` | BUILD SUCCEEDED |
| `-sdk iphoneos` | BUILD SUCCEEDED |
| `platform=macOS` | BUILD SUCCEEDED |

Forked from upstream `eb6cead`; re-applied on upstream `09ff139` (2026-10-23),
where the unguarded package still fails the simulator build with
`Unable to resolve module dependency: 'CoreAI'`.

## Keeping it

Re-apply on every upstream bump until #49 lands. The change is mechanical —
two lines per file — so `git rebase` usually carries it, and
`swift/Sources/{CoreAILanguageModels,CoreAIShared}/**/*.swift` is the complete
set.
