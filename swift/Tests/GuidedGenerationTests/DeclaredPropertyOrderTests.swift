// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("DeclaredPropertyOrder")
struct DeclaredPropertyOrderTests {
    @Test func propertiesFollowXOrder() throws {
        let schema = #"{"type":"object","x-order":["outcome","people","direction"],"properties":{"direction":{"type":"string"},"outcome":{"type":"string"},"people":{"type":"array","items":{"type":"string"}}},"required":["outcome","people","direction"]}"#
        let written = try #require(DeclaredPropertyOrder.json(Data(schema.utf8)))
        let outcome = try #require(written.range(of: #""outcome":"#))
        let people = try #require(written.range(of: #""people":"#))
        let direction = try #require(written.range(of: #""direction":"#))
        #expect(outcome.lowerBound < people.lowerBound && people.lowerBound < direction.lowerBound)
        #expect(try JSONSerialization.jsonObject(with: Data(written.utf8)) as? NSDictionary
            == JSONSerialization.jsonObject(with: Data(schema.utf8)) as? NSDictionary)
    }

    @Test func nestedObjectsAndUndeclaredPropertiesKeepAStableOrder() throws {
        let schema = #"{"properties":{"b":{"x-order":["z","y"],"properties":{"y":{},"z":{}}},"a":{}},"x-order":["b"]}"#
        let written = try #require(DeclaredPropertyOrder.json(Data(schema.utf8)))
        #expect(written == #"{"properties":{"b":{"properties":{"z":{},"y":{}},"x-order":["z","y"]},"a":{}},"x-order":["b"]}"#)
        #expect(DeclaredPropertyOrder.json(Data(written.utf8)) == written)
    }

    @Test func notJSONIsRefused() {
        #expect(DeclaredPropertyOrder.json(Data("not json {".utf8)) == nil)
    }
}
