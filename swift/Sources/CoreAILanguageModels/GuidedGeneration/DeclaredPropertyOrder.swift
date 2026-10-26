#if canImport(CoreAI)
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Writes a JSON schema with every object's `properties` in their declared
/// order.
///
/// `GenerationSchema` encodes `properties` from a dictionary, so their order in
/// the JSON is arbitrary and changes from one encoding to the next, while
/// xgrammar generates an object's fields in the order its `properties` are
/// written. A model then writes its fields in an order it was never trained on,
/// differently on every request. The declared order travels beside them as
/// `x-order`; every object that has one gets its `properties` in that order,
/// any not named in it after, by name. Other keys are written by name, so the
/// same schema always gives the same text.
enum DeclaredPropertyOrder {
    /// The schema rewritten, or nil when `data` is not JSON.
    static func json(_ data: Data) -> String? {
        guard let root = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) else {
            return nil
        }
        var out = ""
        return write(root, into: &out) ? out : nil
    }

    private static func write(_ value: Any, into out: inout String) -> Bool {
        if let object = value as? [String: Any] {
            return write(object, into: &out)
        }
        if let array = value as? [Any] {
            out += "["
            for (index, element) in array.enumerated() {
                if index > 0 { out += "," }
                guard write(element, into: &out) else { return false }
            }
            out += "]"
            return true
        }
        return leaf(value, into: &out)
    }

    private static func write(_ object: [String: Any], into out: inout String) -> Bool {
        out += "{"
        for (index, key) in object.keys.sorted().enumerated() {
            if index > 0 { out += "," }
            guard leaf(key, into: &out), let member = object[key] else { return false }
            out += ":"
            if key == "properties", let properties = member as? [String: Any] {
                let declared = (object["x-order"] as? [String] ?? []).filter { properties[$0] != nil }
                let names = declared + properties.keys.sorted().filter { !declared.contains($0) }
                out += "{"
                for (position, name) in names.enumerated() {
                    if position > 0 { out += "," }
                    guard leaf(name, into: &out), let schema = properties[name] else { return false }
                    out += ":"
                    guard write(schema, into: &out) else { return false }
                }
                out += "}"
            } else if !write(member, into: &out) {
                return false
            }
        }
        out += "}"
        return true
    }

    /// A string, number, boolean or null, as JSONSerialization writes it.
    private static func leaf(_ value: Any, into out: inout String) -> Bool {
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed]),
            let text = String(data: data, encoding: .utf8)
        else { return false }
        out += text
        return true
    }
}
#endif
