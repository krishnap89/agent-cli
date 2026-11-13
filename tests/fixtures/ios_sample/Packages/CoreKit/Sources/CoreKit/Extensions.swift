import Foundation

public extension String {
    var isValidID: Bool { count > 3 && allSatisfy { $0.isLetter || $0.isNumber } }

    func trimmed() -> String { trimmingCharacters(in: .whitespaces) }
}

extension User: CustomStringConvertible {
    public var description: String { "User(\(displayName()))" }

    public func displayName() -> String { name.trimmed() }
}
