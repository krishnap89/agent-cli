import UIKit

final class HomeViewController: UIViewController {
    private var userID = ""

    func configure(with id: String) {
        userID = id
    }

    func invalidate() {
        userID = ""
    }
}
