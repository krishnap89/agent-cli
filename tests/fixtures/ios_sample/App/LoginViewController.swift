import UIKit
import CoreKit

final class LoginViewController: UIViewController {
    @IBOutlet private weak var loginButton: UIButton!
    var repository: UserRepositoryProtocol = Container.shared.repository

    override func viewDidLoad() {
        super.viewDidLoad()
        loginButton.setImage(UIImage(named: "logo"), for: .normal)
        title = NSLocalizedString("login_title", comment: "")
        view.backgroundColor = UIColor(named: "AccentColor")
    }

    @IBAction func loginTapped(_ sender: Any) {
        performLogin()
    }

    private func performLogin() {
        let home = storyboard?.instantiateViewController(withIdentifier: "Home") as? HomeViewController
        home?.configure(with: "me")
        navigationController?.pushViewController(home!, animated: true)
    }
}
