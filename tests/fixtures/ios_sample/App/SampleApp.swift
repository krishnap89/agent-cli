import SwiftUI
import CoreKit

@main
struct SampleApp: App {
    @StateObject private var model = Container.shared.makeViewModel()

    var body: some Scene {
        WindowGroup {
            UserListView()
                .environmentObject(model)
        }
    }
}
