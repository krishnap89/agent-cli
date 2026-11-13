import SwiftUI
import CoreKit

struct UserListView: View {
    @EnvironmentObject var model: UserListViewModel
    @State private var query = ""

    var body: some View {
        NavigationStack {
            List(model.users) { user in
                Text(user.displayName())
                    .onTapGesture { model.select(user) }
            }
            .onAppear { model.refresh(id: currentID()) }
            .navigationTitle(Text("users_title"))
        }
    }

    private func currentID() -> String {
        query.isValidID ? query : Self.defaultID
    }

    static let defaultID = "me"
}

#Preview {
    UserListView().environmentObject(Container.shared.makeViewModel())
}
