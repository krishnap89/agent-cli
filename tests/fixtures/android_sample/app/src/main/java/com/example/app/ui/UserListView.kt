package com.example.app.ui

import android.content.Context
import android.util.AttributeSet
import android.view.View
import com.example.data.User

class UserListView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    private val adapter = UserAdapter()

    fun bind(users: List<User>) {
        adapter.submit(users)
    }
}

class UserAdapter {
    fun submit(items: List<User>) {
        items.forEach { notifyItem(it) }
    }

    fun notifyItem(user: User) {}

    fun invalidate() {}
}
