package com.example.app

import android.os.Bundle
import android.widget.Button
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.example.app.databinding.ActivityMainBinding
import com.example.data.User
import com.example.data.displayName
import com.example.data.isValidId
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : AppCompatActivity() {
    private val viewModel: MainViewModel by viewModels()
    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(R.layout.activity_main)
        val button = findViewById<Button>(R.id.refresh_button)
        button.setOnClickListener { viewModel.refresh(currentId()) }
        viewModel.users.observe(this) { users -> render(users) }
    }

    private fun currentId(): String {
        val raw = intent.getStringExtra("id") ?: ""
        return if (raw.isValidId()) raw else DEFAULT_ID
    }

    private fun render(users: List<User>) {
        val title = """
            Users: ${users.size}
            first: ${users.firstOrNull()?.displayName()}
        """.trimIndent()
        log("render } { $title")
    }

    companion object {
        const val DEFAULT_ID = "me"

        fun log(msg: String) = println(msg)
    }
}
