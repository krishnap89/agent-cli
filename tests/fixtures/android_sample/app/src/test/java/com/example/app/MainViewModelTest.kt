package com.example.app

import com.example.data.UserRepository
import io.mockk.mockk
import org.junit.Test

class MainViewModelTest {
    private val repo = mockk<UserRepository>()

    @Test
    fun refreshLoads() {
        val vm = MainViewModel(repo)
        vm.refresh("1")
    }
}
