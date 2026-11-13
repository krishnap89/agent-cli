package com.example.app

import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.data.User
import com.example.data.UserRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.launch

@HiltViewModel
class MainViewModel @Inject constructor(
    private val repository: UserRepository,
) : ViewModel() {
    val users = MutableLiveData<List<User>>()

    fun refresh(id: String) {
        viewModelScope.launch {
            val user = repository.load(id)
            users.postValue(listOf(user))
        }
    }

    override fun onCleared() {
        repository.clear()
        super.onCleared()
    }

    inner class Loader {
        fun start() = refresh("all")

        inner class Step {
            fun run() {
                start()
            }
        }
    }
}
