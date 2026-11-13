package com.example.data

interface UserApi {
    suspend fun fetch(id: String): User
    fun fetchAll(): List<User>
}
