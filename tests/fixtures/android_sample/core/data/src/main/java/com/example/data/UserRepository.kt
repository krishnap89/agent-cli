package com.example.data

import javax.inject.Inject

interface Repository {
    fun clear()
}

class UserRepository @Inject constructor(
    private val api: UserApi,
    private val cache: UserCache,
) : Repository {
    suspend fun load(id: String): User {
        val cached = cache.get(id)
        return cached ?: api.fetch(id).also { cache.put(it) }
    }

    override fun clear() {
        cache.clear()
    }

    fun all(): List<User> = api.fetchAll()
}

class UserCache {
    private val map = HashMap<String, User>()

    fun get(id: String): User? = map[id]

    fun put(user: User) {
        map[user.id] = user
    }

    fun clear() = map.clear()

    fun invalidate() = clear()
}
