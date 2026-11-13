package com.example.app

import com.example.data.UserRepository as Repo

/* header /* nested */ comment with fun fake() { fakeCall() } */
class Outer {
    val template = "value: ${compute("}")} and \"${'$'}{notCode()}\""
    private val raw = """
        fun ghost() { haunt() }
        ${'$'}{alsoNotCode()}
        real: ${compute("raw")}
    """"

    fun compute(s: String): Int = s.length

    class Nested {
        fun ping() = Outer().compute("x")

        object Deep {
            fun pong() = Nested().ping()
        }
    }

    fun lambdas(repo: Repo) {
        val f: (Int) -> Int = { x -> x + 1 }
        f(2)
        listOf(1, 2).map { it * 2 }.filter { n -> n > 1 }.forEach(::println)
        run {
            val inner = Nested()
            inner.ping()
        }
        repo.clear()
    }
}

fun Outer.Nested.extra(): Int = ping() + 1

fun Int.squared() = this * this

fun useExtensions() = 3.squared() + Outer.Nested().extra()

// Receiver type unknown (List<Any>.first() is external): two project
// classes have a zero-arg invalidate(), so this edge is ambiguous.
fun invalidateFirst(things: List<Any>) = things.first().invalidate()

// Same situation with a name every collection has: not guessed at all.
fun clearFirst(things: List<Any>) = things.first().clear()
