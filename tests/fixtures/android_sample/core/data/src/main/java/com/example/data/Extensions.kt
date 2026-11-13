package com.example.data

fun String.isValidId(): Boolean = length > 3 && all { it.isLetterOrDigit() }

fun User.displayName(): String = name.trim().ifEmpty { id }
