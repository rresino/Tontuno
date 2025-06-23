package com.rresino.tontuno.embed

/**
 * Interface for text embedding functionality
 */
interface Embedder {
    suspend fun embed(text: String): FloatArray
    fun getDimensions(): Int
    suspend fun close() {} // For cleanup of resources
}

