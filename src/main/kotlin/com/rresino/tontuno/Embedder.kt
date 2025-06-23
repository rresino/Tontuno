package com.rresino.tontuno

import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Interface for text embedding functionality
 */
interface Embedder {
    suspend fun embed(text: String): FloatArray
    fun getDimensions(): Int
}

/**
 * Simple embedder that creates deterministic embeddings based on text content
 * In a real implementation, you would use a proper embedding model
 */
class SimpleEmbedder(private val dimensions: Int = 384) : Embedder {

    override suspend fun embed(text: String): FloatArray {
        // Create a simple but deterministic embedding based on text characteristics
        val embedding = FloatArray(dimensions)

        // Use text hash as seed for reproducible results
        val seed = text.hashCode().toLong()
        val random = Random(seed)

        // Generate base embedding
        repeat(dimensions) { i ->
            embedding[i] = random.nextFloat() * 2f - 1f // Range [-1, 1]
        }

        // Add some semantic-like features based on text content
        val words = text.lowercase().split(Regex("\\W+")).filter { it.isNotEmpty() }
        val wordFeatures = extractWordFeatures(words)

        // Blend random embedding with word features
        wordFeatures.forEachIndexed { index, feature ->
            if (index < dimensions) {
                embedding[index] = (embedding[index] * 0.7f + feature * 0.3f)
            }
        }

        // Normalize the embedding
        return normalizeVector(embedding)
    }

    override fun getDimensions(): Int = dimensions

    /**
     * Extract simple features from words to make embeddings more semantic
     */
    private fun extractWordFeatures(words: List<String>): FloatArray {
        val features = FloatArray(dimensions)

        // Simple heuristics to create semantic-like features
        words.forEachIndexed { wordIndex, word ->
            val wordHash = word.hashCode()
            val featureIndex = kotlin.math.abs(wordHash) % dimensions
            features[featureIndex] += 1f / (wordIndex + 1f) // Decay by position
        }

        return features
    }

    /**
     * Normalize vector to unit length
     */
    private fun normalizeVector(vector: FloatArray): FloatArray {
        val norm = sqrt(vector.map { it * it }.sum())
        return if (norm > 0f) {
            vector.map { it / norm }.toFloatArray()
        } else {
            vector
        }
    }
}
