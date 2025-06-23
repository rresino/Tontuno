package com.rresino.tontuno

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import mu.KotlinLogging
import kotlin.math.sqrt

private val logger = KotlinLogging.logger {}

/**
 * Interface for vector storage and similarity search
 */
interface VectorStore {
    suspend fun store(embeddedDocument: EmbeddedDocument)
    suspend fun search(queryEmbedding: FloatArray, topK: Int = 5): List<SearchResult>
    suspend fun getDocumentCount(): Int
    suspend fun clear()
}

/**
 * In-memory vector store implementation with cosine similarity search
 */
class InMemoryVectorStore : VectorStore {
    private val documents = mutableListOf<EmbeddedDocument>()
    private val mutex = Mutex()

    override suspend fun store(embeddedDocument: EmbeddedDocument) = mutex.withLock {
        // Remove existing document with same ID if present
        documents.removeIf { it.document.id == embeddedDocument.document.id }
        documents.add(embeddedDocument)
        logger.debug { "Stored document: ${embeddedDocument.document.id}" }
    }

    override suspend fun search(queryEmbedding: FloatArray, topK: Int): List<SearchResult> = mutex.withLock {
        if (documents.isEmpty()) {
            logger.warn { "No documents in vector store for search" }
            return emptyList()
        }

        // Calculate cosine similarity for all documents
        val similarities = documents.map { embeddedDoc ->
            val similarity = cosineSimilarity(queryEmbedding, embeddedDoc.embedding)
            SearchResult(embeddedDoc.document, similarity, embeddedDoc.embedding)
        }

        // Return top K results sorted by similarity (descending)
        return similarities
            .sortedByDescending { it.similarity }
            .take(topK)
            .also { results ->
                logger.debug { "Search returned ${results.size} results, top similarity: ${results.firstOrNull()?.similarity}" }
            }
    }

    override suspend fun getDocumentCount(): Int = mutex.withLock {
        documents.size
    }

    override suspend fun clear() = mutex.withLock {
        documents.clear()
        logger.info { "Vector store cleared" }
    }

    /**
     * Calculate cosine similarity between two vectors
     */
    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) {
            throw IllegalArgumentException("Vector dimensions must match: ${a.size} vs ${b.size}")
        }

        var dotProduct = 0f
        var normA = 0f
        var normB = 0f

        for (i in a.indices) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        normA = sqrt(normA)
        normB = sqrt(normB)

        return if (normA > 0f && normB > 0f) {
            dotProduct / (normA * normB)
        } else {
            0f
        }
    }
}
