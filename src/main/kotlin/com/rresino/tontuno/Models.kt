package com.rresino.tontuno

import kotlinx.serialization.Serializable

/**
 * Represents a document in the knowledge base
 */
@Serializable
data class Document(
    val id: String,
    val content: String,
    val metadata: Map<String, String> = emptyMap()
)

/**
 * Represents an embedded document with its vector representation
 */
data class EmbeddedDocument(
    val document: Document,
    val embedding: FloatArray,
    val timestamp: Long = System.currentTimeMillis()
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as EmbeddedDocument

        if (document != other.document) return false
        if (!embedding.contentEquals(other.embedding)) return false
        if (timestamp != other.timestamp) return false

        return true
    }

    override fun hashCode(): Int {
        var result = document.hashCode()
        result = 31 * result + embedding.contentHashCode()
        result = 31 * result + timestamp.hashCode()
        return result
    }
}

/**
 * Represents a search result with similarity score
 */
data class SearchResult(
    val document: Document,
    val similarity: Float,
    val embedding: FloatArray
)
