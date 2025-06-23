package com.rresino.tontuno

import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class VectorStoreTest {

    @Test
    fun testVectorStorage() = runTest {
        // Arrange
        val vectorStore = InMemoryVectorStore()
        val embedding = floatArrayOf(0.1f, 0.2f, 0.3f)
        val document = Document("test", "Test content")
        val embeddedDoc = EmbeddedDocument(document, embedding)

        // Act
        vectorStore.store(embeddedDoc)
        val count = vectorStore.getDocumentCount()

        // Assert
        assertEquals(1, count)
    }

    @Test
    fun testSimilaritySearch() = runTest {
        // Arrange
        val vectorStore = InMemoryVectorStore()
        val embedding1 = floatArrayOf(1.0f, 0.0f, 0.0f)
        val embedding2 = floatArrayOf(0.0f, 1.0f, 0.0f)
        val queryEmbedding = floatArrayOf(0.9f, 0.1f, 0.0f) // More similar to embedding1

        val doc1 = EmbeddedDocument(Document("1", "Content 1"), embedding1)
        val doc2 = EmbeddedDocument(Document("2", "Content 2"), embedding2)

        // Act
        vectorStore.store(doc1)
        vectorStore.store(doc2)
        val results = vectorStore.search(queryEmbedding, topK = 2)

        // Assert
        assertEquals(2, results.size)
        assertEquals("1", results.first().document.id) // Most similar should be first
        assertTrue(results.first().similarity > results.last().similarity)
    }
}
