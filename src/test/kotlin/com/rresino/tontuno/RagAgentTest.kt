package com.rresino.tontuno

import kotlinx.coroutines.test.runTest
import kotlin.test.*

class RagAgentTest {

    @Test
    fun testDocumentAdditionAndRetrieval() = runTest {
        // Arrange
        val embedder = SimpleEmbedder()
        val vectorStore = InMemoryVectorStore()
        val ragAgent = RagAgent(embedder, vectorStore)

        val document = Document("test1", "Kotlin is a programming language")

        // Act
        ragAgent.addDocument(document)
        val response = ragAgent.query("What is Kotlin?")

        // Assert
        assertNotNull(response)
        assertTrue(response.contains("Kotlin"))
        assertTrue(response.contains("programming"))
    }

    @Test
    fun testMultipleDocumentsRetrieval() = runTest {
        // Arrange
        val embedder = SimpleEmbedder()
        val vectorStore = InMemoryVectorStore()
        val ragAgent = RagAgent(embedder, vectorStore)

        val documents = listOf(
            Document("1", "Kotlin is modern and concise"),
            Document("2", "Java is verbose but widely used"),
            Document("3", "Python is great for data science")
        )

        // Act
        ragAgent.addDocuments(documents)
        val response = ragAgent.query("programming languages")

        // Assert
        assertNotNull(response)
        assertTrue(response.isNotEmpty())
    }

    @Test
    fun testEmptyKnowledgeBase() = runTest {
        // Arrange
        val embedder = SimpleEmbedder()
        val vectorStore = InMemoryVectorStore()
        val ragAgent = RagAgent(embedder, vectorStore)

        // Act
        val response = ragAgent.query("What is Kotlin?")

        // Assert
        assertTrue(response.contains("don't have enough information"))
    }
}

