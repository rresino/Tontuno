package com.rresino.tontuno

import kotlinx.coroutines.runBlocking
import mu.KotlinLogging

private val logger = KotlinLogging.logger {}

fun main() = runBlocking {
    logger.info { "Starting RAG Learning Agent..." }

    // Initialize components with different embedder options
    println("=== RAG Learning Agent Demo ===\n")
    println("Choose embedder type:")
    println("1. Simple Embedder (Fast, basic)")
    println("2. Sentence Transformers via API (Requires API)")
    println("3. ONNX Sentence Transformers (Local model)")

    // Initialize components
    // Opción 1: Factory con auto-detección
    val embedder = EmbedderFactory.createFromEnvironment()
    // Opción 2: Directo por tipo
    // val embedder = OllamaEmbedder()
    // Opción 3: Factory con configuración
    // val embedder = EmbedderFactory.createEmbedder(
    //    EmbedderFactory.EmbedderType.OLLAMA,
    //    mapOf("modelName" to "nomic-embed-text")
    // )
    val vectorStore = InMemoryVectorStore()
    val ragAgent = RagAgent(embedder, vectorStore)

    // Example usage
    println("=== RAG Learning Agent Demo ===\n")

    // Add some learning material
    val documents = listOf(
        Document("1", "Kotlin is a modern programming language that runs on the JVM and is fully interoperable with Java."),
        Document("2", "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation."),
        Document("3", "Vector databases store high-dimensional vectors and enable similarity search."),
        Document("4", "Machine learning embeddings convert text into numerical vectors that capture semantic meaning."),
        Document("5", "Coroutines in Kotlin provide a way to write asynchronous code in a sequential manner.")
    )

    // Store documents in the RAG system
    println("📚 Adding documents to knowledge base...")
    documents.forEach { doc ->
        ragAgent.addDocument(doc)
        println("  ✓ Added: ${doc.content.take(50)}...")
    }

    println("\n🔍 Querying the knowledge base...\n")

    // Query examples
    val queries = listOf(
        "What is Kotlin?",
        "How does RAG work?",
        "Tell me about vector databases",
        "What are embeddings?",
        "Explain coroutines"
    )

    queries.forEach { query ->
        println("Q: $query")
        val response = ragAgent.query(query, topK = 2)
        println("A: $response\n")
        println("---")
    }

    // Show statistics
    println(ragAgent.getStats())
}
