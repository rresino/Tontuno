package com.rresino.tontuno

import com.rresino.tontuno.embed.*
import mu.KotlinLogging

private val logger = KotlinLogging.logger {}

/**
 * Main RAG (Retrieval-Augmented Generation) agent that combines
 * document retrieval with response generation
 */
class RagAgent(
    private val embedder: Embedder,
    private val vectorStore: VectorStore,
    private val responseGenerator: ResponseGenerator = SimpleResponseGenerator()
) {

    /**
     * Add a document to the knowledge base
     */
    suspend fun addDocument(document: Document) {
        logger.info { "Adding document to knowledge base: ${document.id}" }

        // Generate embedding for the document
        val embedding = embedder.embed(document.content)

        // Store in vector database
        val embeddedDocument = EmbeddedDocument(document, embedding)
        vectorStore.store(embeddedDocument)

        logger.debug { "Document added successfully: ${document.id}" }
    }

    /**
     * Add multiple documents at once
     */
    suspend fun addDocuments(documents: List<Document>) {
        logger.info { "Adding ${documents.size} documents to knowledge base" }
        documents.forEach { addDocument(it) }
    }

    /**
     * Query the knowledge base and generate a response
     */
    suspend fun query(query: String, topK: Int = 3): String {
        logger.info { "Processing query: $query" }

        // Generate embedding for the query
        val queryEmbedding = embedder.embed(query)

        // Search for relevant documents
        val searchResults = vectorStore.search(queryEmbedding, topK)

        if (searchResults.isEmpty()) {
            logger.warn { "No relevant documents found for query: $query" }
            return "I don't have enough information to answer that question."
        }

        logger.debug { "Found ${searchResults.size} relevant documents" }

        // Generate response using retrieved documents
        return responseGenerator.generateResponse(query, searchResults)
    }

    /**
     * Get statistics about the knowledge base
     */
    suspend fun getStats(): String {
        val docCount = vectorStore.getDocumentCount()
        val dimensions = embedder.getDimensions()
        val embedderType = embedder::class.simpleName

        return """
            📊 RAG Agent Statistics:
            - Documents in knowledge base: $docCount
            - Embedding dimensions: $dimensions
            - Embedder type: $embedderType
        """.trimIndent()
    }

    /**
     * Clear the entire knowledge base
     */
    suspend fun clearKnowledgeBase() {
        logger.info { "Clearing knowledge base" }
        vectorStore.clear()
    }

    /**
     * Close resources (important for API-based embedders)
     */
    suspend fun close() {
        embedder.close()
    }
}
