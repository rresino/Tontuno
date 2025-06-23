package com.rresino.tontuno

/**
 * Interface for generating responses based on retrieved documents
 */
interface ResponseGenerator {
    fun generateResponse(query: String, searchResults: List<SearchResult>): String
}

/**
 * Simple response generator that combines information from retrieved documents
 */
class SimpleResponseGenerator : ResponseGenerator {

    override fun generateResponse(query: String, searchResults: List<SearchResult>): String {
        if (searchResults.isEmpty()) {
            return "I don't have relevant information to answer your question."
        }

        // Get the most relevant documents
        val topResults = searchResults.take(3)

        // Create response based on similarity scores and content
        val responseBuilder = StringBuilder()

        // Use the most similar document as primary source
        val primaryResult = topResults.first()
        responseBuilder.append("Based on my knowledge: ")
        responseBuilder.append(primaryResult.document.content)

        // Add supporting information if available
        if (topResults.size > 1) {
            val supportingInfo = topResults.drop(1)
                .filter { it.similarity > 0.3f } // Only include reasonably similar results
                .joinToString(" ") { it.document.content }

            if (supportingInfo.isNotEmpty()) {
                responseBuilder.append(" Additionally, $supportingInfo")
            }
        }

        // Add confidence indicator based on similarity scores
        val avgSimilarity = topResults.map { it.similarity }.average()
        val confidence = when {
            avgSimilarity > 0.8 -> " (High confidence)"
            avgSimilarity > 0.5 -> " (Medium confidence)"
            else -> " (Low confidence - information may not be directly relevant)"
        }

        responseBuilder.append(confidence)

        return responseBuilder.toString()
    }
}
