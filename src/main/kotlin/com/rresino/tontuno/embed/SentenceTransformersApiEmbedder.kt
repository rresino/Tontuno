package com.rresino.tontuno.embed

import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.plugins.logging.*
import io.ktor.client.request.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import mu.KotlinLogging

private val logger = KotlinLogging.logger {}

/**
 * Sentence Transformers embedder using HTTP API
 * This can work with Hugging Face Inference API or local servers
 */
class SentenceTransformersApiEmbedder(
    private val apiUrl: String = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
    private val apiKey: String? = null, // Set your HuggingFace API key here
    private val dimensions: Int = 384
) : Embedder {

    var fallbackEmbedder: Embedder? = null

    private val httpClient = HttpClient(CIO) {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
            })
        }
        install(Logging) {
            level = LogLevel.INFO
        }
    }

    @Serializable
    data class EmbeddingRequest(val inputs: String)

    override suspend fun embed(text: String): FloatArray {
        return try {
            withContext(Dispatchers.IO) {
                val response = httpClient.post(apiUrl) {
                    contentType(ContentType.Application.Json)

                    // Add API key if available
                    apiKey?.let { key ->
                        headers {
                            append("Authorization", "Bearer $key")
                        }
                    }

                    setBody(EmbeddingRequest(text))
                }

                when (response.status) {
                    HttpStatusCode.OK -> {
                        val embedding: List<List<Float>> = response.body()
                        // The API returns a list of embeddings, we take the first one
                        val result = embedding.firstOrNull()?.toFloatArray()
                            ?: throw IllegalStateException("Empty embedding response")

                        logger.debug { "Successfully generated embedding via API for text: ${text.take(50)}..." }
                        result
                    }
                    HttpStatusCode.ServiceUnavailable -> {
                        logger.warn { "Model is loading, using fallback embedder" }
                        fallbackEmbedder?.embed(text)
                            ?: throw IllegalStateException("API unavailable and no fallback configured")
                    }
                    else -> {
                        logger.error { "API request failed with status: ${response.status}" }
                        fallbackEmbedder?.embed(text)
                            ?: throw IllegalStateException("API request failed: ${response.status}")
                    }
                }
            }
        } catch (e: Exception) {
            logger.error(e) { "Error calling embedding API, falling back to simple embedder" }
            fallbackEmbedder?.embed(text)
                ?: throw IllegalStateException("API failed and no fallback available", e)
        }
    }

    override fun getDimensions(): Int = dimensions

    override suspend fun close() {
        httpClient.close()
        fallbackEmbedder?.close()
    }
}
