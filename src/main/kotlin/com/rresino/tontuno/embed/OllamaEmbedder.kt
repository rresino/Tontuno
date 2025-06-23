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
 * Ollama embedder using local Ollama API
 * Requires Ollama to be running locally with an embedding model
 */
class OllamaEmbedder(
    private val baseUrl: String = "http://localhost:11434",
    private val modelName: String = "nomic-embed-text",
    private val dimensions: Int = 768
) : Embedder {

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
    data class OllamaEmbedRequest(
        val model: String,
        val prompt: String
    )

    @Serializable
    data class OllamaEmbedResponse(
        val embedding: List<Double>
    )

    override suspend fun embed(text: String): FloatArray {
        return withContext(Dispatchers.IO) {
            try {
                val response = httpClient.post("$baseUrl/api/embeddings") {
                    contentType(ContentType.Application.Json)
                    setBody(OllamaEmbedRequest(
                        model = modelName,
                        prompt = text
                    ))
                }

                if (response.status == HttpStatusCode.OK) {
                    val embedResponse: OllamaEmbedResponse = response.body()
                    val embedding = embedResponse.embedding.map { it.toFloat() }.toFloatArray()

                    logger.debug { "Generated Ollama embedding for text: ${text.take(50)}..." }
                    embedding
                } else {
                    throw IllegalStateException("Ollama API request failed: ${response.status}")
                }
            } catch (e: Exception) {
                logger.error(e) { "Error calling Ollama embedding API" }
                throw e
            }
        }
    }

    override fun getDimensions(): Int = dimensions

    override suspend fun close() {
        httpClient.close()
    }

    /**
     * Check if Ollama is running and the model is available
     */
    suspend fun isAvailable(): Boolean {
        return try {
            val response = httpClient.get("$baseUrl/api/tags")
            response.status == HttpStatusCode.OK
        } catch (e: Exception) {
            false
        }
    }

    /**
     * List available models in Ollama
     */
    suspend fun listModels(): List<String> {
        return try {
            @Serializable
            data class OllamaModel(val name: String)
            @Serializable
            data class OllamaModelsResponse(val models: List<OllamaModel>)

            val response = httpClient.get("$baseUrl/api/tags")
            if (response.status == HttpStatusCode.OK) {
                val modelsResponse: OllamaModelsResponse = response.body()
                modelsResponse.models.map { it.name }
            } else {
                emptyList()
            }
        } catch (e: Exception) {
            logger.error(e) { "Error listing Ollama models" }
            emptyList()
        }
    }
}
