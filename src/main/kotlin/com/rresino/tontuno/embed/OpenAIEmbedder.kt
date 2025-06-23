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
 * OpenAI embeddings using their API
 */
class OpenAIEmbedder(
    private val apiKey: String,
    private val model: String = "text-embedding-3-small",
    private val dimensions: Int = 1536
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
    data class OpenAIEmbeddingRequest(
        val input: String,
        val model: String,
        val dimensions: Int? = null
    )

    @Serializable
    data class OpenAIEmbeddingResponse(
        val data: List<EmbeddingData>,
        val model: String,
        val usage: Usage
    )

    @Serializable
    data class EmbeddingData(
        val embedding: List<Float>,
        val index: Int
    )

    @Serializable
    data class Usage(
        val prompt_tokens: Int,
        val total_tokens: Int
    )

    override suspend fun embed(text: String): FloatArray {
        return withContext(Dispatchers.IO) {
            try {
                val response = httpClient.post("https://api.openai.com/v1/embeddings") {
                    contentType(ContentType.Application.Json)
                    headers {
                        append("Authorization", "Bearer $apiKey")
                    }

                    setBody(OpenAIEmbeddingRequest(
                        input = text,
                        model = model,
                        dimensions = if (model.contains("text-embedding-3")) dimensions else null
                    ))
                }

                if (response.status == HttpStatusCode.OK) {
                    val embeddingResponse: OpenAIEmbeddingResponse = response.body()
                    val embedding = embeddingResponse.data.firstOrNull()?.embedding
                        ?: throw IllegalStateException("No embedding in response")

                    logger.debug { "Generated OpenAI embedding for text: ${text.take(50)}..." }
                    embedding.toFloatArray()
                } else {
                    throw IllegalStateException("OpenAI API request failed: ${response.status}")
                }
            } catch (e: Exception) {
                logger.error(e) { "Error calling OpenAI embedding API" }
                throw e
            }
        }
    }

    override fun getDimensions(): Int = dimensions

    override suspend fun close() {
        httpClient.close()
    }
}
