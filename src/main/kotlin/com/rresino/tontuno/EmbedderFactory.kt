package com.rresino.tontuno

import com.rresino.tontuno.embed.*
import mu.KotlinLogging

private val logger = KotlinLogging.logger {}

/**
 * Factory class to create embedders based on configuration
 */
object EmbedderFactory {

    enum class EmbedderType {
        SIMPLE,
        SENTENCE_TRANSFORMERS_API,
        ONNX_SENTENCE_TRANSFORMERS,
        OPENAI,
        OLLAMA
    }

    /**
     * Create an embedder based on type and configuration
     */
    fun createEmbedder(
        type: EmbedderType,
        config: Map<String, String> = emptyMap()
    ): Embedder {
        return when (type) {
            EmbedderType.SIMPLE -> {
                val dimensions = config["dimensions"]?.toIntOrNull() ?: 384
                SimpleEmbedder(dimensions)
            }

            EmbedderType.SENTENCE_TRANSFORMERS_API -> {
                val apiUrl = config["apiUrl"]
                    ?: "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
                val apiKey = config["apiKey"]
                val dimensions = config["dimensions"]?.toIntOrNull() ?: 384

                SentenceTransformersApiEmbedder(apiUrl, apiKey, dimensions).apply {
                    fallbackEmbedder = SimpleEmbedder(dimensions)
                    logger.info { "Created Sentence Transformers API embedder with fallback" }
                }
            }

            EmbedderType.ONNX_SENTENCE_TRANSFORMERS -> {
                val modelPath = config["modelPath"]
                    ?: throw IllegalArgumentException("modelPath required for ONNX embedder")
                val tokenizerPath = config["tokenizerPath"]
                    ?: throw IllegalArgumentException("tokenizerPath required for ONNX embedder")
                val dimensions = config["dimensions"]?.toIntOrNull() ?: 384

                OnnxSentenceTransformersEmbedder(modelPath, tokenizerPath, dimensions)
            }

            EmbedderType.OPENAI -> {
                val apiKey = config["apiKey"]
                    ?: throw IllegalArgumentException("apiKey required for OpenAI embedder")
                val model = config["model"] ?: "text-embedding-3-small"
                val dimensions = config["dimensions"]?.toIntOrNull() ?: 1536

                OpenAIEmbedder(apiKey, model, dimensions)
            }

            EmbedderType.OLLAMA -> {
                val baseUrl = config["baseUrl"] ?: "http://localhost:11434"
                val modelName = config["modelName"] ?: "nomic-embed-text"
                val dimensions = config["dimensions"]?.toIntOrNull() ?: 768

                OllamaEmbedder(baseUrl, modelName, dimensions)
            }
        }
    }

    /**
     * Create embedder from environment variables
     */
    fun createFromEnvironment(): Embedder {
        val embedderType = System.getenv("RAG_EMBEDDER_TYPE")?.let {
            EmbedderType.valueOf(it.uppercase())
        } ?: EmbedderType.SIMPLE

        val config = mutableMapOf<String, String>()

        // Common configs
        System.getenv("RAG_EMBEDDER_DIMENSIONS")?.let { config["dimensions"] = it }

        // API-specific configs
        System.getenv("HUGGINGFACE_API_KEY")?.let { config["apiKey"] = it }
        System.getenv("OPENAI_API_KEY")?.let { config["apiKey"] = it }
        System.getenv("RAG_EMBEDDER_API_URL")?.let { config["apiUrl"] = it }
        System.getenv("RAG_EMBEDDER_MODEL")?.let { config["model"] = it }

        // ONNX-specific configs
        System.getenv("RAG_MODEL_PATH")?.let { config["modelPath"] = it }
        System.getenv("RAG_TOKENIZER_PATH")?.let { config["tokenizerPath"] = it }

        // Ollama-specific configs
        System.getenv("OLLAMA_BASE_URL")?.let { config["baseUrl"] = it }
        System.getenv("OLLAMA_EMBED_MODEL")?.let { config["modelName"] = it }

        return createEmbedder(embedderType, config)
    }
}
