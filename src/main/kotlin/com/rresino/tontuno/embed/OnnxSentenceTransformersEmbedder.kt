package com.rresino.tontuno.embed

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import mu.KotlinLogging
import java.io.File
import java.nio.file.Paths
import kotlin.math.sqrt

private val logger = KotlinLogging.logger {}

/**
 * Local ONNX-based Sentence Transformers embedder
 * This runs models locally without API calls
 */
class OnnxSentenceTransformersEmbedder(
    private val modelPath: String,
    private val tokenizerPath: String,
    private val dimensions: Int = 384
) : Embedder {

    private var ortSession: OrtSession? = null
    private var tokenizer: HuggingFaceTokenizer? = null
    private val ortEnvironment = OrtEnvironment.getEnvironment()

    init {
        initializeModel()
    }

    private fun initializeModel() {
        try {
            // Load ONNX model
            val modelFile = File(modelPath)
            if (!modelFile.exists()) {
                throw IllegalArgumentException("Model file not found: $modelPath")
            }

            val sessionOptions = OrtSession.SessionOptions()
            ortSession = ortEnvironment.createSession(modelPath, sessionOptions)

            // Load tokenizer
            val tokenizerFile = File(tokenizerPath)
            if (!tokenizerFile.exists()) {
                throw IllegalArgumentException("Tokenizer file not found: $tokenizerPath")
            }

            tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerPath))

            logger.info { "ONNX Sentence Transformers model loaded successfully" }

        } catch (e: Exception) {
            logger.error(e) { "Failed to initialize ONNX model" }
            throw e
        }
    }

    override suspend fun embed(text: String): FloatArray {
        return withContext(Dispatchers.Default) {
            try {
                val session = ortSession ?: throw IllegalStateException("Model not initialized")
                val tokenizer = tokenizer ?: throw IllegalStateException("Tokenizer not initialized")

                // Tokenize input text
                val encoding = tokenizer.encode(text)
                val inputIds = encoding.ids.map { it.toLong() }.toLongArray()
                val attentionMask = encoding.attentionMask.map { it.toLong() }.toLongArray()

                // Create ONNX tensors
                val inputIdsTensor = OnnxTensor.createTensor(ortEnvironment, arrayOf(inputIds))
                val attentionMaskTensor = OnnxTensor.createTensor(ortEnvironment, arrayOf(attentionMask))

                // Run inference
                val inputs = mapOf(
                    "input_ids" to inputIdsTensor,
                    "attention_mask" to attentionMaskTensor
                )

                val result = session.run(inputs)

                // Extract embeddings (usually the last hidden state)
                val embeddings = result.get(0).value as Array<Array<FloatArray>>

                // Mean pooling to get sentence embedding
                val sentenceEmbedding = meanPooling(embeddings[0], attentionMask.map { it.toFloat() }.toFloatArray())

                // Cleanup
                inputIdsTensor.close()
                attentionMaskTensor.close()
                result.close()

                logger.debug { "Generated ONNX embedding for text: ${text.take(50)}..." }
                sentenceEmbedding

            } catch (e: Exception) {
                logger.error(e) { "Error generating ONNX embedding" }
                throw e
            }
        }
    }

    /**
     * Perform mean pooling on token embeddings to get sentence embedding
     */
    private fun meanPooling(tokenEmbeddings: Array<FloatArray>, attentionMask: FloatArray): FloatArray {
        val dimensions = tokenEmbeddings[0].size
        val result = FloatArray(dimensions)
        var totalWeight = 0f

        for (i in tokenEmbeddings.indices) {
            val weight = attentionMask[i]
            totalWeight += weight

            for (j in 0 until dimensions) {
                result[j] += tokenEmbeddings[i][j] * weight
            }
        }

        // Average and normalize
        for (i in result.indices) {
            result[i] /= totalWeight
        }

        return normalizeVector(result)
    }

    /**
     * Normalize vector to unit length
     */
    private fun normalizeVector(vector: FloatArray): FloatArray {
        val norm = sqrt(vector.map { it * it }.sum())
        return if (norm > 0f) {
            vector.map { it / norm }.toFloatArray()
        } else {
            vector
        }
    }

    override fun getDimensions(): Int = dimensions

    override suspend fun close() {
        ortSession?.close()
        tokenizer?.close()
    }
}
