# Tontuno: A sophisticated Retrieval-Augmented Generation (RAG) system built in Kotlin that combines document storage, semantic search, and intelligent response generation.

> [!WARNING] 
> This is a example project to learn and practice
> Don't use in real applications or real environments

## Features

- **Multiple Embedding Options**: Support for Simple, Sentence Transformers API, ONNX local models, and OpenAI embeddings
- **Vector Storage**: In-memory vector database with cosine similarity search
- **Async Processing**: Built with Kotlin coroutines for non-blocking operations
- **Fallback System**: Automatic fallback to simple embedder if API services fail
- **Type Safety**: Full Kotlin type safety with comprehensive error handling
- **Testing**: Complete test suite included

## Quick Start

1. **Clone and build:**
   ```bash
   ./gradlew build
   ```

2. **Run the demo:**
   ```bash
   ./gradlew run
   ```

## Embedder Configuration

### Environment Variables

Set these environment variables to configure different embedders:

```bash
# Embedder type (SIMPLE, SENTENCE_TRANSFORMERS_API, ONNX_SENTENCE_TRANSFORMERS, OPENAI)
export RAG_EMBEDDER_TYPE=SENTENCE_TRANSFORMERS_API

# API keys
export HUGGINGFACE_API_KEY=your_hf_token_here
export OPENAI_API_KEY=your_openai_key_here

# Model configuration
export RAG_EMBEDDER_DIMENSIONS=384
export RAG_EMBEDDER_MODEL=text-embedding-3-small

# ONNX local model paths
export RAG_MODEL_PATH=/path/to/model.onnx
export RAG_TOKENIZER_PATH=/path/to/tokenizer.json
```

### Programmatic Configuration

```kotlin
// Using factory
val embedder = EmbedderFactory.createEmbedder(
   EmbedderFactory.EmbedderType.SENTENCE_TRANSFORMERS_API,
   mapOf(
      "apiKey" to "your_api_key",
      "dimensions" to "384"
   )
)

// Or from environment
val embedder = EmbedderFactory.createFromEnvironment()
```

## Usage Examples

### Basic Usage

```kotlin
val embedder = EmbedderFactory.createFromEnvironment()
val vectorStore = InMemoryVectorStore()
val ragAgent = RagAgent(embedder, vectorStore)

// Add documents
ragAgent.addDocument(Document("1", "Kotlin is a modern programming language"))

// Query
val response = ragAgent.query("What is Kotlin?")
println(response)

// Cleanup
ragAgent.close()
```

### Batch Processing

```kotlin
val documents = listOf(
   Document("1", "Kotlin documentation content"),
   Document("2", "Java interoperability guide"),
   Document("3", "Coroutines tutorial")
)

ragAgent.addDocuments(documents)
val response = ragAgent.query("How does Kotlin work with Java?")
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RagAgent      │    │   Embedder      │    │  VectorStore    │
│                 │────│                 │    │                 │
│ - addDocument() │    │ - embed()       │    │ - store()       │
│ - query()       │    │ - getDimensions │    │ - search()      │
│ - getStats()    │    │ - close()       │    │ - clear()       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│ ResponseGen     │──────────────┘
                        │                 │
                        │ - generateResp  │
                        └─────────────────┘
```

## Dependencies

- **Kotlin Coroutines**: Async processing
- **Ktor Client**: HTTP API calls
- **ONNX Runtime**: Local model inference
- **HuggingFace Tokenizers**: Text tokenization
- **Kotlin Serialization**: JSON handling

## Testing

Run tests:
```bash
./gradlew test
```

## Performance Notes

- **Simple Embedder**: Fastest, good for prototyping
- **API Embedders**: High quality, requires network
- **ONNX Local**: Good balance, requires model files
- **Vector Search**: O(n) linear search, consider specialized DBs for large datasets

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit pull request
