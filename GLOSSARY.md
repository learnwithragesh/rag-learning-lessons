# RAG Glossary

Quick definitions of every term used in this course. The lesson number shows where each term is explained in depth.

| Term | Definition | Lesson |
|---|---|---|
| **RAG** | Retrieval-Augmented Generation: retrieve relevant text, add it to the prompt, let the LLM generate an answer from it | 1 |
| **LLM** | Large Language Model: generates text (llama3.1, Claude, GPT…) | 1 |
| **Hallucination** | A fluent, confident, but false statement made up by an LLM | 1 |
| **Knowledge cutoff** | The date after which a model has no training data | 1 |
| **Fine-tuning** | Further training a model on your data. Good for style and behaviour, poor for injecting facts | 1 |
| **Context window** | Maximum number of tokens a model can read in one prompt | 1 |
| **Token** | A piece of a word that models count in (≈ ¾ of an English word) | 3 |
| **Embedding** | A vector of numbers representing the meaning of a text | 2 |
| **Embedding model** | A neural network that turns text into embeddings (nomic-embed-text, all-minilm) | 2 |
| **Dimension** | The number of values in an embedding (768 for nomic-embed-text) | 2 |
| **Cosine similarity** | cos of the angle between two vectors: 1 = same direction, 0 = unrelated | 2 |
| **Normalisation** | Scaling a vector to length 1. Then dot product = cosine similarity | 2 |
| **Contrastive learning** | Training that pulls related pairs together and pushes unrelated pairs apart | 2 |
| **Asymmetric search / task prefix** | Different prefixes for queries vs documents (`search_query:` / `search_document:`) | 2 |
| **Chunk** | A piece of a document that's embedded and retrieved as a unit | 3 |
| **Chunk overlap** | Words repeated between neighbouring chunks so boundary facts aren't split | 3 |
| **Contextual chunk header** | Document title / section prepended to each chunk so it makes sense alone | 3 |
| **Contextual retrieval** | LLM-written context summary prepended to each chunk before indexing | 3, 7 |
| **Metadata** | Extra fields stored with a chunk (source, section, date, permissions) | 3, 4 |
| **Vector store / vector DB** | A database that stores embeddings and finds the most similar ones | 4 |
| **Top-k** | The k highest-scoring results of a search | 4 |
| **kNN** | k-nearest-neighbour search: exact, compares against every vector | 4 |
| **ANN** | Approximate nearest neighbour: fast, slightly inexact search for large collections | 4 |
| **HNSW** | Hierarchical Navigable Small World: the most common ANN graph index | 4 |
| **IVF / PQ** | Cluster-based index / vector compression for ANN | 4 |
| **Metadata filtering** | Restricting search to chunks whose metadata matches (e.g. permissions) | 4 |
| **Grounding** | Making the LLM answer only from the provided evidence | 5 |
| **Citation** | A reference in the answer (e.g. [1]) to the passage that supports it | 5 |
| **System prompt** | Instructions that set the model's rules and behaviour | 5 |
| **Prompt injection** | Malicious instructions hidden in input text (e.g. inside a document) | 5, 8 |
| **Temperature** | Randomness of generation: 0 = deterministic | 1, 5 |
| **Eval set** | Labelled questions with expected sources and answers, used to measure quality | 6 |
| **Hit@k / Recall@k** | Fraction of questions whose correct chunk is in the top-k | 6 |
| **MRR** | Mean Reciprocal Rank: average of 1/rank of the correct chunk | 6 |
| **Correctness** | Does the answer contain the right facts? | 6 |
| **Faithfulness / groundedness** | Is every claim in the answer supported by the retrieved passages? | 6 |
| **LLM-as-a-judge** | Using an LLM to grade outputs. Must be calibrated on known cases | 6 |
| **Judge calibration** | Testing a judge on labelled examples to see if it agrees with humans | 6 |
| **BM25** | Classic keyword ranking function (term frequency × rarity, length-normalised) | 7 |
| **TF / IDF** | Term frequency (count in chunk) / inverse document frequency (rarity across chunks) | 7 |
| **Stopwords** | Very common words ignored by keyword search ("the", "what") | 7 |
| **Hybrid search** | Combining keyword (BM25) and vector search | 7 |
| **RRF** | Reciprocal Rank Fusion: merges ranked lists by summing 1/(c + rank) | 7 |
| **Bi-encoder** | Encodes query and document separately (embeddings). Fast | 7 |
| **Cross-encoder / reranker** | Reads query and document together to score relevance. Accurate but slower | 7 |
| **Two-stage retrieval** | Fast retriever finds candidates → reranker picks the best | 7 |
| **HyDE** | Hypothetical Document Embeddings: search with an LLM-written fake answer | 7 |
| **Query rewriting / multi-query / decomposition** | Transforming the question before retrieval | 7 |
| **MMR** | Maximal Marginal Relevance: select relevant *and* diverse results | 7 |
| **Parent-document retrieval** | Search small chunks, return their larger parent section | 7 |
| **Ingestion pipeline** | Parse → chunk → embed → upsert, kept in sync with the sources | 8 |
| **Semantic cache** | Reusing answers for questions similar to previous ones | 8 |
| **Agentic RAG** | The LLM decides when and what to search, via tool calls in a loop | 8 |
| **GraphRAG** | Retrieval over a knowledge graph of entities and relations | 8 |
| **CRAG / Self-RAG** | The model checks retrieval quality and corrects course (re-search, fallback) | 8 |
