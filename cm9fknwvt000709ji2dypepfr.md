---
title: "🧠 Zero to Hero : Text Vectorization"
datePublished: Sun Apr 13 2025 11:39:11 GMT+0000 (Coordinated Universal Time)
cuid: cm9fknwvt000709ji2dypepfr
slug: zero-to-hero-text-vectorization
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1744543734968/4c5af39a-6261-48f7-beee-64ac1adee78c.webp
tags: nlp, tf-idf, word2vec, word-embedding, bag-of-words, glove, fasttext, text-vectorization

---

## 📌 What is Text Vectorization?

In NLP, machines can’t understand text directly — they understand numbers. **Text vectorization** is the process of converting textual data into numerical vectors so that we can feed them into machine learning or deep learning models.

---

## 1️⃣ Bag-of-Words (BoW)

### 🔍 Intuition:

Bag-of-Words is the simplest and most classic text vectorization technique. It represents text as a "bag" (multiset) of its words, **ignoring grammar and word order** but keeping multiplicity (frequency).

**Example:**  
Let’s say we have two sentences:

* “I love NLP”
    
* “I love AI and NLP”
    

We first build a vocabulary of unique words:  
`[I, love, NLP, AI, and]`

Each sentence becomes a vector of word counts:

| Sentence | I | love | NLP | AI | and |
| --- | --- | --- | --- | --- | --- |
| I love NLP | 1 | 1 | 1 | 0 | 0 |
| I love AI and NLP | 1 | 1 | 1 | 1 | 1 |

---

### 🧪 Code Example: Bag-of-Words with Scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample corpus
corpus = [
    "I love NLP",
    "I love AI and NLP"
]

# Initialize BoW Vectorizer
vectorizer = CountVectorizer()

# Fit and transform the corpus
X = vectorizer.fit_transform(corpus)

# Show feature names (vocabulary)
print("Vocabulary:", vectorizer.get_feature_names_out())

# Convert to array for better visualization
print("BoW Vectors:\n", X.toarray())
```

### 🧠 Notes:

* `CountVectorizer` automatically lowercases and tokenizes.
    
* Output is a **sparse matrix**; use `.toarray()` to view it.
    

---

## 2️⃣ N-Grams

### 🔍 Intuition:

**N-grams** capture **sequences** of words to retain partial context.

* **Unigram**: 1 word
    
* **Bigram**: 2-word sequence
    
* **Trigram**: 3-word sequence
    

For: “I love NLP”, the bigrams would be:

* “I love”
    
* “love NLP”
    

This helps capture short phrases, which are often meaningful.

---

### 🧪 Code Example: Bigrams with CountVectorizer

```python
# Bigram model
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))

# Transform corpus into bigrams
X_bigram = bigram_vectorizer.fit_transform(corpus)

# View vocabulary
print("Bigram Vocabulary:", bigram_vectorizer.get_feature_names_out())
print("Bigram Vectors:\n", X_bigram.toarray())
```

### 🧠 Notes:

* `ngram_range=(2,2)` means only bigrams.
    
* You can use `ngram_range=(1,2)` to include unigrams + bigrams.
    

---

## 3️⃣ TF-IDF (Term Frequency-Inverse Document Frequency)

### 🔍 Intuition:

BoW is simple but **treats all words equally**. Common words like "the", "is", etc. dominate even if they’re not informative.

TF-IDF weighs each word by:

* **TF** (term frequency): how often a word appears in a document.
    
* **IDF** (inverse document frequency): how *unique* the word is across documents.
    
    $$TF-IDF(w,d) = \text{TF}(w, d) \times \log\left(\frac{N}{DF(w)}\right)$$
    

Where:

* **N** is total number of documents
    
* **DF(w)** is the number of documents containing word ww
    

So **rare but meaningful words** get higher weight.

---

### 🧪 Code Example: TF-IDF with Scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

# Show vocabulary and TF-IDF values
print("TF-IDF Vocabulary:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Vectors:\n", X_tfidf.toarray())
```

### 🧠 Notes:

* TF-IDF is useful when you want to down-weight common terms.
    
* You can combine **n-grams** with TF-IDF too:
    

```python
# TF-IDF with unigrams and bigrams
tfidf_ngram = TfidfVectorizer(ngram_range=(1,2))
X_tfidf_ngram = tfidf_ngram.fit_transform(corpus)
```

---

## 📈 When to Use What?

| Technique | Pros | Cons | Use Case |
| --- | --- | --- | --- |
| BoW | Simple, fast | Ignores word order and context | Baseline models |
| N-grams | Adds partial context | High dimensionality | Text classification |
| TF-IDF | Weighs rare words more | Still lacks semantics | Keyword extraction, search |

# Word Embeddings

### 📌 What Are Word Embeddings?

Unlike traditional methods like **Bag-of-Words (BoW)** or **TF-IDF**, which treat words as discrete tokens, **word embeddings** represent words in **continuous vector spaces**. These embeddings capture **semantic relationships** between words, allowing machines to understand **similarity** (e.g., "king" is similar to "queen", "cat" is similar to "dog").

---

## 1️⃣ Word2Vec (Word to Vector)

### 🔍 Intuition:

**Word2Vec** is a predictive model that learns dense word vectors by predicting context (surrounding words) of a target word. It has two main architectures:

1. **CBOW (Continuous Bag of Words)**: Predicts a word given its context (surrounding words).
    
2. **Skip-gram**: Predicts the context (surrounding words) given a word.
    

### 🔥 Why Word2Vec?

Word2Vec embeddings are **learned based on context**, meaning it can capture semantic meaning, unlike BoW or TF-IDF. Words that appear in similar contexts will have similar vector representations.

**Example:**  
If we train Word2Vec on the sentence: "The cat sat on the mat", the embeddings of "cat" and "mat" might be close in the vector space.

---

### 🧪 Code Example: Training Word2Vec using Gensim

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download the NLTK tokenizer
nltk.download('punkt')

# Sample text corpus
corpus = [
    "I love programming in Python",
    "NLP is an exciting field",
    "Word embeddings help in NLP",
    "Word2Vec and GloVe are popular methods for embeddings"
]

# Tokenize each sentence
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Train the Word2Vec model
model = Word2Vec(tokenized_corpus, vector_size=50, window=3, min_count=1, sg=0)

# Get the vector for the word 'nlp'
vector_nlp = model.wv['nlp']
print(f"Word Vector for 'nlp':\n", vector_nlp)

# Find the most similar words to 'nlp'
similar_words = model.wv.most_similar('nlp', topn=3)
print(f"Words most similar to 'nlp':", similar_words)
```

### 🧠 Notes:

* `vector_size=50`: The size of the word vectors.
    
* `window=3`: The number of words around the target word to consider.
    
* `sg=0`: CBOW architecture. Set `sg=1` for Skip-gram.
    

---

## 2️⃣ GloVe (Global Vectors for Word Representation)

### 🔍 Intuition:

**GloVe** is a **count-based** method that constructs word vectors based on the **co-occurrence matrix** of a corpus. Unlike Word2Vec (which is predictive), GloVe tries to learn the word vectors by factoring the **global co-occurrence matrix** of words.

GloVe seeks to capture **global statistical information** about word co-occurrence, meaning that words that appear in similar contexts across the entire corpus will have similar vectors.

### 🔥 Why GloVe?

While Word2Vec is based on local context (windows), GloVe leverages the **entire corpus** to create better word representations, especially when there are large datasets.

---

### 🧪 Code Example: Using Pre-trained GloVe Embeddings

While you can train your own GloVe model, a more common approach is to use pre-trained GloVe vectors (trained on large corpora like Wikipedia or Common Crawl).

Let's download pre-trained GloVe embeddings and use them:

1. Download the **GloVe vectors** from: [GloVe embeddings](https://nlp.stanford.edu/projects/glove/)
    
2. Load the embeddings into Python:
    

```python
import numpy as np

# Load pre-trained GloVe vectors (Here we use the 50-dimensional version)
def load_glove_model(glove_file):
    model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            model[word] = vector
    return model

# Load GloVe embeddings
glove_model = load_glove_model("glove.6B.50d.txt")

# Get the vector for the word 'programming'
vector_programming = glove_model.get('programming')
print(f"Vector for 'programming':", vector_programming)

# Find similarity between two words (Euclidean distance or cosine similarity)
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([glove_model['python']], [glove_model['programming']])
print(f"Cosine similarity between 'python' and 'programming': {similarity[0][0]}")
```

### 🧠 Notes:

* Pre-trained GloVe embeddings can be used to save time if you have a large vocabulary and don’t want to train your own model.
    
* Cosine similarity is commonly used to measure similarity between word vectors.
    

---

## 3️⃣ FastText (Fast and Context-Aware Word Embeddings)

### 🔍 Intuition:

**FastText**, developed by Facebook, is an **extension of Word2Vec**. While Word2Vec treats words as atomic units, **FastText breaks words into smaller subword units** (character n-grams). This helps in capturing **out-of-vocabulary (OOV) words**.

For example, the word "playing" can be broken into subwords like "play", "lay", "ing", etc. This allows FastText to generate embeddings for words that were not seen during training, which is **very useful** for languages with rich morphology.

### 🔥 Why FastText?

FastText performs **better on morphologically rich languages** (e.g., Finnish, Turkish) and can generate embeddings for **rare words** or **misspelled words**.

---

### 🧪 Code Example: Using FastText from Gensim

```python
from gensim.models import FastText

# Train a FastText model (using the same corpus as before)
fasttext_model = FastText(tokenized_corpus, vector_size=50, window=3, min_count=1, sg=0)

# Get the vector for 'nlp'
vector_fasttext_nlp = fasttext_model.wv['nlp']
print(f"FastText Vector for 'nlp':", vector_fasttext_nlp)

# Get vector for an out-of-vocabulary word 'deep'
vector_deep = fasttext_model.wv['deep']
print(f"FastText Vector for 'deep':", vector_deep)

# Find most similar words to 'nlp'
similar_words_fasttext = fasttext_model.wv.most_similar('nlp', topn=3)
print(f"FastText Similar words to 'nlp':", similar_words_fasttext)
```

### 🧠 Notes:

* **Subword-based representation** helps FastText capture semantic meaning even for rare or unknown words.
    
* FastText handles **OOV words** and **word morphology** better than traditional methods.
    

---

## 📈 Comparison of Word Embeddings

| Feature | Word2Vec | GloVe | FastText |
| --- | --- | --- | --- |
| **Model Type** | Predictive (CBOW/Skip-gram) | Count-based (Co-occurrence matrix factorization) | Subword-based (Character n-grams) |
| **Strengths** | Learns context-sensitive embeddings | Captures global co-occurrence stats | Handles rare/OOV words well |
| **Weaknesses** | Needs large data to perform well | Static embeddings | Larger training time |
| **Use Case** | Text classification, Semantic similarity | Semantic similarity, Large corpus | Morphologically rich languages, OOV words |

---

## 📚 Conclusion

We've covered a broad spectrum of text vectorization techniques:

1. **Bag-of-Words (BoW)**: A simple yet effective method that ignores word order.
    
2. **N-grams**: A powerful extension that considers word sequences.
    
3. **TF-IDF**: Enhances BoW by weighing words based on their frequency across documents.
    
4. **Word2Vec**: Uses context to create dense and meaningful embeddings.
    
5. **GloVe**: A count-based model that captures global word relationships.
    
6. **FastText**: Extends Word2Vec to capture subword information, handling OOV words.
    

---

## Streamlit App 👉🏼 [Text Vectorization Visualizer](https://text-vectorization-visualizer.streamlit.app/) [\[ Click Me \]](https://text-vectorization-visualizer.streamlit.app/)