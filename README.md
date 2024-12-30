---

# Text Preprocessing and Word2Vec Model Training

## Overview
This project focuses on preprocessing text data for Natural Language Processing (NLP) tasks, training a Word2Vec model, and visualizing word embeddings. It uses the **Spam Classification** dataset to identify spam messages based on the processed text features.

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Text Preprocessing](#text-preprocessing)
4. [Word2Vec Model](#word2vec-model)
5. [Visualization](#visualization)
6. [Model Training](#model-training)
7. [Usage](#usage)
8. [Contributing](#contributing)

## Installation

To run the code, you will need to install the following libraries:

```bash
pip install nltk pandas gensim scikit-learn matplotlib seaborn
```

Ensure that you also download the necessary NLTK data files (i.e., stopwords, wordnet, punkt) for text processing.

## Dataset

The dataset used in this analysis is a **spam classification dataset**, specifically **spam.csv**, which contains text messages along with labels indicating whether the message is spam or ham (non-spam). 

Columns in the dataset:
- `v1`: Label column (ham/spam)
- `v2`: Message text column (text messages to be processed)

## Text Preprocessing

Text preprocessing includes several important steps:
1. **Tokenization**: Text is split into individual words (tokens).
2. **Lowercasing**: All words are converted to lowercase for uniformity.
3. **Punctuation Removal**: Punctuation marks are removed from the tokens.
4. **Stopword Removal**: Common words such as "the", "and", etc., are removed.
5. **Stemming**: Words are reduced to their root form (e.g., "running" becomes "run").
6. **Lemmatization**: Words are reduced to their base form (e.g., "better" becomes "good").

The function `preprocess_text` handles all of these steps.

## Word2Vec Model

Word2Vec is a technique to create word embeddings, where words are represented as vectors in a continuous vector space. In this project:
1. **Training**: The Word2Vec model is trained on the processed text data.
2. **Word Embeddings**: You can obtain the word embeddings (vectors) for individual words.

You can load the model and retrieve embeddings for any word in the vocabulary.

### Example:
```python
word_embedding = word2vec_model.wv['free']
```

## Visualization

1. **Most Frequent Words**: A bar chart visualizing the most common words in the dataset after preprocessing.
2. **Word Embedding Visualization**: Using **PCA** (Principal Component Analysis), we visualize a few word embeddings in 2D space to understand how similar words are placed near each other.

## Model Training

The dataset is split into training and testing sets using **train_test_split** from `scikit-learn`. The text data is then ready for further analysis or classification tasks, such as training a spam classifier using a logistic regression model or any other classification algorithm.

## Usage

1. Clone or download the repository.
2. Ensure that you have the required dataset (`spam.csv`) in the correct directory.
3. Install the necessary libraries using the command above.
4. Run the script to preprocess the text data, train the Word2Vec model, and visualize the results.
5. You can extend the project by training a classifier on the processed text data.

