import math
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def tokenize(document):
    # Tokenize the document (split into individual words)
    return document.lower().split()

def calculate_tf(tokens):
    # Calculate Term Frequency (TF)
    tf_counter = Counter(tokens)
    total_terms = len(tokens)
    tf = {term: count / total_terms for term, count in tf_counter.items()}
    return tf

def calculate_idf(documents):
    # Calculate Inverse Document Frequency (IDF)
    total_documents = len(documents)
    idf = {}
    for document in documents:
        tokens_set = set(tokenize(document))
        for term in tokens_set:
            idf[term] = idf.get(term, 0) + 1

    idf = {term: math.log(total_documents / count) for term, count in idf.items()}
    return idf

def calculate_tf_idf(tf, idf):
    # Calculate TF-IDF
    tf_idf = {term: tf_value * idf.get(term, 0) for term, tf_value in tf.items()}
    return tf_idf

# Example documents
documents = [
    "Cats are beautiful creatures.",
    "Dogs are loyal pets.",
    "Birds can fly in the sky."
]

# Tokenize documents
tokenized_documents = [tokenize(doc) for doc in documents]

# Calculate TF for each document
tfs = [calculate_tf(tokens) for tokens in tokenized_documents]

# Calculate IDF
idf = calculate_idf(documents)

# Calculate TF-IDF for "cats" in each document
term = "cats"
tf_idfs_for_cats = []

for tf in tfs:
    # Check if "cats" is in the document
    if term in tf:
        tf_idf = calculate_tf_idf({term: tf[term]}, idf)
    else:
        tf_idf = {term: 0}  # If "cats" is not in the document, TF-IDF is 0
    tf_idfs_for_cats.append(tf_idf)

# Extract TF-IDF values for visualization
tf_idf_values = [tf_idf[term] for tf_idf in tf_idfs_for_cats]

# Plotting the TF-IDF values
plt.figure(figsize=(8, 6))
plt.bar(range(len(tf_idf_values)), tf_idf_values, color='skyblue')
plt.xlabel('Document')
plt.ylabel('TF-IDF Score for "cats"')
plt.title('TF-IDF Scores for the term "cats" in Each Document')
plt.xticks(range(len(tf_idf_values)), ['Document {}'.format(i+1) for i in range(len(tf_idf_values))])
plt.show()
