from collections import Counter

# Example documents
documents = [
    "The cat sat on the mat.",
    "The dog chased the cat."
]

# Tokenizing documents
tokenized_documents = [doc.lower().split() for doc in documents]

# Creating vocabulary
vocabulary = set(word for doc in tokenized_documents for word in doc)

# Creating Bag of Words representation
bag_of_words = []
for doc in tokenized_documents:
    word_count = Counter(doc)
    bow_vector = [word_count.get(word, 0) for word in vocabulary]
    bag_of_words.append(bow_vector)

# Printing Bag of Words representation
print("Vocabulary:", list(vocabulary))
for i, doc_vector in enumerate(bag_of_words, start=1):
    print(f"Document {i}:", doc_vector)

import matplotlib.pyplot as plt

# Plotting the Bag of Words representation
for i, doc_vector in enumerate(bag_of_words, start=1):
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(vocabulary)), doc_vector, tick_label=list(vocabulary))
    plt.title(f"Document {i} - Bag of Words")
    plt.xlabel("Word")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()