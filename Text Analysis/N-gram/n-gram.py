from nltk import ngrams

sentence = "I love to eat pizza."

words = sentence.split()

for i in range(1,5):
    n_grams = ngrams(words, i)
    print(f"{i}-grams")
    for gram in n_grams:
        print(' '.join(gram))
    print()