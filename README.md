# Movie-Review-Sentiment-Classifier
🎥 🎬 🎥 🎬 Classifies movie review sentences into positive or negative categories.

Uses the movie review corpus described here: http://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.pdf and classifies sentences based on their unigram/bigram features into positive and negative categories. Uses nltk, sklearn and numpy. Best results are with unigram-bigram features and a Multinomial Naive Bayes model, with around a 78% accuracy rate. Removes stop words, punctuation, whitespace, infrequently occurring features and words containing digits. Compares to a baseline model which guesses labels for sentences based on a random number generator.

Assignment for COMP 550- Natural Language Processing.
