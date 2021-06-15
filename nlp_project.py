#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nltk
import random
import pickle

from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords, state_union
from nltk.corpus import gutenberg
from nltk.corpus import wordnet
from nltk.corpus import movie_reviews

from statistics import mode
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

# 1 NLTK introduction
def nltk_intro():
    example_text = "Hello Mr. Smith, how are you today? The weather is great and Python is awesome. The sky is pinkish-blue."
    print(sent_tokenize(example_text))
    print(word_tokenize(example_text))

# 2 Stop Words
def stop_words():
    example_sentence = "The is an example sentence showing stop word filtration."
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(example_sentence)
    filtered_sentence = []
    for w in words:
        if w not in stop_words:
            filtered_sentence.append(w)
    print(filtered_sentence)

# 3 Stemming
def porter_stemmer():
    ps = PorterStemmer()
    example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]
    for w in example_words:
        print(ps.stem(w))
    new_text = "Its is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
    words = word_tokenize(new_text)
    for w in words:
        print(ps.stem(w))

# 4,5,6 Part of Speech, Chunking, and Chinking
def process_content():
    train_text = state_union.raw("2005-GWBush.txt")
    sample_text = state_union.raw("2006-GWBush.txt")
    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    tokenized  = custom_sent_tokenizer.tokenize(sample_text)
    
    try:
       for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}""" # Chunking
            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{""" # Chinking
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
    except Exception as e:
        print(str(e))

# 7 Named Entity Recognition
def entity_recognition():
    train_text = state_union.raw("2005-GWBush.txt")
    sample_text = state_union.raw("2006-GWBush.txt")
    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    tokenized  = custom_sent_tokenizer.tokenize(sample_text)
    
    try:
       for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            namedEnt = nltk.ne_chunk(tagged)
            namedEnt.draw()
    except Exception as e:
        print(str(e))

# 8 Lemmatizing
def word_lemmatize():
    lemmatizer = WordNetLemmatizer()
    print(lemmatizer.lemmatize("better"))
    print(lemmatizer.lemmatize("better", pos="a"))

# 9 NLTK Corpora
def nltk_corpora():
    sample = gutenberg.raw("bible-kjv.txt")
    tok = sent_tokenize(sample)
    print(tok)

# 10 WordNet
def word_net():
    syns = wordnet.synsets("program")
    print(syns[0].name()) # synset
    print(syns[0].definition()) # definition
    print(syns[0].examples()) # examples

    synonyms = []
    antonyms = []
    for syn in wordnet.synsets("good"):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if (l.antonyms()):
                antonyms.append(l.antonyms()[0].name())
    print(set(synonyms))
    print(set(antonyms))

    w1 = wordnet.synset("ship.n.01")
    w2 = wordnet.synset("boat.n.01")
    print(w1.wup_similarity(w2))

# 11 Text Classification
def text_classification():
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append((list(movie_reviews.words(fileid)), category))
    random.shuffle(documents)
    
    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())
    
    all_words = nltk.FreqDist(all_words)
    print(all_words.most_common(15))
    print(all_words("stupid")) 

# 12, 13, 14 Words as Features, Naive Bayes, Pickle
def word_feature():
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append((list(movie_reviews.words(fileid)), category))
    random.shuffle(documents)
    
    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())
    
    all_words = nltk.FreqDist(all_words)
    
    word_features = list(all_words.keys())[:3000]
    #print((find_features(word_features, movie_reviews.words("neg/cv000_29416.txt"))))
    
    # Naive Bayes
    featuresets = [(find_features(word_features, rev), category) for (rev, category) in documents]
    training_set = featuresets[:1900]
    testing_set = featuresets[1900:]
    
    classifier = nltk.NaiveBayesClassifier.train(training_set) # posterior = prior occurence x liklihood / evidence
    print("Naive Bayes Algo Accuracy %: ", (nltk.classify.accuracy(classifier, testing_set))*100)
    classifier.show_most_informative_features(15)
    
    #Pickle
    #classifier_f = open("naivebayes.pickle", "rb")
    #classifier = pickle.load(classifier_f)
    #classifier_f.close()
    #print("Naive Bayes Algo Accuracy %: ", (nltk.classify.accuracy(classifier, testing_set))*100)
    #classifier.show_most_informative_features(15)
    
    #save_classifier = open("naivebayes.pickle", "wb")
    #pickle.dump(classifier, save_classifier)
    #save_classifier.close()

def find_features(word_features, document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
      
# 15, 16, 17 Scikit-Learn Incorporation, Combining Algos with Vote, and Investigating with Bias
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

def data_accuracy(training_set, testing_set):
    # Original Naive Bayes
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Original Naive Bayes Algo Accuracy %: ", (nltk.classify.accuracy(classifier, testing_set))*100)
    classifier.show_most_informative_features(15)
    
    # Multinomial Naive Bayes
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print("MNB_classifier Accuracy %: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
    
    # Bernoulli Naive Bayes
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    print("BernoulliNB_classifier Accuracy %: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

    # LogisticRegression
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    print("LogisticRegression_classifier Accuracy %: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
    
    # SGDClassifier
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    print("SGDClassifier_classifier Accuracy %: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
    
    # LinearSVC
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    print("LinearSVC_classifier Accuracy %: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
    
    # NuSVC
    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(training_set)
    print("NuSVC_classifier Accuracy %: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
    
    # Combining Algos with Vote
    voted_classifier = VoteClassifier(classifier, 
                                      MNB_classifier, 
                                      BernoulliNB_classifier, 
                                      LogisticRegression_classifier, 
                                      SGDClassifier_classifier, 
                                      LinearSVC_classifier, 
                                      NuSVC_classifier)
    
    print("voted_classifier Accuracy %: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
    print("Classification:", voted_classifier.classify(testing_set[0][0]), 
          "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[1][0]), 
          "Confidence %:", voted_classifier.confidence(testing_set[1][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[2][0]), 
          "Confidence %:", voted_classifier.confidence(testing_set[2][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[3][0]), 
          "Confidence %:", voted_classifier.confidence(testing_set[3][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[4][0]), 
          "Confidence %:", voted_classifier.confidence(testing_set[4][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[5][0]), 
          "Confidence %:", voted_classifier.confidence(testing_set[5][0])*100)

def scikit_learn():
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append((list(movie_reviews.words(fileid)), category))
    #random.shuffle(documents)
    
    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())    
    all_words = nltk.FreqDist(all_words)    
    word_features = list(all_words.keys())[:3000]
    
    featuresets = [(find_features(word_features, rev), category) for (rev, category) in documents]
    training_set = featuresets[:1900]
    testing_set = featuresets[1900:]
    data_accuracy(training_set, testing_set)

# 18 New Training Data
def find_features_tokenize(word_features, document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features    

def new_train():
    # Same method but different data sets
    documents = []
    all_words = []
    
    short_pos = open("short_reviews/positive.txt", "r").read()
    short_neg = open("short_reviews/negative.txt", "r").read()
    
    for r in short_pos.split('\n'):
        documents.append((r, "pos"))
    
    for r in short_neg.split("\n"):
        documents.append((r, "neg"))
        
    short_pos_words = word_tokenize(short_pos)
    short_neg_words = word_tokenize(short_neg)
    
    for w in short_pos_words:
        all_words.append(w.lower())
    
    for w in short_neg_words:
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)    
    word_features = list(all_words.keys())[:5000]
    
    featuresets = [(find_features_tokenize(word_features, rev), category) for (rev, category) in documents]
    random.shuffle(featuresets)
    
    training_set = featuresets[:10000]
    testing_set = featuresets[10000:]
    data_accuracy(training_set, testing_set)

# 19 Sentiment Analysis Module
def data_accuracy_pickle(training_set, testing_set):
    # Original Naive Bayes
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Original Naive Bayes Algo Accuracy %: ", (nltk.classify.accuracy(classifier, testing_set))*100)
    classifier.show_most_informative_features(15)
    
    save_classifier = open("pickled_algos/originalnaivebayes5k.pickle", "wb+")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()
    
    # Multinomial Naive Bayes
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print("MNB_classifier Accuracy %: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
        
    save_classifier = open("pickled_algos/MNB_classifier5k.pickle", "wb+")
    pickle.dump(MNB_classifier, save_classifier)
    save_classifier.close()
    
    # Bernoulli Naive Bayes
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    print("BernoulliNB_classifier Accuracy %: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
            
    save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle", "wb+")
    pickle.dump(BernoulliNB_classifier, save_classifier)
    save_classifier.close()

    # LogisticRegression
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    print("LogisticRegression_classifier Accuracy %: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
            
    save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle", "wb+")
    pickle.dump(LogisticRegression_classifier, save_classifier)
    save_classifier.close()
    
    # SGDClassifier
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    print("SGDClassifier_classifier Accuracy %: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
            
    save_classifier = open("pickled_algos/SGDClassifier_classifier5k.pickle", "wb+")
    pickle.dump(SGDClassifier_classifier, save_classifier)
    save_classifier.close()
    
    # LinearSVC
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    print("LinearSVC_classifier Accuracy %: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
            
    save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle", "wb+")
    pickle.dump(LinearSVC_classifier, save_classifier)
    save_classifier.close()
    
    # NuSVC
    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(training_set)
    print("NuSVC_classifier Accuracy %: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
            
    save_classifier = open("pickled_algos/NuSVC_classifier5k.pickle", "wb+")
    pickle.dump(NuSVC_classifier, save_classifier)
    save_classifier.close()
    
    # Combining Algos with Vote
    voted_classifier = VoteClassifier(classifier, 
                                      MNB_classifier, 
                                      BernoulliNB_classifier, 
                                      LogisticRegression_classifier, 
                                      SGDClassifier_classifier, 
                                      LinearSVC_classifier, 
                                      NuSVC_classifier)
    
    print("voted_classifier Accuracy %: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
    print("Classification:", voted_classifier.classify(testing_set[0][0]), 
          "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[1][0]), 
          "Confidence %:", voted_classifier.confidence(testing_set[1][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[2][0]), 
          "Confidence %:", voted_classifier.confidence(testing_set[2][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[3][0]), 
          "Confidence %:", voted_classifier.confidence(testing_set[3][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[4][0]), 
          "Confidence %:", voted_classifier.confidence(testing_set[4][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[5][0]), 
          "Confidence %:", voted_classifier.confidence(testing_set[5][0])*100)

def sentiment_analysis():
    documents = []
    all_words = []
    
    short_pos = open("short_reviews/positive.txt", "r").read()
    short_neg = open("short_reviews/negative.txt", "r").read()

    # j=adjective, r=adverb, v=verb
    allowed_word_types = ["J"]
    
    for p in short_pos.split('\n'):
        documents.append( (p, "pos") )
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())
    
    for p in short_neg.split('\n'):
        documents.append( (p, "neg") )
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())
    
    save_documents = open("pickled_algos/documents.pickle", "wb+")
    pickle.dump(documents, save_documents)
    save_documents.close()
    
    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:5000]
    
    save_word_features = open("pickled_algos/word_features5k.pickle", "wb+")
    pickle.dump(word_features, save_word_features)
    save_word_features.close()
    
    featuresets = [(find_features_tokenize(word_features, rev), category) for (rev, category) in documents]    
    random.shuffle(featuresets)
    
    training_set = featuresets[:10000]
    testing_set = featuresets[10000:]
    data_accuracy_pickle(training_set, testing_set)

def sentiment_mod(text):    
    word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
    word_features = pickle.load(word_features5k_f)
    word_features5k_f.close()
    
    # Original Naive Bayes    
    open_file = open("pickled_algos/originalnaivebayes5k.pickle", "rb")
    classifier = pickle.load(open_file)
    open_file.close()
    
    # Multinomial Naive Bayes
    open_file = open("pickled_algos/MNB_classifier5k.pickle", "rb")
    MNB_classifier = pickle.load(open_file)
    open_file.close()
    
    # Bernoulli Naive Bayes
    open_file = open("pickled_algos/BernoulliNB_classifier5k.pickle", "rb")
    BernoulliNB_classifier = pickle.load(open_file)
    open_file.close()

    # LogisticRegression
    open_file = open("pickled_algos/LogisticRegression_classifier5k.pickle", "rb")
    LogisticRegression_classifier = pickle.load(open_file)
    open_file.close()
    
    # SGDClassifier
    open_file = open("pickled_algos/SGDClassifier_classifier5k.pickle", "rb")
    SGDClassifier_classifier = pickle.load(open_file)
    open_file.close()
    
    # LinearSVC
    open_file = open("pickled_algos/LinearSVC_classifier5k.pickle", "rb")
    LinearSVC_classifier = pickle.load(open_file)
    open_file.close()
    
    # NuSVC
    open_file = open("pickled_algos/NuSVC_classifier5k.pickle", "rb")
    NuSVC_classifier = pickle.load(open_file)
    open_file.close()
    
    # Combining Algos with Vote
    voted_classifier = VoteClassifier(classifier, 
                                      MNB_classifier, 
                                      BernoulliNB_classifier, 
                                      LogisticRegression_classifier, 
                                      SGDClassifier_classifier, 
                                      LinearSVC_classifier, 
                                      NuSVC_classifier)
    
    # Sentiment
    feats = find_features_tokenize(word_features, text)
    
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

# Pickle sentiment analysis first, then pass the text
#sentiment_analysis()    
#print(sentiment_mod("This movie is terrible. The plot is bad and the character dipiction are silly."))
#print(sentiment_mod("This movie isso interesting. Full of context and vivid details."))
    