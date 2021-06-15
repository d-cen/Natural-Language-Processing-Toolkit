import nltk
import random
import pickle

from nltk.tokenize import word_tokenize
from statistics import mode
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

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

def find_features_tokenize(word_features, document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features  

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

if __name__ == "__main__":
    # Pickle sentiment analysis first, then pass the text
    sentiment_analysis()    
    print(sentiment_mod("This movie is terrible. The plot is bad and the character dipiction are silly."))
    print(sentiment_mod("This movie isso interesting. Full of context and vivid details."))
    
