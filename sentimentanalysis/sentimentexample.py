# -*- coding: utf-8 -*-
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.tokenize import StanfordTokenizer
import math,itertools,codecs
from nltk.classify import MaxentClassifier

negfilepath = "F:/course/sentimentcode/sentiment_analysis_python-master/polarityData/rt-polaritydata/rt-polarity-neg.txt"
posfilepath = "F:/course/sentimentcode/sentiment_analysis_python-master/polarityData/rt-polaritydata/rt-polarity-pos.txt"

def getstopword():
    filename = "F:/course/sentimentcode/stopword.txt"
    file_object = codecs.open(filename,'r','utf-8')
    try:
        all_the_text = file_object.read()
        arr = all_the_text.split()
    finally:
        file_object.close()
    return arr
#all the words
def bagofwords(words):
    return dict([(word,True) for word in words])
    
#bigrams
def bigram(words, score_fn = BigramAssocMeasures.chi_sq, n = 1000):
    bigramfinder = BigramCollocationFinder.from_words(words)
    bigrams = bigramfinder.nbest(score_fn, n)
    return bagofwords(bigrams)

#mixed all the words with bigrams
def bigram_words(words, score_fn = BigramAssocMeasures.chi_sq, n=1000):
    bigramfinder = BigramCollocationFinder.from_words(words)
    bigrams = bigramfinder.nbest(score_fn, n)
    return bagofwords(words+bigrams)
    
def readwordarr(isTokenize = True):
    posWords = []
    negWords = []
    stopwords = getstopword()
    if isTokenize:
        tokenizer = StanfordTokenizer()
        with open(negfilepath, 'r', encoding = 'utf-8') as sentences:
            arr = tokenizer.tokenize(sentences.read())
            for line in arr:
                linearr = line.split()
                wordset = set()
                for word in linearr:
                    if word in stopwords:
                        continue
                    wordset.add(word) 
                negWords.append(list(wordset))
        with open(posfilepath, 'r', encoding = 'utf-8') as sentences:
            arr = tokenizer.tokenize(sentences.read())
            for line in arr:
                linearr = line.split()
                wordset = set()
                for word in linearr:
                    if word in stopwords:
                        continue
                    wordset.add(word)
                posWords.append(list(wordset))       
    else:
        with open(negfilepath, 'r', encoding = 'utf-8') as sentences:
            lines = sentences.readlines()
            for line in lines:
                linearr=line.split()
                wordset = set()
                for word in linearr:
                    if word in stopwords:
                        continue
                    wordset.add(word)
                negWords.append(list(wordset))
        with open(posfilepath, 'r', encoding = 'utf-8') as sentences:
            lines = sentences.readlines()
            for line in lines:
                linearr=line.split()
                wordset = set()
                for word in linearr:
                    if word in stopwords:
                        continue
                    wordset.add(word)
                posWords.append(list(wordset))
    return posWords,negWords

#calcuate word scores    
def create_word_scores(posWords,negWords, presense = False):
    # (posWords,negWords) = readwordarr()
    wordscores = {}
    wordfd = FreqDist()
    conditionwordfd = ConditionalFreqDist()
    if not presense:
        posWords = list(itertools.chain(*posWords))
        negWords = list(itertools.chain(*negWords))
        
        
        for word in posWords:
            wordfd[word]+=1
            conditionwordfd['pos'][word]+=1
            
        for word in negWords:
            wordfd[word]+=1
            conditionwordfd['neg'][word]+=1
    
    else:
        for wordarr in posWords:
            flag = dict()
            for word in wordarr:
                if word in flag:
                    continue
                flag[word]=1
                wordfd[word]+=1
                conditionwordfd['pos'][word]+=1
        for wordarr in negWords:
            flag = dict()
            for word in wordarr:
                if word in flag:
                    continue
                flag[word]=1
                wordfd[word]+=1
                conditionwordfd['neg'][word]+=1
                
    pos_word_count = conditionwordfd['pos'].N()
    neg_word_count = conditionwordfd['neg'].N()
    totalcount = pos_word_count + neg_word_count
    for word,freq in wordfd.items():
        pos_score = BigramAssocMeasures.chi_sq(conditionwordfd['pos'][word], (freq, pos_word_count), totalcount)
        neg_score = BigramAssocMeasures.chi_sq(conditionwordfd['neg'][word], (freq, neg_word_count), totalcount)
        wordscores[word] = pos_score + neg_score
    return wordscores

#calculate scores of word and bigram    
def create_word_bigram_scores(posWords, negWords, n = 5000):
    # (posWords,negWords) = readwordarr()
    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))
    bigramfinder = BigramCollocationFinder.from_words(posWords)
    posbigrams = bigramfinder.nbest(BigramAssocMeasures.chi_sq, n)
    bigramfinder = BigramCollocationFinder.from_words(negWords)
    negbigrams = bigramfinder.nbest(BigramAssocMeasures.chi_sq, n)
    posWords = posWords + posbigrams
    negWords = negWords + negbigrams
    wordscores = {}
    wordfd = FreqDist()
    conditionwordfd = ConditionalFreqDist()
    for word in posWords:
        wordfd[word]+=1
        conditionwordfd['pos'][word]+=1
        
    for word in negWords:
        wordfd[word]+=1
        conditionwordfd['neg'][word]+=1
    
    pos_word_count = conditionwordfd['pos'].N()
    neg_word_count = conditionwordfd['neg'].N()
    totalcount = pos_word_count + neg_word_count
    for word,freq in wordfd.items():
        pos_score = BigramAssocMeasures.chi_sq(conditionwordfd['pos'][word], (freq, pos_word_count), totalcount)
        neg_score = BigramAssocMeasures.chi_sq(conditionwordfd['neg'][word], (freq, neg_word_count), totalcount)
        wordscores[word] = pos_score + neg_score
    return wordscores
    
def findbestwords(wordscores, number):
    best_vals = sorted(wordscores.items(), key = lambda d:d[1], reverse = True)[:number]
    bestwords = set([w for w,s in best_vals]) 
    return bestwords
    
def bestwordfeature(words):
    return dict([(word, True) for word in words if word in bestwords])
    
def posfeature(posWords,featureextraction):
    posfeatures = []
    for pos in posWords:
        poswords = [featureextraction(pos),'pos']
        posfeatures.append(poswords)
    return posfeatures
    
def negfeature(negWords, featureextraction):
    negfeatures = []
    for neg in negWords:
        negwords = [featureextraction(neg), 'neg']
        negfeatures.append(negwords)
    return negfeatures

def seperate_train_test(posfeature, negfeature):
    posCutoff = int(math.floor(len(posfeature)*3/4))
    negCutoff = int(math.floor(len(negfeature)*3/4))
    train = posfeature[:posCutoff] + negfeature[:negCutoff]
    test = posfeature[posCutoff:] + negfeature[negCutoff:]
    return train,test
    
def score(trainset, testset, classifier):
    classifier = SklearnClassifier(classifier)
    classifier._vectorizer.sort = False
    classifier.train(trainset)
    (test, tag_test) = zip(*testset)
    pred = classifier.classify_many(test)
    return accuracy_score(tag_test, pred)
    
    
def maxenscore(trainset, testset):
    me_classifier = MaxentClassifier.train(trainset, algorithm='iis', trace=0, max_iter=1, min_lldelta=0.5)
    # (test, tag_test) = zip(*testset)
    # pred = me_classifier.classify(test)
    return nltk.classify.accuracy(me_classifier, testset)
    

(posWords, negWords) = readwordarr()
# posfeatures = posfeature(posWords, bagofwords)
# negfeatures = negfeature(negWords, bagofwords)
# (train,test) = seperate_train_test(posfeatures, negfeatures)
# print("....with all words....")
# print('BernoulliNB`s accuracy is %f' %score(train,test,BernoulliNB()))
# print('MultinomiaNB`s accuracy is %f' %score(train,test,MultinomialNB()))
# print('LogisticRegression`s accuracy is %f' %score(train,test,LogisticRegression()))
# print('SVC`s accuracy is %f' %score(train,test,SVC()))
# print('LinearSVC`s accuracy is %f' %score(train,test,LinearSVC()))
# print('NuSVC`s accuracy is %f' %score(train,test,NuSVC()))

# posfeatures = posfeature(posWords, bigram)
# negfeatures = negfeature(negWords, bigram)
# (train,test) = seperate_train_test(posfeatures, negfeatures)
# print("....with bigram  words....")
# print('BernoulliNB`s accuracy is %f' %score(train,test,BernoulliNB()))
# print('MultinomiaNB`s accuracy is %f' %score(train,test,MultinomialNB()))
# print('LogisticRegression`s accuracy is %f' %score(train,test,LogisticRegression()))
# print('SVC`s accuracy is %f' %score(train,test,SVC()))
# print('LinearSVC`s accuracy is %f' %score(train,test,LinearSVC()))
# print('NuSVC`s accuracy is %f' %score(train,test,NuSVC()))

# posfeatures = posfeature(posWords, bigram_words)
# negfeatures = negfeature(negWords, bigram_words)
# (train,test) = seperate_train_test(posfeatures, negfeatures)
# print("....with bigram and words....")
# print('BernoulliNB`s accuracy is %f' %score(train,test,BernoulliNB()))
# print('MultinomiaNB`s accuracy is %f' %score(train,test,MultinomialNB()))
# print('LogisticRegression`s accuracy is %f' %score(train,test,LogisticRegression()))
# print('SVC`s accuracy is %f' %score(train,test,SVC()))
# print('LinearSVC`s accuracy is %f' %score(train,test,LinearSVC()))
# print('NuSVC`s accuracy is %f' %score(train,test,NuSVC()))

#select 1000
word_score1 = create_word_scores(posWords,negWords, True)
# bestwords = findbestwords(word_score1, 1000)
# posfeatures = posfeature(posWords, bestwordfeature)
# negfeatures = negfeature(negWords, bestwordfeature)
# (train,test) = seperate_train_test(posfeatures, negfeatures)
# print("....with 1000 features....")
# print('BernoulliNB`s accuracy is %f' %score(train,test,BernoulliNB()))
# print('MultinomiaNB`s accuracy is %f' %score(train,test,MultinomialNB()))
# print('LogisticRegression`s accuracy is %f' %score(train,test,LogisticRegression()))
# print('SVC`s accuracy is %f' %score(train,test,SVC()))
# print('LinearSVC`s accuracy is %f' %score(train,test,LinearSVC()))
# print('NuSVC`s accuracy is %f' %score(train,test,NuSVC()))

# print("....with 100 features....")
# bestwords = findbestwords(word_score1, 100)
# posfeatures = posfeature(posWords, bestwordfeature)
# negfeatures = negfeature(negWords, bestwordfeature)
# (train,test) = seperate_train_test(posfeatures, negfeatures)

# print('BernoulliNB`s accuracy is %f' %score(train,test,BernoulliNB()))
# print('MultinomiaNB`s accuracy is %f' %score(train,test,MultinomialNB()))
# print('LogisticRegression`s accuracy is %f' %score(train,test,LogisticRegression()))
# print('SVC`s accuracy is %f' %score(train,test,SVC()))
# print('LinearSVC`s accuracy is %f' %score(train,test,LinearSVC()))
# print('NuSVC`s accuracy is %f' %score(train,test,NuSVC()))

print("....with 5000 features ....")

bestwords = findbestwords(word_score1, 5000)
posfeatures = posfeature(posWords, bestwordfeature)
negfeatures = negfeature(negWords, bestwordfeature)
(train,test) = seperate_train_test(posfeatures, negfeatures)
print('BernoulliNB`s accuracy is %f' %score(train,test,BernoulliNB()))
print('MultinomiaNB`s accuracy is %f' %score(train,test,MultinomialNB()))
print('LogisticRegression`s accuracy is %f' %score(train,test,LogisticRegression()))
print('SVC`s accuracy is %f' %score(train,test,SVC()))
print('LinearSVC`s accuracy is %f' %score(train,test,LinearSVC()))
print('NuSVC`s accuracy is %f' %score(train,test,NuSVC()))
print('Max entronopy `s accuracy is %f' %maxenscore(train,test))


# print("....with 10000 features....")

# bestwords = findbestwords(word_score1, 10000)
# posfeatures = posfeature(posWords, bestwordfeature)
# negfeatures = negfeature(negWords, bestwordfeature)
# (train,test) = seperate_train_test(posfeatures, negfeatures)
# print('BernoulliNB`s accuracy is %f' %score(train,test,BernoulliNB()))
# print("....with 1500 features....")

# bestwords = findbestwords(word_score1, 1500)
# posfeatures = posfeature(posWords, bestwordfeature)
# negfeatures = negfeature(negWords, bestwordfeature)
# (train,test) = seperate_train_test(posfeatures, negfeatures)
# print('BernoulliNB`s accuracy is %f' %score(train,test,BernoulliNB()))

# print("....with 1500 features....")

# bestwords = findbestwords(word_score1, 2500)
# posfeatures = posfeature(posWords, bestwordfeature)
# negfeatures = negfeature(negWords, bestwordfeature)
# (train,test) = seperate_train_test(posfeatures, negfeatures)
# print('BernoulliNB`s accuracy is %f' %score(train,test,BernoulliNB()))

# print("....with 3500 features....")

# bestwords = findbestwords(word_score1, 3500)
# posfeatures = posfeature(posWords, bestwordfeature)
# negfeatures = negfeature(negWords, bestwordfeature)
# (train,test) = seperate_train_test(posfeatures, negfeatures)
# print('BernoulliNB`s accuracy is %f' %score(train,test,BernoulliNB()))

# print("....mixing word and bigram....")
# word_score2 = create_word_bigram_scores(posWords,negWords)
# print("...with 5000 features....")
# bestwords = findbestwords(word_score2, 5000)
# posfeatures = posfeature(posWords, bestwordfeature)
# negfeatures = negfeature(negWords, bestwordfeature)
# (train,test) = seperate_train_test(posfeatures, negfeatures)
# print('BernoulliNB`s accuracy is %f' %score(train,test,BernoulliNB()))
# print('MultinomiaNB`s accuracy is %f' %score(train,test,MultinomialNB()))
# print('LogisticRegression`s accuracy is %f' %score(train,test,LogisticRegression()))
# print('SVC`s accuracy is %f' %score(train,test,SVC()))
# print('LinearSVC`s accuracy is %f' %score(train,test,LinearSVC()))
# print('NuSVC`s accuracy is %f' %score(train,test,NuSVC()))
# print("....with 2000 features....")
# bestwords = findbestwords(word_score2, 2000)
# posfeatures = posfeature(posWords, bestwordfeature)
# negfeatures = negfeature(negWords, bestwordfeature)
# (train,test) = seperate_train_test(posfeatures, negfeatures)
# print('BernoulliNB`s accuracy is %f' %score(train,test,BernoulliNB()))
# print('MultinomiaNB`s accuracy is %f' %score(train,test,MultinomialNB()))
# print('LogisticRegression`s accuracy is %f' %score(train,test,LogisticRegression()))
# print('SVC`s accuracy is %f' %score(train,test,SVC()))
# print('LinearSVC`s accuracy is %f' %score(train,test,LinearSVC()))
# print('NuSVC`s accuracy is %f' %score(train,test,NuSVC()))
# print("....with 7000 features....")
# bestwords = findbestwords(word_score2, 7000)
# posfeatures = posfeature(posWords, bestwordfeature)
# negfeatures = negfeature(negWords, bestwordfeature)
# (train,test) = seperate_train_test(posfeatures, negfeatures)
# print('BernoulliNB`s accuracy is %f' %score(train,test,BernoulliNB()))
# print('MultinomiaNB`s accuracy is %f' %score(train,test,MultinomialNB()))
# print('LogisticRegression`s accuracy is %f' %score(train,test,LogisticRegression()))
# print('SVC`s accuracy is %f' %score(train,test,SVC()))
# print('LinearSVC`s accuracy is %f' %score(train,test,LinearSVC()))
# print('NuSVC`s accuracy is %f' %score(train,test,NuSVC()))
# print("....with 10000 features....")
# bestwords = findbestwords(word_score2, 10000)
# posfeatures = posfeature(posWords, bestwordfeature)
# negfeatures = negfeature(negWords, bestwordfeature)
# (train,test) = seperate_train_test(posfeatures, negfeatures)
# print('BernoulliNB`s accuracy is %f' %score(train,test,BernoulliNB()))
# print('MultinomiaNB`s accuracy is %f' %score(train,test,MultinomialNB()))
# print('LogisticRegression`s accuracy is %f' %score(train,test,LogisticRegression()))
# print('SVC`s accuracy is %f' %score(train,test,SVC()))
# print('LinearSVC`s accuracy is %f' %score(train,test,LinearSVC()))
# print('NuSVC`s accuracy is %f' %score(train,test,NuSVC()))





    
    
    