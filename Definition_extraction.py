import nltk
import csv
from string import punctuation
import random
from nltk.tokenize import TreebankWordTokenizer

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

defTerms = []
candidateDef = []
labels = []

def separate_data_and_labels(data_in):
    Terms =[]
    Sentences = []
    Label = []
    for line in data_in:
        try:
            Terms.append(line[0])
        except:
            print(line)
        Sentences.append(line[1])
        try:
            Label.append(line[2])
        except:
            print(line)
    return Terms,Sentences,Label

def posFirstArticle(tokens):
    listOfArticles = set(['a','an','the'])
    for i,word in enumerate(tokens,1):
        if word.lower() in listOfArticles:
            return str(i)
    return str(0)

def numPunctuation(tokens):
    punctNum = 0
    for term in tokens:
        if term in punctuation:
            punctNum += 1
    if punctNum > 0:
        return float(punctNum)/len(tokens)
##            return punctNum
    else:
        return 0

def subWordPerdicate(candidateSent,defTerm,tokens):   
    startInd = 0
    endInd = 0
    predicateTagSet = set(['NN','NNP','NNS','NNPS','JJ'])
    tagged = nltk.pos_tag(tokens)
    if defTerm.lower() not in candidateSent.lower():
        return ''
    if ' ' in defTerm:
        defTerm = nltk.word_tokenize(defTerm)[-1]
    for index,(word,tag) in enumerate(tagged,1):
        if word.lower() == defTerm.lower():
            listOfWords = []
            startInd = index
            continue
        if word.lower() != defTerm.lower() and tag in predicateTagSet and startInd > 0:
            endInd = index
            if tagged[index-2][0] not in punctuation:
                break
    if endInd == 0:
        if startInd > 0:
            endInd = startInd
        else:
            endInd = 1
    listOfWords = tokens[startInd:endInd-1]
    return ' '.join(listOfWords)


def wordBeforeDef(tokens,defTerm):
    for index,word in enumerate(tokens):
        if word.lower()==defTerm.lower() and index > 0:
            return tokens[index-1]
    return ''

def wordPOS(tokens,defTerm):
    tagged = nltk.pos_tag(tokens)
    tokensLower = [word.lower() for word in tokens]
    tagset = [tag for (word,tag) in tagged]
    if len(tokens) <= 5:
        centerPOS = tagset
        leftPOS = tagset
        rightPOS = tagset
    if defTerm.lower() in tokensLower:
        wordIndex = tokensLower.index(defTerm.lower())
        leftLimitL = wordIndex
        rightLimitL = wordIndex+5
        leftLimitR = wordIndex- 4
        rightLimitR = wordIndex+1
        leftLimit = wordIndex - 2
        rightLimit = wordIndex+2+1  
        if wordIndex < 2 and wordIndex <= len(tokens)-2:
            leftLimit = 0
            rightLimit = wordIndex+2+1
        centerPOS = tagset[leftLimit:rightLimit]
        leftPOS = tagset[leftLimitL:rightLimitL]
        rightPOS = tagset[leftLimitR:rightLimitR]
        if wordIndex == 0:
            rightPOS.append(tagset[0])
        if wordIndex == 1:
            rightPOS = tagset[0:wordIndex+2]
    else:
        centerPOS = []
        leftPOS = []
        rightPOS = []
    wordcenterPOS = ' '.join(centerPOS)
    wordleftPOS = ' '.join(leftPOS)
    wordrightPOS = ' '.join(rightPOS)
    return wordcenterPOS,wordleftPOS,wordrightPOS


def getFeatures(featureInput):
    """ Get several features of the cndidate sentence"""
    featureList = []
    for defTerm,candidateSent in featureInput:
        tokens = nltk.word_tokenize(candidateSent)
        features = {}
        POScenter,POSleft,POSright = wordPOS(tokens,defTerm)
        features['Pos of first Article'] = posFirstArticle(tokens)
##            features['Num Punct Marks'] = numPunctuation(tokens)
        features['Subj words Predicate'] = subWordPerdicate(candidateSent,defTerm,tokens)
        features['Word before def term'] = wordBeforeDef(tokens,defTerm)
        features['POS centered word'] = POScenter
        features['POS left word'] = POSleft
##            features['POS right word'] = POSright            
        featureList.append(features)
    return featureList

    
class featureTransformer(TransformerMixin):
    
    def fit(self, *_):
        return self

    def transform(self,featureInput,*_):
        featureList = getFeatures(featureInput)
        return featureList



class defnExtractor(TransformerMixin):
    """Separate the definition term from the candidate sentence in the input."""
    def fit(self,*_):
        return self

    def transform(self,featureIn,*_):
        definitionList = [definition for defTerm,definition in featureIn]
        return definitionList

   
##______________________________________________________________________

from sklearn.linear_model import SGDClassifier

def main():    
    with open("DefinitionsCorpus.tsv") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        raw_data = list(tsvreader)
        
    raw_data = raw_data[1:]
    ##random.shuffle(raw_data)
    train_data = raw_data[100:]
    defTerms, candidateDef, labels = separate_data_and_labels(train_data)
    featureIn = zip(defTerms,candidateDef)

    test_data =raw_data[:100]
    defTerms_test, candidateDef_test, labels_test = separate_data_and_labels(test_data)
    featureIn_test = zip(defTerms_test,candidateDef_test)

    classifier = Pipeline([
        ('combinedFeatures',FeatureUnion([
            ('otherFeatures',Pipeline([
                ('FT',featureTransformer()),
                ('DictVect',DictVectorizer())
                ]))##,
    ##        ('bagOfWords',Pipeline([
    ##            ('definitions',defnExtractor()),
    ##            ('count_vectorizer', CountVectorizer(ngram_range=(2,5), min_df=1,
    ##                                tokenizer=TreebankWordTokenizer().tokenize))
    ##            ]))
            ])),
        ('SGDClassifier',SGDClassifier(n_iter = 110,loss = 'modified_huber'))
        ])

    classifier.fit(featureIn,labels)

    predictions = classifier.predict(featureIn_test)
    score = classifier.score(featureIn_test,labels_test)
    print('Num of test samples',len(test_data))
    print('Accuracy:',score)
    ##print '\n\n',predictions
    print('\n','done')
    ##

if __name__=="__main__":
    main()


