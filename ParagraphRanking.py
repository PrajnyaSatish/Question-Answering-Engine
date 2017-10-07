import nltk
import re, math

from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.porter import PorterStemmer
from collections import Counter

class ParagraphFeatures():
    """ ParagraphFeatures() has methods to relevent paragraphs from documents
        based on their features"""
    
    def splitToPara(self,Document):
        """ Splits a document into is constituent paragraphs."""
        title = ''
        paragraphs = re.split(r'\n\n',Document)
        if len(paragraphs[0]) < 100:
            title = paragraphs[0]
        return paragraphs

    def parsePara(self,Paragraph):
        """ Parse a paragraph i.e, tokenize the words, remove stopwords and
            stem the words using Porter Stemmer.""" 
        wordList = []
        Para=re.sub(r'[^a-zA-Z0-9 ]',' ',Paragraph)
        tokens = nltk.word_tokenize(Para)
        wordList = [word.lower() for word in tokens if word.istitle() or word]
        sw = stopwords.words('english')
        wordList = [x for x in wordList if x not in sw]
        porter = PorterStemmer()
        wordList = [ str(porter.stem(word)) for word in wordList]
        return wordList

    def queryWordCount(self,queryTerms,paraTerms):
        """ returns the number of query terms that are present in the
            paragraph."""
        wordCount = 0
        for word in queryTerms:
            if word in paraTerms:
                wordCount +=1
        return wordCount

    def longestCommmonTerms(self,queryTerms, paraTerms):
        """ Returns in 'int' the number of query terms that are adjacent to
            each other in the paragraph."""
        table = [[0] * (len(paraTerms) + 1) for _ in xrange(len(queryTerms) + 1)]
        for i, ca in enumerate(queryTerms, 1):
            for j, cb in enumerate(paraTerms, 1):
                table[i][j] = (
                    table[i - 1][j - 1] + 1 if ca == cb else
                    max(table[i][j - 1], table[i - 1][j]))
        return table[-1][-1]
    

    def tfidfQuery(self,queryTerms,idf):
        """ Get the normalized tf-idf score of the query"""    
        tf_idfQuery = []
        tfQuery = Counter(queryTerms)
        
        if len(queryTerms) == 0:
            return 0
        
        # Get the tf-idf of the query
        for word in queryTerms:        
            tfQuery[word] = 1+math.log(tfQuery[word])
            tf_idfQuery.append(float(tfQuery[word])*idf[word])
        ## Normalized tf-idf for the query
        normTfIdf = math.sqrt(sum(map(lambda x: x**2, tf_idfQuery)))
        try:
            return map(lambda x:x/normTfIdf, tf_idfQuery)
        except ZeroDivisionError: 
            return [0]


    def cosineSimilarity(self,queryTerms,paraTerms,idf):
        """Get the cosine similarity of the paragraph and the query by taking
            the dot product of the normalized tfidf of the query and the
            normalized tf of the document."""
        tfscorePara = []    
        tfPara = Counter(paraTerms)
        tfIdfQueryNorm = self.tfidfQuery(queryTerms,idf)
        if len(queryTerms) == 0:
            return 0
        
        ## Get the tf of the Para
        for word in queryTerms:        
            if word in tfPara:
                tfscorePara.append(tfPara[word])
            else:
                tfscorePara.append(0)
        
        ## Normalized tf for the paragraph            
        tfParaWeighted = map(lambda x:math.log(1+x), tfscorePara)
        normFactor = math.sqrt(sum(map(lambda x: x**2, tfParaWeighted)))
        if normFactor == 0:
            return 0
        tfParaNormalized = map(lambda x:x/normFactor, tfParaWeighted)

        ## Dot Product
        if len(tfIdfQueryNorm) != len(tfParaNormalized):
            return 0
        
        return sum(i[0] * i[1] for i in zip(tfIdfQueryNorm, tfParaNormalized))

    def bigramCounter(self,queryBiTokens,wordList):
        """ Counts the number of bigrams that are common between the query
        and the paragraph."""
        paraBigrams = list(nltk.bigrams(wordList))
        paraBiTokens = [' '.join(token) for token in paraBigrams]
        BiCount = 0
        for word in queryBiTokens:
            if word in paraBiTokens:
                BiCount += 1
        return BiCount

    def namedEntCheck(self,Paragraph,namedEnts):
        """ Check whether the named entities in the query are present in the
            candidate paragraph."""
        for name in namedEnts:
            if name.lower() in Paragraph.lower():
                return 1
        return 0

    def paraFeatures(self,query,para,queryBiTokens,namedEnts,idf):
        """ get the features for paragragh ranking. """
        wordList = self.parsePara(para)
    ##    print
    ##    print para
        # First check how many words in the query are present in the document
        wordCount = self.queryWordCount(query,wordList)
        wordCountNorm = float(wordCount)/len(query) 
    ##    print 'Num of query words in para:', wordCountNorm
        
        # Check for the longest sequence of question words.
        longestCommonLength = self.longestCommmonTerms(query, wordList)
        commonLengthNorm = float(longestCommonLength)/len(query)    
    ##    print 'Longest sequence of query terms:',commonLengthNorm

        # Find the similarity score between the paragraph and the query terms
        simScore = self.cosineSimilarity(query,wordList,idf)
    ##    print 'Cosine Similarity score:',simScore

        # Find the number of query bigrams that are also present in the paragraph
        bigramCount = self.bigramCounter(queryBiTokens,wordList)
        bigramCountNorm = float(bigramCount)/len(queryBiTokens)
    ##    print 'Num of similar bigrams: ', bigramCountNorm, queryBiTokens

        # Check if the named entities in the query are present in the paragraph
        namedEntPresent = self.namedEntCheck(para,namedEnts)

        paraScore = (wordCountNorm+(2*commonLengthNorm)+simScore+
                     bigramCountNorm+namedEntPresent)/6
    #    print 'Score for the para:',paraScore
        return paraScore


def main():
    queryTerms = ['alcon', 'product']
    idf = {}
    idf['alcon']=0.5
    idf['product']=1
    namedEnts = ['Alcon']
    paragraph = "Alcon employs nearly 14 thousand to manufacture Alcon products around the world. Alcon's manufacturing and distribution facilities are categorized by eye care products, with sites dedicated to the manufacturing of surgical equipment and devices, pharmaceuticals, and contact lenses and lens care products."
    queryBigrams = list(nltk.bigrams(queryTerms))
    queryBiTokens = [' '.join(token) for token in queryBigrams]
    feature_score = ParagraphFeatures().paraFeatures(queryTerms,paragraph,
                                                     queryBiTokens,namedEnts,idf)
    print feature_score

if __name__=="__main__":
    main()
    


