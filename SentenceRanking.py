import nltk
import re

from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.porter import PorterStemmer

class SentenceFeatures():
    """ Class to rank sentences based on the query terms."""

    def SplitToSentence(self,Paragraph):
        """ Splits a paragraph into its constituent sentences."""
        replaced = re.sub('\n',' ',Paragraph)
        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', replaced)
        return [sent for sent in sentences if sent]

    def parseSent(self,candidateSentence):
        """ Parse the sentence i.e, split to tokens,remove stopwords,
        stem the words. """
        line = candidateSentence.lower()
        line=re.sub(r'[^a-z0-9 ]',' ',line) #put spaces instead of non-alphanumeric characters characters
        line=line.split()
        sw = stopwords.words('english')
        line=[x for x in line if x not in sw]  #eliminate the stopwords
        porter = PorterStemmer()
        Terms=[ str(porter.stem(word)) for word in line]
        return Terms

    def wordCount(self,queryTerms,sentTokens):
        """ Get the number of query words that are also present
        in the sentence."""
        qWordCount = 0
        for word in queryTerms:
            if word in sentTokens:
                qWordCount += 1
        return qWordCount

    def longestCommmonTerms(self,queryTerms, sentTokens):
        """ Get the longest common sequence of query words between the
        candidate sentence and the query words.
        Function returns a number which is represents the longest common
        sequence."""
        table = [[0] * (len(sentTokens) + 1) for _ in xrange(len(queryTerms) + 1)]
        for i, ca in enumerate(queryTerms, 1):
            for j, cb in enumerate(sentTokens, 1):
                table[i][j] = (
                    table[i - 1][j - 1] + 1 if ca == cb else
                    max(table[i][j - 1], table[i - 1][j]))
        return table[-1][-1]

    def noveltyFactor(self,queryTerms, parsedSentence):
        """ Returns the number of sentence terms not in the query."""
        return len(filter(lambda x: x not in queryTerms, parsedSentence))

    def ne_tag_sent(self,candidateSentence,AnswerToCheck):
        """ If there is a named entity same as the answer type then return 1,
        otherwise return 0.
        For ex - if the answer type is 'HUM' and there is a person tag 'PERSON',
        then return 1.
        AnswerToCheck has been mapped from Penn Tree bank
        - PERSON,GPE,CD to HUM,LOC,NUM."""
        sentTokens = nltk.word_tokenize(candidateSentence)
        tagged = nltk.pos_tag(sentTokens)
        ne_tagged = nltk.ne_chunk(tagged)
        for subtree in ne_tagged.subtrees():
            if subtree.label() == AnswerToCheck:
                return 1
        for word,tag in tagged:
            if tag == AnswerToCheck:
                return 1
        return 0

    def answerTypeMatching(self,candidateSentence,answerType,AnswerTypeToCheck):
        """The current nltk tagger gives good tags only for persons,places
        and numbers. So while doing NER recognition do only for types."""
        if answerType in AnswerTypeToCheck:
            ## Check to see if there are any named entities of the right answer type
            return self.ne_tag_sent(candidateSentence,AnswerTypeToCheck[answerType])
        else:
            return 0

    def sentFeatures(self,queryTerms,candidateSentence,answerType,paraScore):
        """ Features for sentence extraction.
        Function return a score in the range of 0 to 1. """
        sentTokens = nltk.word_tokenize(candidateSentence)
        parsedSent = self.parseSent(candidateSentence)
        ## Feature 1. Number of query words in the sentence
        queryWordCount = float(self.wordCount(queryTerms,parsedSent))/len(queryTerms)
##        print 'Word count:',queryWordCount
        
        ## Feature 2. Length of the longest sequence of question
        ## terms that appear in the answer
        longestSeqLength = float(self.longestCommmonTerms(queryTerms, parsedSent))/len(queryTerms)
##        print 'Longest common words:',longestSeqLength
        
        ## Feature 3. Novelty factor: A word in the candidate is
        ## not in the query.
        try:
            noveltyFactorSent = round(1-float(self.noveltyFactor(queryTerms, parsedSent))/len(parsedSent),2)
        except:
            noveltyFactorSent = 0
##        print 'Novelty factor:',noveltyFactorSent

        ## Feature 4. Candidate contains the correct answer type.
        AnswerTypesToCheck = {'HUM':'PERSON', 'LOC':'GPE', 'NUM':'CD'}
        answerTypeLabel = self.answerTypeMatching(candidateSentence,answerType,AnswerTypesToCheck)
##        print 'Answer Type Present:',answerTypeLabel

        ## Feature 5. Sentences with higher ranked paragraphs have a better score
##        print 'paraScore:',paraScore
        
        return round(((queryWordCount+(2*longestSeqLength)+noveltyFactorSent+
                       answerTypeLabel+paraScore)/6),4)


def main():
    Paragraph = """Focus Toric has been designed especially for correcting
    astigmatism. Toric soft contact lenses can give you the same vision clarity
    you enjoyed while wearing glasses. Advanced technology and unique prism
    design corrects astigmatism and is as affordable as a good pair of glasses."""
    sentences = SentenceFeatures().SplitToSentence(Paragraph)
    for s in sentences:
        print s
    queryTerms = ['ceo','alcon']
    candidateSentence = 'Mr. Mike Ball is the division head and CEO of Alcon.'
    answerType = 'HUM'
    paraScore = 0.8 ## Assumed
    sentFeat = SentenceFeatures().sentFeatures(queryTerms,candidateSentence,
                                           answerType,paraScore)
    print sentFeat

if __name__ == "__main__":
    main()
    
                                           
    


    

    




    

    
