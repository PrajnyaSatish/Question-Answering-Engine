import nltk
import re, math

from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.porter import PorterStemmer
from collections import Counter


class ParagraphFeatures:
    """ ParagraphFeatures() has methods to relevent paragraphs from documents
        based on their features"""

    def split_to_paragraph(self, document):
        """ Splits a document into is constituent paragraphs."""
        title = ''
        paragraphs = re.split(r'\n\n', document)
        if len(paragraphs[0]) < 100:
            title = paragraphs[0]
        return paragraphs

    def parse_para(self, paragraph):
        """ Parse a paragraph i.e, tokenize the words, remove stopwords and
            stem the words using Porter Stemmer."""
        para = re.sub(r'[^a-zA-Z0-9 ]', ' ', paragraph)
        tokens = nltk.word_tokenize(para)
        word_list = [word.lower() for word in tokens if word.istitle() or word]
        sw = stopwords.words('english')
        word_list = [x for x in word_list if x not in sw]
        porter = PorterStemmer()
        word_list = [str(porter.stem(word)) for word in word_list]
        return word_list

    def queryWordCount(self, queryTerms, paraTerms):
        """ returns the number of query terms that are present in the
            paragraph."""
        wordCount = 0
        for word in queryTerms:
            if word in paraTerms:
                wordCount += 1
        return wordCount

    def longestCommmonTerms(self, queryTerms, paraTerms):
        """ Returns in 'int' the number of query terms that are adjacent to
            each other in the paragraph."""
        table = [[0] * (len(paraTerms) + 1) for _ in range(len(queryTerms) + 1)]
        for i, ca in enumerate(queryTerms, 1):
            for j, cb in enumerate(paraTerms, 1):
                table[i][j] = (
                    table[i - 1][j - 1] + 1 if ca == cb else
                    max(table[i][j - 1], table[i - 1][j]))
        return table[-1][-1]

    def tfidfQuery(self, queryTerms, idf):
        """ Get the normalized tf-idf score of the query"""
        tf_idfQuery = []
        tfQuery = Counter(queryTerms)

        if len(queryTerms) == 0:
            return 0

        # Get the tf-idf of the query
        for word in queryTerms:
            tfQuery[word] = 1 + math.log(tfQuery[word])
            tf_idfQuery.append(float(tfQuery[word]) * idf[word])
        # Normalized tf-idf for the query
        norm_tf_idf = math.sqrt(sum(map(lambda x: x ** 2, tf_idfQuery)))
        try:
            return map(lambda x: x / norm_tf_idf, tf_idfQuery)
        except ZeroDivisionError:
            return [0]

    def cosine_similarity(self, query_terms, para_terms, idf):
        """Get the cosine similarity of the paragraph and the query by taking
            the dot product of the normalized tfidf of the query and the
            normalized tf of the document."""
        tfscore_para = []
        tf_para = Counter(para_terms)
        tf_idf_query_norm = self.tfidfQuery(query_terms, idf)
        if len(query_terms) == 0:
            return 0

        # Get the tf of the Para
        for word in query_terms:
            if word in tf_para:
                tfscore_para.append(tf_para[word])
            else:
                tfscore_para.append(0)

        ## Normalized tf for the paragraph            
        tfParaWeighted = map(lambda x: math.log(1 + x), tfscore_para)
        normFactor = math.sqrt(sum(map(lambda x: x ** 2, tfParaWeighted)))
        if normFactor == 0:
            return 0
        tfParaNormalized = map(lambda x: x / normFactor, tfParaWeighted)

        ## Dot Product
        if len(tf_idf_query_norm) != len(tfParaNormalized):
            return 0

        return sum(i[0] * i[1] for i in zip(tf_idf_query_norm, tfParaNormalized))

    def bigramCounter(self, queryBiTokens, wordList):
        """ Counts the number of bigrams that are common between the query
        and the paragraph."""
        paraBigrams = list(nltk.bigrams(wordList))
        paraBiTokens = [' '.join(token) for token in paraBigrams]
        BiCount = 0
        for word in queryBiTokens:
            if word in paraBiTokens:
                BiCount += 1
        return BiCount

    def namedEntCheck(self, Paragraph, namedEnts):
        """ Check whether the named entities in the query are present in the
            candidate paragraph."""
        for name in namedEnts:
            if name.lower() in Paragraph.lower():
                return 1
        return 0

    def paraFeatures(self, query, para, queryBiTokens, namedEnts, idf):
        """ get the features for paragragh ranking. """
        wordList = self.parse_para(para)
        ##    print
        ##    print para
        # First check how many words in the query are present in the document
        wordCount = self.queryWordCount(query, wordList)
        wordCountNorm = float(wordCount) / len(query)
        ##    print 'Num of query words in para:', wordCountNorm

        # Check for the longest sequence of question words.
        longestCommonLength = self.longestCommmonTerms(query, wordList)
        commonLengthNorm = float(longestCommonLength) / len(query)
        ##    print 'Longest sequence of query terms:',commonLengthNorm

        # Find the similarity score between the paragraph and the query terms
        simScore = self.cosine_similarity(query, wordList, idf)
        ##    print 'Cosine Similarity score:',simScore

        # Find the number of query bigrams that are also present in the paragraph
        bigram_count = self.bigramCounter(queryBiTokens, wordList)
        bigram_count_norm = float(bigram_count) / len(queryBiTokens)
        ##    print 'Num of similar bigrams: ', bigram_count_norm, queryBiTokens

        # Check if the named entities in the query are present in the paragraph
        named_ent_present = self.namedEntCheck(para, namedEnts)

        para_score = (wordCountNorm + (2 * commonLengthNorm) + simScore +
                     bigram_count_norm + named_ent_present) / 6
        #    print 'Score for the para:',para_score
        return para_score


def main():
    queryTerms = ['alcon', 'product']
    idf = {}
    idf['alcon'] = 0.5
    idf['product'] = 1
    named_ents = ['Alcon']
    paragraph = "Alcon employs nearly 14 thousand to manufacture Alcon products around the world. Alcon's manufacturing and distribution facilities are categorized by eye care products, with sites dedicated to the manufacturing of surgical equipment and devices, pharmaceuticals, and contact lenses and lens care products."
    queryBigrams = list(nltk.bigrams(queryTerms))
    queryBiTokens = [' '.join(token) for token in queryBigrams]
    feature_score = ParagraphFeatures().paraFeatures(queryTerms, paragraph,
                                                     queryBiTokens, named_ents, idf)
    print(feature_score)


if __name__ == "__main__":
    main()
