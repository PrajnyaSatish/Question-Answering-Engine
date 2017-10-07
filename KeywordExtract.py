import nltk
import re, math

from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.porter import PorterStemmer

class KeywordExtractor():
    """ Extract the keywords from the question."""

    def extract_entity_names(self,t):
        """ Get the named entities.
            t = Parse tree of the sentence."""
        entity_names = []
        if hasattr(t, 'label') and t.label:
            if t.label() == 'NE':
                entity_names.append(' '.join([child[0] for child in t]))
            else:
                for child in t:
                    entity_names.extend(extract_entity_names(child))
        return entity_names    

        
    def query_named_entities(self,question):
        """ Extracts the named entities in the query."""
        ## get the Named Entitites such as names of people, locations, etc.
        ## These words are assigned rank 1
        question = re.sub(r'[^a-zA-Z0-9 ]',' ',question)
        token = nltk.word_tokenize(question)    
        tagged  = nltk.pos_tag(token)
        ne_chunked = nltk.ne_chunk(tagged, binary = True)
        entity_names = []    
        for tree in ne_chunked:
            entity_names.extend(self.extract_entity_names(tree))
        return entity_names,tagged
    

    def keyword_extractor(self,question):
        """ Extract the keyword from the question based on rules
            on the POS of the question."""
        stop = set(stopwords.words('english'))
        keywords = []
        entity_names,tagged = self.query_named_entities(question)

        ## Grammar : 2. Get complex nouns (nouns that occur consecutively) along
        ##              with their adjectives.
        ##           3. Get all other complex nominals.
        ##           4. Get nouns with their adjectives.
        ##           5. Get all other nouns.
        ##           6. Get all the verbs.
        ##           7. Any other words remaining
        grammar = r"""
            2: {<JJ.*|CD>+<NN.*><NN.*>+} 
            3: {<NN.*><NN.*>+}
            4: {<JJ.*|CD>+<NN.*>+}
            5: {<NN.*>+}
            6: {<VB.*>+}
            7: {<CC|CD|DT|EX|IN|JJ.*|LS|MD|PDT|POS|PRP.*>}
            7: {<RB.*|RP|SYM|TO|UH>}
            8: {<FW>}
            """
        CN_parser = nltk.RegexpParser(grammar)
        chunked = CN_parser.parse(tagged)
        for subtree in chunked.subtrees():
            t = subtree
            t = ' '.join(word for word,tag in t.leaves())
            keywords.append(tuple((str(subtree.label()),t)))
            keywords = [(tag, words) for (tag, words) in keywords if tag != 'S']
        
        ## Assign rank 1. to named entities
        keyW = []
        for rank,word in keywords:
            words = re.split(' ',word)
            for w in words:
                if w in entity_names:
                    rank_new = 1
                    keyW.append((rank_new,w))
                elif w not in stop:
                    rank_new = int(rank)
                    keyW.append((rank_new,w))
        return keyW


    def get_Query(self,keywords,rank_select):
        """ Get the final query terms from the keywords based on the rank."""
        queryTerms = []
        queryTerms = [word for rank,word in keywords if rank <= rank_select]
        queryTerms = [t.lower() for t in queryTerms]
        porter = PorterStemmer()
        queryTerms=[ str(porter.stem(word)) for word in queryTerms]
        return queryTerms

    
def main():
    question = raw_input("Enter your question: ")
    keywords = KeywordExtractor().keyword_extractor(question)
    rank_select = 7
    queryTerms = KeywordExtractor().get_Query(keywords,rank_select)
    if len(queryTerms) <= 1:
        queryBigrams = queryTerms
        queryBiTokens = queryBigrams
    else:
        queryBigrams = list(nltk.bigrams(queryTerms))
        queryBiTokens = [' '.join(token) for token in queryBigrams]
    print queryTerms,queryBiTokens

if __name__=='__main__':
    main()
