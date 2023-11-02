import nltk
import os, glob
import re, math

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import defaultdict, Counter, OrderedDict
from sklearn.externals import joblib
from nltk.corpus import wordnet as wn


### -------------- 1. Answer Type Classification ----------------------- ###

def get_tags(question):
    que =  question.lower()
    #remove question marks
    que1 = que.replace('?','')
    token = nltk.word_tokenize(que1)    
    tagged = nltk.pos_tag(token)
    tag_dict = {}
    for (word,tag) in tagged:
        tag_dict[word]=tag
    return tag_dict    

def get_predictions(question):
    test_features = get_tags(question)
    return SVC_Classifier.classify(test_features) 


### ------------------ 2. Get the keywords ----------------------------- ###

def extract_entity_names(t):
    entity_names = []
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
    return entity_names


def keyword_extractor(question):
    stop = set(stopwords.words('english'))
    keywords = []
    question = re.sub(r'[^a-zA-Z0-9 ]',' ',question)
    token = nltk.word_tokenize(question)    
    tagged  = nltk.pos_tag(token)
    ## get the Named Entitites such as names of people, locations, etc.
    ## These words are assigned rank 1
    ne_chunked = nltk.ne_chunk(tagged, binary = True)
    entity_names = []    
    for tree in ne_chunked:
        entity_names.extend(extract_entity_names(tree))
    
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


def get_Query(keywords,rank_select):
	queryTerms = []
	queryTerms = [word for rank,word in keywords if rank <= rank_select]
	queryTerms = [t.lower() for t in queryTerms]
	porter = PorterStemmer()
	queryTerms=[ str(porter.stem(word)) for word in queryTerms]
	return queryTerms

    
## Get the tf-idf score of the query
def tfidfQuery(queryTerms,idf):
    """ get the normalized tf-idf score of the query"""    
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
        return 0



### ---------------------------- **** ------------------------------------- ###


### ------------------ Perform Information Retrieval ---------------------- ###

def getNames(Document):
    """ Send all the words with the 'NNP' tag for soundex codes.
    These words are names."""
    line=re.sub(r'[^a-zA-Z0-9 ]',' ',Document)
    tokens = nltk.word_tokenize(line)
    tagged = nltk.pos_tag(tokens)
    properNouns = [word for (word,tag) in tagged if tag == 'NNP']
    return properNouns
    

def getSoundex(name):
    """Get the soundex code for the string"""
    name = name.upper()
    soundex = ""
    soundex += name[0]
    dictionary = {"BFPV": "1", "CGJKQSXZ":"2", "DT":"3", "L":"4", "MN":"5", "R":"6", "AEIOUHWY":"."}

    for char in name[1:]:
        for key in dictionary.keys():
            if char in key:
                code = dictionary[key]
                if code != soundex[-1]:
                    soundex += code

    soundex = soundex.replace(".", "")
    soundex = soundex[:4].ljust(4, "0")
    return soundex


def getTerms(Document):
    """ This returns the stemmed terms after removing the stopwords
        in the documents"""
    line = Document.lower()
    line=re.sub(r'[^a-z0-9 ]',' ',line) #put spaces instead of non-alphanumeric characters characters
    line=line.split()
    sw = stopwords.words('english')
    line=[x for x in line if x not in sw]  #eliminate the stopwords
    porter = PorterStemmer()
    Terms=[ str(porter.stem(word)) for word in line]
    return Terms

def createIndex(DocID,Terms):
    invIndex = {}
    for index,word in enumerate(Terms):
        try:
            invIndex[word][1].append(index)
        except:
            invIndex[word] = [DocID,[index]]
    return invIndex		

def getPostings(terms):
    #all terms in the list are guaranteed to be in the index
    return [invIndex[term] for term in terms]

def getDocsFromPostings(postings):
    #no empty list in postings
    return [[x[0] for x in p] for p in postings]
    

def intersectLists(lists):
    #get a list of documents in which query terms are present
    if len(lists)==0:
        return []
    lists.sort(key=len)
    return list(reduce(lambda x,y: set(x)|set(y),lists))
    

def extractDocs(q):
##    length = len(q)
##    for term in q:
##        if term not in invIndex:
##            #if a term doesn't appear in the index
##            #there can't be any document maching it
##            return []

    postings = getPostings(q)   #all the query terms are in the index
    docs = getDocsFromPostings(postings)    #get the documents indexed for each terms
    docs = intersectLists(docs)  #get the documents tha contain all the query terms

    return docs


def buildInvIndex():
    """ this function builds the inverted index of all the documents."""
    invIndex =  {}
    df = defaultdict(int)
    idf = defaultdict(float)
    numOfTextFiles = 0

    porter = PorterStemmer()
    soundex = defaultdict(list)
    doubleMetaphone = defaultdict(list)
    
    for filename in glob.glob('TextFiles/*'):
        f = open(filename, 'rU')
        raw_data = f.read()
        f.close()

        Terms = getTerms(raw_data)
        DocName = filename.split('\\')[-1]
        pageIndex = createIndex(DocName,Terms)
        numOfTextFiles += 1

        #Build the soundex (phonetic similarity) hashtable for all the names
        properNouns = getNames(raw_data)
        for word in set(properNouns):
            soundex_code = getSoundex(word)
            stemmedWord = str(porter.stem(word.lower()))
            if stemmedWord in pageIndex:
                if soundex_code in soundex and stemmedWord not in soundex[soundex_code]:
                    soundex[soundex_code].extend([stemmedWord])
                else:
                    soundex[soundex_code] = [stemmedWord]
                    

        for word in pageIndex:
            ## Create the df scores of all the words
            postings = pageIndex[word]
            df[word] += len(postings[1])
            
            ## Merging the page dictionaries to form the meta - dictionary
            if word in invIndex:
                invIndex[word].extend(element for element in [pageIndex[word]])
            else:
                invIndex[word] = [pageIndex[word]]

    # Build the idf for all the words in the corpus
    for word in df:
        idf[word] = math.log(numOfTextFiles/float(df[word]))

    return invIndex,df,idf,numOfTextFiles,soundex


### --------------------------- *********** ----------------------- ###

### ----------------- Answer Extraction -------------------------- ###

def splitToPara(Document):
    """ Splits a document into is constituent paragraphs."""
    title = ''
    paragraphs = re.split(r'\n\n',Document)
    if len(paragraphs[0]) < 100:
        title = paragraphs[0]
    return title,paragraphs[1:]


def parsePara(Paragraph):
    wordList = []
    Para=re.sub(r'[^a-zA-Z0-9 ]',' ',Paragraph)
    tokens = nltk.word_tokenize(Para)
    wordList = [word.lower() for word in tokens if word.istitle() or word]
    sw = stopwords.words('english')
    wordList = [x for x in wordList if x not in sw]
    porter = PorterStemmer()
    wordList = [ str(porter.stem(word)) for word in wordList]
    return wordList



### -------------- 1. Etraction of features for paragragh ranking -------- ###
def queryWordCount(queryTerms,paraTerms):
    wordCount = 0
    for word in queryTerms:
        if word in paraTerms:
            wordCount +=1
    return wordCount


def longestCommmonTerms(queryTerms, paraTerms):
    table = [[0] * (len(paraTerms) + 1) for _ in range(len(queryTerms) + 1)]
    for i, ca in enumerate(queryTerms, 1):
        for j, cb in enumerate(paraTerms, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]


def cosineSimilarity(queryTerms,paraTerms,tfIdfQueryNorm):
    """Get the cosine similarity of the paragraph and the query by taking the
    dot product of the normalized tfidf of the query and the normalized
    tf of the document
    """
    tfscorePara = []    
    tfPara = Counter(paraTerms)
    
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
            
def bigramCounter(queryBiTokens,wordList):
    paraBigrams = list(nltk.bigrams(wordList))
    paraBiTokens = [' '.join(token) for token in paraBigrams]
    BiCount = 0
    for word in queryBiTokens:
        if word in paraBiTokens:
            BiCount += 1
    return BiCount
    
### ----------------- ******************* ---------------------- ###

def paraFeatures(query,para,queryBiTokens):
    """ get the features for paragragh ranking. """
    wordList = parsePara(para)
##    print
##    print para
    # First check how many words in the query are present in the document
    wordCount = queryWordCount(query,wordList)
    wordCountNorm = float(wordCount)/len(query) 
##    print 'Num of query words in para:', wordCountNorm
    
    # Check for the longest sequence of question words.
    longestCommonLength = longestCommmonTerms(query, wordList)
    commonLengthNorm = float(longestCommonLength)/len(query)    
##    print 'Longest sequence of query terms:',commonLengthNorm

    # Find the similarity score between the paragraph and the query terms
    simScore = cosineSimilarity(query,wordList,tfIdfQueryNorm)
##    print 'Cosine Similarity score:',simScore

    # Find the number of query bigrams that are also present in the paragraph
    bigramCount = bigramCounter(queryBiTokens,wordList)
    bigramCountNorm = float(bigramCount)/len(queryBiTokens)
##    print 'Num of similar bigrams: ', bigramCountNorm, queryBiTokens

    paraScore = (wordCountNorm+(2*commonLengthNorm)+simScore+bigramCountNorm)/5
#    print 'Score for the para:',paraScore
    return paraScore


def getReleventParas(queryTerms, queryBiTokens):    
    paraScores = []
    for filename in releventDocs:
        f = open(os.path.join('TextFiles', filename),'rU')
        raw_paragraphs = f.read()
        f.close()
        
        title, splitParas = splitToPara(raw_paragraphs)
        for para in splitParas:
            paraScoreTemp = paraFeatures(queryTerms,para,queryBiTokens)
            if paraScoreTemp != float(0):
                paraScores.append((round(paraScoreTemp,4),para))
    return paraScores


### ----------------- ******************* ---------------------- ###

### -------------- 2. Etraction of features for sentence ranking -------- ###

def SplitToSentence(Paragraph):
    replaced = re.sub('\n',' ',Paragraph)
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', replaced)
    return [sent for sent in sentences if sent]

def parseSent(candidateSentence):
    line = candidateSentence.lower()
    line=re.sub(r'[^a-z0-9 ]',' ',line) #put spaces instead of non-alphanumeric characters characters
    line=line.split()
    sw = stopwords.words('english')
    line=[x for x in line if x not in sw]  #eliminate the stopwords
    porter = PorterStemmer()
    Terms=[ str(porter.stem(word)) for word in line]
    return Terms

    
def wordCount(queryTerms,sentTokens):    
    qWordCount = 0
    for word in queryTerms:
        if word in sentTokens:
            qWordCount += 1
    return qWordCount



def noveltyFactor(queryTerms, parsedSentence):
    return len(filter(lambda x: x not in queryTerms, parsedSentence))
    

def ne_tag_sent(candidateSentence,AnswerToCheck):
    sentTokens = nltk.word_tokenize(candidateSentence)
    tagged = nltk.pos_tag(sentTokens)
    ne_tagged = nltk.ne_chunk(tagged)
    for subtree in ne_tagged.subtrees():
        if subtree.label() == AnswerToCheck:
            return 1
    return 0

## The current nltk tagger gives good tags only for persons,places and numbers
## So while doing NER recognition do only for types.
def answerTypeMatching(candidateSentence,answerType,AnswerTypeToCheck):
    if answerType in AnswerTypeToCheck:
        ## Check to see if there are any named entities of the right answer type
        return ne_tag_sent(candidateSentence,AnswerTypeToCheck[answerType])
    else:
        return 0

                
def sentFeatures(queryTerms,candidateSentence,answerType,paraScore):
    """ Features for sentence extraction."""
    parsedSent = parseSent(candidateSentence)
    ## Feature 1. Number of query words in the sentence
    queryWordCount = float(wordCount(queryTerms,parsedSent))/len(queryTerms)
#    print 'Word count:',queryWordCount
    
    ## Feature 2. Length of the longest sequence of question
    ## terms that appear in the answer
    longestSeqLength = float(longestCommmonTerms(queryTerms, parsedSent))/len(queryTerms)
#    print 'Longest common words:',longestSeqLength
    
    ## Feature 3. Novelty factor: A word in the candidate is
    ## not in the query.
    try:
        noveltyFactorSent = round(1-float(noveltyFactor(queryTerms, parsedSent))/len(parsedSent),2)
    except:
        noveltyFactorSent = 0
#    print 'Novelty factor:',noveltyFactorSent

    ## Feature 4. Candidate contains the correct answer type.
    AnswerTypesToCheck = {'HUM':'PERSON', 'LOC':'GPE', 'NUM':'CD'}
    answerTypeLabel = answerTypeMatching(candidateSentence,answerType,AnswerTypesToCheck)
#    print 'Answer Type Present:',answerTypeLabel

    ## Feature 5. Sentences with higher ranked paragraphs have a better score
#    print 'paraScore:',paraScore
    
    return round(((queryWordCount+(2*longestSeqLength)+noveltyFactorSent+
                   answerTypeLabel+paraScore)/6),4)


def getSent(releventParas,queryTerms,answerType):
    releventSent = []
    candidates = [(score,SplitToSentence(para)) for score,para in releventParas]
    for score,paraCandidate in candidates:
        for sentence in paraCandidate:
            sentenceScore = sentFeatures(queryTerms,sentence,answerType,score)
            if sentenceScore > 0.2:
                releventSent.append((sentenceScore,sentence))
    return releventSent

### ----------------- ******************* ---------------------- ###


def getQuestion():
    question = input("Enter your question: ")
    keywords = keyword_extractor(question)
    rank_select = 7
    queryTerms = get_Query(keywords,rank_select)
    if len(queryTerms) <= 1:
        queryBigrams = queryTerms
        queryBiTokens = queryBigrams
    else:
        queryBigrams = list(nltk.bigrams(queryTerms))
        queryBiTokens = [' '.join(token) for token in queryBigrams]
    return question,queryTerms,queryBiTokens


def questionProc(question):
    porter = PorterStemmer()
    questionTokens = nltk.word_tokenize(question)
    questionTagged = nltk.pos_tag(questionTokens)
    questionStemmed = [str(porter.stem(word.lower())) for word in questionTokens]
    allTerms = OrderedDict(zip(questionStemmed,questionTagged))
    return allTerms

def modifyTerm(queStemTag,termToModify):
    porter = PorterStemmer()
    alphabet_sets = set(["BFPV","CGJKQSXZ", "DT", "MN","AE","OU","EI"])
    if queStemTag[termToModify][1] in ['NN','NNP']:
        soundex_code = getSoundex(queStemTag[termToModify][0])
        if soundex_code in SoundexCodes:
        # If the word has the same soundex code as another word in the text files
            if len(SoundexCodes[soundex_code]) == 1:
            ## If there is only one possible alternative
                return SoundexCodes[soundex_code][0]
            else:
            ## If there are multiple similar words find a way to merge the relevent documents for each of the alternatives
                return SoundexCodes[soundex_code][0]
        elif soundex_code not in SoundexCodes:
                # If the soundex code differ by only the first alphabet                
                codes_alternatives = [codes for codes in SoundexCodes if soundex_code[1:] in codes]
                codes_final = []
                for setOfLetters in alphabet_sets:
                    codes_final.extend([codes for codes in codes_alternatives
                                   if soundex_code[0] in setOfLetters
                                   and codes[0] in setOfLetters])
                if len(codes_final) > 0:
                    return SoundexCodes[codes_final[0]][0]
    else:
        # Look for synonyms of the word that are present in the inverted index
        synonymsets = wn.synsets(termToModify)
        synonyms = set([word.name() for syn in synonymsets for word in syn.lemmas()])
        if len(synonyms) > 0:
                for altWord in synonyms:
                    if str(porter.stem(altWord)) in invIndex and '_' not in str(altWord):
                        return str(altWord)
    # If you cannot find phonetically similar words or synonyms return an empty string
    return ''


def decideAnswer(answerType, queryTerms, releventParas):
    if answerType in ['HUM','LOC','NUM']:
            releventSent = getSent(releventParas,queryTerms,answerType)
            releventSent.sort(reverse=True)
            probableSentences = releventSent[:10]
            if len(probableSentences) > 0:
                return probableSentences[0]
            elif len(releventParas) > 0:
                return releventParas[0]
            else:
                return "Sorry I can't seem to find the answer for your question"
    elif answerType in ['ABBR','DESC']:
        if len(releventParas) > 0:
            return releventParas[0]
        else:
            return "I do not have the answer to your question"
    elif answerType in ['ENTY']:
            releventSent = getSent(releventParas,queryTerms,answerType)
            releventSent.sort(reverse=True)
            probableSentences = releventSent[:10]
            if len(probableSentences) > 0:
                return probableSentences[0]
            elif len(releventParas) > 0:
                return releventParas[0]
            else:
                return "I cannot help you with this question"

while True:
    print()
    question,queryTerms,queryBiTokens = getQuestion()
    print('\nQuery terms: ', queryTerms)

    filename = 'SVM_MODEL/SVM_Model.sav'
    SVC_Classifier = joblib.load(filename)
    answerType = get_predictions(question)
    #answerType = 'DESC'
    print('\nAnswer type: ', answerType)

    invIndex,df,idf,numOfTextFiles,SoundexCodes = buildInvIndex()

    queStemTag = questionProc(question)
    queryTerms = [modifyTerm(queStemTag,word) if word not in invIndex else word for word in queryTerms]
    queryTerms = [word for word in queryTerms if len(word) > 0]   
    print('\nQuery terms: ', queryTerms)

    tfIdfQueryNorm = tfidfQuery(queryTerms,idf)
    releventDocs = extractDocs(queryTerms)
    print ("\nRelevent Documents:",releventDocs)

    ### Print top ten relevent paras
    paraScores = getReleventParas(queryTerms, queryBiTokens)
    paraScores.sort(reverse=True)
    releventParas = paraScores[:3]
  
    print("\nThe answer:")
    finalAnswer = decideAnswer(answerType, queryTerms, releventParas)
    print(finalAnswer)


