# QA-Engine
A Simple QA Engine using Information Retrieval. Text to be retrieved is stored in TextFiles directory

### QASystem.py : 
It is the main of the program. Contains the following components - 
* Answer Type Classification (answer_type_classification.py) : 
  * Uses SVM one versus all classification to classify questions based on the answer type expected.(Like "Where", "who", "what".) 
  * KeywordExtract.py :
    * Extracts keywords in a sentence using named entity recognition and chunking tool in NLTK.

* Information Extraction :
  * Implements a search engine based on an inverted index of the search space.
  * Also used the soundex algorithm for correcting names which may not be recognied correctly to those that are present in the vocabulary     in the KB

* Answer Extraction
  * Paragraph Ranking (ParagraphRanking.py)
    * The documents with the candidate responses are searched for relevant paragraphs. The paragraphs are ranked in order of relevance.         The highest scoring paragraph is taken as response for "why" (descriptive) type of questions.
  * Sentence Ranking (SentenceRanking.py)
    * The paragraphs are split into sentences which are again sorted in order of relevance. The most relevant answer is taken as response       to all other types of questions. 
  * Definition_extraction.py : 
    * Uses ML to separate definitions of a keyword from all other types of candidate responses. This is used only when the response             expected is a definition. For example, "what" type of questions
