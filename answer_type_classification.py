import os
import nltk

from string import punctuation
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion

def get_labels(raw_data):
    coarse_label = []
    fine_label = []
    question = []

    for line in raw_data:
        split_line = line.split(":")
        coarse_l = split_line[0]
        fine_l = split_line[1].split(' ')[0]
        
        coarse_label.append(coarse_l)
        fine_label.append(fine_l)
        
        mod_line = line.replace(coarse_l,'')
        mod_line = mod_line.replace(':','')
        mod_line = mod_line.replace(fine_l,'')
        question.append(mod_line)
    return(question,coarse_label,fine_label)        


    

##def get_tags(question):
##    que =  question.lower()
##    #remove question marks
##    que1 = que.replace('?','')
##    token = nltk.word_tokenize(que1)    
##    tagged = nltk.pos_tag(token)
##    tag_dict = {}
##    for (word,tag) in tagged:
##        tag_dict[word]=tag
##    return tag_dict    
        

def get_np(question):
    que = question.lower()
    que1=que.replace('?','')
    token = nltk.word_tokenize(que1)
    tagged = nltk.pos_tag(token)
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(grammar)
    np_chunk = cp.parse(tagged)
    np = []
    for subtree in np_chunk.subtrees():
        if subtree.label()=='NP':
            strings = [' '.join(word for (word,tag) in list(subtree))]
            np.extend(strings)           
    return np   

def get_tags(question):
    que =  question.lower()
    #remove question marks
    que1 = que.replace('?','')
    token = nltk.word_tokenize(que1)    
    tagged = nltk.pos_tag(token)
    return tagged

def combined_features(question):
    tag_dict = {}
    numOfNouns = 0
    taggedWords = get_tags(question)
    listOfNPs = get_np(question)
    for word,tag in taggedWords:
        tag_dict[word] = tag
##    if len(listOfNPs) > 0:
##        for nounPhrase in listOfNPs:        
##            tag_dict[nounPhrase] = 'NounPhrase'
##    for word,tag in taggedWords:
##        if tag == 'WDT|WP|WP$|WRB':
##            tag_dict[word] = 'qWords'
##    for word,tag in taggedWords:
##        if tag == 'NN|NNP|NNS|NNPS':
##            numOfNouns += 1
##    tag_dict['numOfNouns'] = numOfNouns
    return tag_dict

#combined_features = FeatureUnion([("Tags",get_tags(question)),("NPs",get_np(question))])

file_path = os.path.join('C:/Users/DComp2/Desktop/python learn/get_data/training_data','train_5500.txt')
text_file = open(file_path,"rU")
raw_data = text_file.readlines()
text_file.close()

question,coarse_label,fine_label = get_labels(raw_data)
labeled_data = zip(question,coarse_label)


print "Extracting Featuresets now..."
featuresets = [(combined_features(que),cl) for (que,cl) in labeled_data]
print "Featuresets ready"
print



### ----------------- Train the data for coarse labels --------------- ###
print"Starting train..."
from sklearn.multiclass import OneVsRestClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC
SVC_classifier = SklearnClassifier(OneVsRestClassifier(LinearSVC(C=100)))
SVC_classifier.train(featuresets)
print "training complete!"
### -------------------- ********** ------------------- ###



### ------------ Get the data based on fine labels -------------- ###
##
##zipped_data = zip(question,coarse_label,fine_label)
##fine_label_HUM = [(que,fl) for (que,cl,fl) in zipped_data if cl == 'HUM']
##fine_label_NUM = [(que,fl) for (que,cl,fl) in zipped_data if cl == 'NUM']
##fine_label_LOC = [(que,fl) for (que,cl,fl) in zipped_data if cl == 'LOC']
##fine_label_ENTY = [(que,fl) for (que,cl,fl) in zipped_data if cl == 'ENTY']
##fine_label_DESC = [(que,fl) for (que,cl,fl) in zipped_data if cl == 'DESC']
##fine_label_ABBR = [(que,fl) for (que,cl,fl) in zipped_data if cl == 'ABBR']
##
##
##print
##print
##print "Extracting Featuresets for fine labels now..."
##featuresets_HUM = [(get_tags(que),fl) for (que,fl) in fine_label_HUM]
##featuresets_NUM = [(get_tags(que),fl) for (que,fl) in fine_label_NUM]
##featuresets_LOC = [(get_tags(que),fl) for (que,fl) in fine_label_LOC]
##featuresets_ENTY = [(get_tags(que),fl) for (que,fl) in fine_label_ENTY]
##featuresets_DESC = [(get_tags(que),fl) for (que,fl) in fine_label_DESC]
##featuresets_ABBR = [(get_tags(que),fl) for (que,fl) in fine_label_ABBR]
##
##print "Featuresets ready"
##
##print"Start train for fine labels..."
##SVC_classifier_HUM = SklearnClassifier(OneVsRestClassifier(LinearSVC(C=100)))
##SVC_classifier_HUM.train(featuresets_HUM)
##
##SVC_classifier_NUM = SklearnClassifier(OneVsRestClassifier(LinearSVC(C=100)))
##SVC_classifier_NUM.train(featuresets_NUM)
##
##SVC_classifier_LOC = SklearnClassifier(OneVsRestClassifier(LinearSVC(C=15)))
##SVC_classifier_LOC.train(featuresets_LOC)
##
##SVC_classifier_DESC = SklearnClassifier(OneVsRestClassifier(LinearSVC(C=100)))
##SVC_classifier_DESC.train(featuresets_DESC)
##
##SVC_classifier_ENTY = SklearnClassifier(OneVsRestClassifier(LinearSVC()))
##SVC_classifier_ENTY.train(featuresets_ENTY)
##
##SVC_classifier_ABBR = SklearnClassifier(OneVsRestClassifier(LinearSVC()))
##SVC_classifier_ABBR.train(featuresets_ABBR)
##print "training complete!"
print
### -------------------- ********** ------------------- ###




### ------------- Test the data for coarse label ----------------------- ###
# This part of the code can be commented out after sufficient
# accuracy is reached

file_path_test = os.path.join('C:/Users/DComp2/Desktop/python learn/get_data/training_data','test_data.txt')
text_file_test = open(file_path_test,"rU")
raw_data_test = text_file_test.readlines()
text_file_test.close()

question_test,coarse_label_test,fine_label_test = get_labels(raw_data_test)
labeled_data_test = zip(question_test,coarse_label_test)


test_data = question_test
print("test_data_length",len(test_data))
predict_labels = []
for que in test_data:
    test_features =combined_features(que)
    predicted = SVC_classifier.classify(test_features)
    predict_labels.append(predicted)

import numpy as np
true_label = np.array(coarse_label_test)
predicted_label = np.array(predict_labels)
from sklearn.metrics import precision_recall_fscore_support
precision,recall,fscore,support = precision_recall_fscore_support(true_label, predicted_label)
from collections import Counter
counts = Counter(coarse_label_test)
all_set=zip(precision,recall,fscore,support)
print
for precision,recall,fscore,support in all_set:
    print 'Precision:',round(precision,2)
    print 'Recall:', round(recall,2)
    print 'F1:',round(fscore,2)
    print 'NumOfCases:',support
    print
print	
test_features = [(combined_features(que),cl) for (que,cl) in labeled_data_test]
print("test_features_length", len(test_features))
print("Accuracy:",nltk.classify.accuracy(SVC_classifier, test_features))



### ------------------ *************** -------------------------- ###

### ------------------ Test the data for fine label -------------------------- ###

##def get_predictions(question):
##    test_features = get_tags(question)
##    coarse_predicted = SVC_classifier.classify(test_features)
##    fine_predicted = []
##    if coarse_predicted == 'NUM':
##        fine_predicted = SVC_classifier_NUM.classify(test_features)
##    elif coarse_predicted == 'HUM':
##        fine_predicted = SVC_classifier_HUM.classify(test_features)
##    elif coarse_predicted == 'LOC':
##        fine_predicted = SVC_classifier_LOC.classify(test_features)
##    elif coarse_predicted == 'ENTY':
##        fine_predicted = SVC_classifier_ENTY.classify(test_features)
##    elif coarse_predicted == 'DESC':
##        fine_predicted = SVC_classifier_DESC.classify(test_features)
##    elif coarse_predicted == 'ABBR':
##        fine_predicted = SVC_classifier_ABBR.classify(test_features)
##    return(coarse_predicted, fine_predicted)
##    

##output_list = []    
##for que in test_data:
##    output_list.append(get_predictions(que))
##    
##



print
print("DONE!")
