
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.collocations import *
import nltk, string
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import wordnet
import enchant
import sklearn.metrics as metrics
from nltk.metrics.distance import jaccard_distance,edit_distance
from nltk.wsd import lesk
from sklearn.feature_extraction.text import TfidfVectorizer
# import data
stemmer = SnowballStemmer('english')
cachedStopWords = stopwords.words("english")



# remove stop words from a wordlist
def remove_stopwords(words,cachedStopWords):
    return [word for word in words if word not in cachedStopWords]
    
    
#-------------------------------------------    
# get synonyms of a wordlist and add to document. adjacent to original word
def get_synonyms(document,n):

    words = tokenizer(document)

    words=remove_stopwords(words,cachedStopWords)
    morphy=[]
    for i in words:
        print 'for words ',i
        word_forms=[i]
        syns=wordnet.synsets(i)
        try:
            wordset=list(set([str(k.name().split('.')[0]) for k in syns][:n]))
            print 'the syns are ',wordset
        except:
            wordset=[]
            print 'error'
        wordset=[j for j in wordset if j not in i]    
        word_forms.extend(wordset)
        print word_forms
        morphy.extend(word_forms)
    
    return ' '.join(morphy)

# tf-idf-------------------------------------------
    
stemmer = nltk.stem.porter.PorterStemmer()
#remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower()))#.translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]
    
#---------------------------------------------
    


# find jacquard sim
def jacquard_sim(text1,text2):
    set1=set(tokenizer(text1))
    set2=set(tokenizer(text2))
    sim=jaccard_distance(set1, set2)#, normalize=True)
    return sim

# find edit distance     
def edit_distance_sim(word1,word2):
    dist=edit_distance(word1,word2)
    return dist  

import difflib as dl

# some other similarity
def diff_sim(text1,text2):
    sim = dl.get_close_matches    
    s = 0
    wa = text1.split()
    wb = text2.split()
    
    for i in wa:
        if sim(i, wb):
            s += 1
    return float(s) / float(len(wa))


# get word definition
def get_definition(word):
    syns = wordnet.synsets(word)
    for s in syns:
        return s.definition()
        

tags=['NN','JJ']       
text="And now for something completely different"


# replace word in query with its definition
def replace_word_with_def(text,tags):
    it=nltk.pos_tag(nltk.word_tokenize(text))
    sent=[]
    for i in it:
        word=i[0]
        pos=i[1]
        define=[]
        if pos in tags:
            define=str(get_definition(word))
            if str(define)=='None':
                sent.append(word)

                continue;
            sent.append(define)  
            continue;
        else:
            sent.append(word)
    
    sentence=' '.join(sent)
    return sentence

replace_word_with_def(text,tags)


# import corpuses for similarity measures
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

#http://www.nltk.org/howto/wordnet.html
# similarity based on path
def jcn_sim(word1,word2):

    word1=str(word1)
    word2=str(word2)
    try:    
        w1=wordnet.synset(wordnet.synsets(word1)[0].name())
    except:
        return np.nan
    try:    
        w2=wordnet.synset(wordnet.synsets(word2)[0].name())
    except:
        return np.nan
    try:        
        sim1=w1.jcn_similarity(w2,semcor_ic)
        sim2=w1.jcn_similarity(w2,brown_ic)
        sim3=w1.lin_similarity(w2,brown_ic) 
        sim4=w1.lin_similarity(w2,semcor_ic) 
    except:
        return np.nan
    score= (sim1+sim2+sim3+sim4)/4
    if score>1:
        return 1
    else:
        return score


#-----------------------------TEST CODE----------
text1='where are the great desk tables'
text2='where are my great cars'
#-----------------------------TEST CODE----------


#my own version of similarity
def my_sim(text1,text2):
    set1=set(tokenizer(text1))
    set2=set(tokenizer(text2))
    
    d={}
    scores=[]
    for i in set1:
        one_word_sim=[]
        for j in set2:
            sim=jcn_sim(i,j)
            one_word_sim.append(sim)
        d[i]=one_word_sim 
            
    for i,j in d.iteritems():
        length=len(j)        
        valid_entries=len([o for o in j if o>=0])
        if valid_entries==0:
            continue
        score=sum([k for k in j if k>=0])/length
        
        scores.append(score)
    return sum(scores)/len(d)

