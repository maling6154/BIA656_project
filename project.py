
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.collocations import *
import nltk
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import wordnet
import enchant
import sklearn.metrics as metrics
from nltk.metrics.distance import jaccard_distance,edit_distance
from nltk.wsd import lesk

# import data
stemmer = SnowballStemmer('english')
cachedStopWords = stopwords.words("english")
df_train = pd.read_csv('Data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('Data//test.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('Data//attributes.csv')
df_product_descrition = pd.read_csv('Data//product_descriptions.csv', encoding="ISO-8859-1")
df_all_desc = pd.read_csv('Data//aggregated_attributes.csv', encoding="ISO-8859-1")

# change column types
df_train=df_train.convert_objects(convert_numeric=True)
df_all_desc=df_all_desc.convert_objects(convert_numeric=True)


# merge data
df_all = df_train.merge(df_all_desc,left_on='product_uid',right_on='product_uid', how='outer')

del df_all['id']

# tokenize text
def tokenizer(text):
    words = nltk.word_tokenize(my_text) 
    word=[i.lower() for i in words]
    return word
    
# find entitites in text
def entitiy_rec(text):
    ner = StanfordNERTagger("ner/classifiers/english.all.3class.distsim.crf.ser.gz", "ner/stanford-ner.jar")
    tags=ner.tag(text)
    return tags

# stem a word list
def stemmer(words):
    stem=[stemmer.stem(word) for word in words]
    return stem

# find n grams
def ngrams(my_text,n):  
    my_text=my_text.decode('ascii','ignore')
    words = tokenizer(my_text)
    words=[i for i in words if   i.isalpha()]
    my_ngrams = nltk.ngrams(words,n)
    grams= sorted(i for i in my_ngrams)
    return grams

# remove stop words from a wordlist
def remove_stopwords(words):
    return [word for word in words if word not in cachedStopWords]
    
# get synonyms of a wordlist
def get_synonyms(words):
    syn= [wordnet.synsets(word) for word in words]
    return syn

  # get word definition  
def get_definition(word):
    syns = wordnet.synsets(word)
    for s in syns:
        return s.definition()

# check spelling for a word
def spell_checker(word):
    d = enchant.Dict("en_US")
    return d.check(word)
    
# suggest correct spelling for a word    
def correct_spell_suggest(word):  
    d = enchant.Dict("en_US")
    return d.suggest(word)

# find lemas
def find_lemmas(word):
    syn=wordnet.synsets(word)
    return syn[0].lemmas()


# find jacquard sim
def jacquard_sim(set1,set2):
    sim=jaccard_distance(set1, set2)#, normalize=True)
    return sim

# find edit distance     
def edit_distance_sim(word1,word2):
    dist=edit_distance(word1,word2)
    return dist    


#http://www.nltk.org/howto/wordnet.html
# similarity based on path
def jcn_sim(word1,word2):
    w1=wordnet.synset(wordnet.synsets(word1)[0].name())
    w2=wordnet.synset(wordnet.synsets(word2)[0].name())
    sim=w1.jcn_similarity(w2,'brown_ic')
    return sim

# find forms of a word    
def word_form(word, pos):
    return wordnet.morphy(word,pos)
    

#lesk : Given an ambiguous word and the context in which the word occurs, 
#Lesk returns a Synset with the highest number of overlapping words between 
# the context sentence and different definitions from each Synset.

# meaning by context
def lesk_word(text,word,pos='a'):
    text=text.split()
    return lesk(text, word, pos=pos)
  
 


