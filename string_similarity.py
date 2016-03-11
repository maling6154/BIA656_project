# 
#   TUTORIAL ABOUT MATCHING STRINGS AND FINDING SIMILARITY SCORE
#


#
import Levenshtein as l


x='arslan'
y='arsalan'

# function returns distance as per how many alphabets need to be changed
# in x to become y
d=l.distance(x,y)

#################################################


from difflib import SequenceMatcher

m = SequenceMatcher(None, "NEWs YOK METS", "NEW YORK MEATS")
n=m.ratio()


#################################################

#another way to do it
from fuzzywuzzy import fuzz, process

n1=fuzz.ratio( "NEWs YOK METS", "NEW YORK MEATS")

n2=fuzz.ratio("YANKEES", "NEW YORK YANKEES") #score =61
n3=fuzz.ratio("NEW YORK METS", "NEW YORK YANKEES")


#################################################
# partial ratio is helpful in matching substrings
#If the shorter string is length m, and the longer 
#string is length n, we’re basically interested in 
#the score of the best matching length-m substring.

n4=fuzz.partial_ratio("YANKEES", "NEW YORK YANKEES") #score =100
n5=fuzz.partial_ratio("NEW YORK METS", "NEW YORK YANKEES")

#################################################

#out of order examples

#The token sort approach involves tokenizing the string in question,
#sorting the tokens alphabetically, and then joining them back into a string. 
n6=fuzz.partial_ratio(" YORK METS NEW", "NEW YORK YANKEES")
n7=fuzz.token_sort_ratio(" YORK METS NEW", "NEW YORK YANKEES")

#################################################
#sets
n8=fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
#################################################

# Extract similar words, strings
choices = ["arslan Falcons", "ARSLAN Jets", "arsalan Giants", "arsLan Cowboys","ARS"]
n9=process.extract("arslan", choices, limit=6)

n10=process.extractOne("arslan", choices)