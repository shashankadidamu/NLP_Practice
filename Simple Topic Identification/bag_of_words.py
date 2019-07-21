#building a counter with bag of words

#import counter, word_tokenizer
from nltk.tokenize import word_tokenize
from collections import Counter

#tokenize the article: 
tokens = word_tokenize(article)

#convert the tokens into lowercase:
lower_tokens = [t.lower() for t in tokens]

#create a counter with lowercase tokens
bow_simple = Counter(lower_tokens)

#print 10 most common tokens
print(bow_simple.most_common(10))
