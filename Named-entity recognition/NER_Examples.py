
#Named Entity Recognition 

#Import Libraries
import nltk

#sample sentence
sentence = '''In new york, I like to ride the Metro to visit Taj and some restaurants rated well by John'''

#tokenize sentence
tokenized_sent = nltk.word_tokenize(sentence)

#parts of speech tagging, pos_tag adds tags for proper nouns, pronouns, adjectives, verbs"""
tagged_sent = nltk.pos_tag(tokenized_sent)

#tree structure 
print(nltk.ne_chunk(tagged_sent))


#####Ex:2

import nltk

#Store any text or article using the below mentioned article variable
article = ''

#tokenize article into sentences
sentences = nltk.sent_tokenize(article)

#tokenize each sentence into words
token_sentences = [nltk.word_tokenize(sent) for sent in sentences]

#tag each tokenized sentence into parts of speech 
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences]

#create named entity chunks
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

#test for stems of tree with 'NE' tags
for sent in chunked_sentences:
	for chunk in sent:
		if hasattr(chunk,"label") and chunk.label()=="NE":
			print(chunk)


#
#create defaultdict
ner_categories = defaultdict(int)

#create nested for loop
for sent in chunked_sentences:
	for chunk in sent:
		if hasattr(chunk,'label'):
			ner_categories[chunk.label()] += 1

#create a list from dictionary keys for chart labels
labels = list(ner_categories.keys())

#create list of values
values = [ner.ner_categories.get(l) for l in labels]

plt.pie(values, labels = labels, autopct='%1.1f%%', startangle=140)

plt.show()





