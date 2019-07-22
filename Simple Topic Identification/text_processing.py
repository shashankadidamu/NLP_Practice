
from nltk.corpus import stopwords

text = """The cat is in the box. The cat likes the box. The box is over the cat."""

tokens = [w for w in word_tokenize(text.lower())
			if w.isalpha()]

no_stops = [t for t in tokens
			if t not in stopwords.words('english')]

Counter(no_stops).most_common(2)


####

# Import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer

#Retain alphabetic words only
alpha_only = [t for t in lower_tokens if t.isalpha()]

#remove all stop words
no_stops = [t for t in alpha_only if t not in english_stops]

#Instantiate WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

#lemmatize all tokens into a new list
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

#creating bag of words
bow = Counter(lemmatized)

print(bow.most_common(10))


### creating and querying a corpus with gensim

from gensim.corpora.dictionary import Dictionary 

dictionary = Dictionary(articles)

computer_id = dictionary.token2id.get("computer")

print(dictionary.get(computer_id))

corpus = [dictionary.doc2bow(article) for article in articles]

print(corpus[4][:10])


# Save the fifth document: doc
doc = corpus[4]

# Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

# Print the top 5 words of the document alongside the count
for word_id, word_count in bow_doc[:5]:
    print(dictionary.get(word_id), word_count)
    
# Create the defaultdict: total_word_count
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count

# Create a sorted list from the defaultdict: sorted_word_count 
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True) 

# Print the top 5 words across all documents alongside the count
for word_id, word_count in sorted_word_count[:5]:
    print(dictionary.get(word_id), word_count)


########## Gensim's tf-idf

 # Import TfidfModel
from gensim.models.tfidfmodel import TfidfModel

# Create a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus)

# Calculate the tfidf weights of doc: tfidf_weights
tfidf_weights = tfidf[doc]

# Print the first five weights
print(tfidf_weights[:5])

# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)



