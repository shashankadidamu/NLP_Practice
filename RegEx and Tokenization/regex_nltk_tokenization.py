#Regex with NLTK Tokenization

 #Import necessary modules
 from nltk.tokenize import regexp_tokenize
 from nltk.tokenize import TweetTokenizer

 tweets = ['This is an #nlp exercise! #python',
 '#NLP is super fun! <3 #learning',
 'Thanks @nlp :) #nlp #python']

 #define a regex pattern to find hashtags 
 pattern1 = r"#\w+"

 #use the pattern on the first tweet in tweets list
 regexp_tokenize(tweets[0], pattern1)

 #write a pattern that matches both mentions and hashtags
 pattern2 = r"([@#]\w+)"

 #use pattern on last tweet in the tweets list
 regexp_tokenize(tweets[-1], pattern2)

 #use Tweettokenizer to tokenize all tweets into one list
 tknzr = TweetTokenizer()
 all_tokens = [tknzr.tokenize(t) for t in tweets]
 print(all_tokens)