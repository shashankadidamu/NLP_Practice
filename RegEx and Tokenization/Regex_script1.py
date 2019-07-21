# Import the regex module
import re

#String
my_string = "Let's write RegEx!  Won't that be fun?  I sure think so.  Can you find 4 sentences?  Or perhaps, all 19 words?"

#Pattern to match sentence endings
sentence_endings = r"[.?!]"

# Split my_string on sentence endings and print the result
print(re.split(sentence_endings, my_string))

# Finding all capitalized words in my_string and printing the result
capitalized_words = r"[A-Z]\w+"
print(re.findall(capitalized_words, my_string))

# Split my_string on spaces and printing the result
spaces = r"\s+"
print(re.split(spaces, my_string))

# Finding all digits in my_string and printing the result
digits = r"\d+"
print(re.findall(digits, my_string))
