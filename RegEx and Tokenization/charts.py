#charting practice

from matplotlib import pyplot as plt
from nltk.tokenize import regexp_tokenize

holy_grail = "SCENE 1: [wind] [clop clop clop] \nKING ARTHUR: Whoa there!  [clop clop clop] \nSOLDIER #1: Halt!  Who goes there?\nARTHUR: It is I, Arthur, son of Uther Pendragon, from the castle of Camelot.  King of the Britons, defeator of the Saxons, sovereign of all England!\nSOLDIER #1: Pull the other one!\nARTHUR: I am, ...  and this is my trusty servant Patsy.  We have ridden the length and breadth of the land in search of knights who will join me in my court at Camelot.  I must speak with your lord and master.\nSOLDIER #1: What?"


#split the script into lines 
lines = holy_grail.split('\n')

#replace all scripts lines for speaker
pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
lines = [re.sub(pattern, '',l) for l in lines]

#tokenize each line 
tokenized_lines = [regexp_tokenize(s, "\w+") for s in lines]

#make a frequency list of lengths
line_num_words = [len(t_line) for t_line in tokenized_lines]

#plot histogram of line lengths
plt.hist(line_num_words)

#show the plot
plt.show()

