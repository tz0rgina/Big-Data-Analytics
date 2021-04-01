import nltk

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup


class DataCleaner():

    def __init__(self, text):
        self.text = text
        tokenizer = ToktokTokenizer()
        self.words = tokenizer.tokenize(self.text)

    # Removing the html strips
    def strip_html(self):
        soup = BeautifulSoup(self.text, "html.parser")
        self.text = soup.get_text()

    # Removing the square brackets
    def remove_between_square_brackets(self):
        self.text = re.sub('\[[^]]*\]', '', self.text)

    # Define function for removing special characters
    def remove_special_characters(self, remove_digits=True):
        pattern = r'[^a-zA-z0-9\s]'
        self.text = re.sub(pattern, '', self.text)

    def remove_numbers(self):
        self.text = re.sub('[-+]?[0-9]+', '', self.text)

    # Stemming the text
    def simple_stemmer(self):
        ps = nltk.porter.PorterStemmer()
        self.words = [ps.stem(word) for word in self.text.split()]

    #Convert all characters to lowercase from list of tokenized words
    def to_lowercase(self):
        new_words = []
        for word in self.words:
            new_word = word.lower()
            new_words.append(new_word)
        self.words = new_words


    def remove_stopwords(self):
        # set stopwords to english
        stop = set(stopwords.words('english'))
        stop.update(["will", "s", "said", "may", "say", "need", "re", "put", "htc", "e",
                     "go", "yet", "u", "made", "much", "said", "one", "two", "make",
                     "say", "got", "says", "come", "used", "take" "according",
                     "still", "set", "m"])
        # print(stop)
        
        # Setting English stopwords
        new_words = []
        for word in self.words:
            if word not in stop:
                new_words.append(word)
        self.words = new_words

    #Remove punctuation from list of tokenized words
    def remove_punctuation(self):
        new_words = []
        for word in self.words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        self.words = new_words
        return self

    def clean(self):

        self.strip_html()
        self.remove_between_square_brackets()
        self.remove_special_characters()
        self.remove_numbers()
        self.to_lowercase()
        self.simple_stemmer()
        self.remove_stopwords()
        self.remove_punctuation()

        return ' '.join(self.words)


def clean_data(data):
    for i, text in enumerate(data):
        txt = DataCleaner(text)
        data[i] = txt.clean()
    return data

# DEBUGGING
# porter = nltk.porter.PorterStemmer()

# # proide a word to be stemmed
# print("Porter Stemmer")
# print(porter.stem("cats"))
# txt = DataCleaner("Cats sat hers on says made the tRoubled, 8, 9 http://www.cars.html !")
# print(txt.clean())
# print(txt.words[0])
#
# import pandas as pd
# import numpy as np
#
# # TEST DATA
# df_test = pd.read_csv("./test.csv")
# # df_test=pd.read_csv("corpusTest.csv")
# df_test.head()
# df_test.loc[0]['Content']
#
# X_test = df_test.iloc[:, 1].values
# test = [X_test[5], X_test[5089], X_test[5087], X_test[1]]
# print(test)
# X_test = np.array(clean_data(test))
# print(test)

