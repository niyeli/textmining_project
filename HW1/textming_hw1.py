
# coding: utf-8

#import nltk
#nltk.download('punkt')
from nltk.stem import  PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
# stop_words = stopwords.words('english')
# print(stop_words)
filename = 'HW1.txt'
file = open(filename, 'rt', encoding='utf-8-sig')    #r 讀取， t 代表文字
text = file.read()
file.close()
# split into words
tokens = word_tokenize(text,language='english')
# convert to lower case
tokens = [w.lower() for w in tokens]
# stemming of words
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
# filter out stop words
stop_words = set(stopwords.words('english'))
words = [w for w in stemmed if not w in stop_words]
print(words)

# 開啟檔案並將結果存入新檔案
fp = open("result.txt", "a+", encoding='utf-8')
for x in words:
    fp.write(x)
    fp.write('\n')
# 關閉檔案
fp.close()





