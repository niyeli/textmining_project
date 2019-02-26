import math
import string
from nltk.stem import  PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from stop_words import get_stop_words
import glob

line=''
doc=[]
df_dict={}
doc_non_repeat=[]

#corpus=[]
doc_num_tf=[]
non_repeat_term_collect=[]
#flist=glob.glob(r'IRTM\*.txt') 
#get all the files from the d`#open each file >> tokenize the content >> and store it in a set
flist = []
for i in range(0,1095):
    flist.append('IRTM\\' + str(i+1) + '.txt')
for fname in flist:
        tfile=open(fname,'rt', encoding='utf-8-sig')
        line=tfile.read() # read the content of file and store in "line"
        # split into words
        tokens = word_tokenize(line,language='english')
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        # stemming of words
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in tokens]
        stop_words = set(stopwords.words('english'))
        words = [w for w in stemmed if w not in stop_words]   #words是list
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in words]
        #remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]  #此文件的term存成list
        #word<3 直接去掉
        words = [word for word in words if len(word) >=3]
        non_repeat_term_collect.append((set(words)))
        doc_non_repeat=list(set(words))   #每個文件不重複的term的list
        for i in range(len(doc_non_repeat)):
            if doc_non_repeat[i] not in df_dict:
                #如果字典還沒有就加進字典
                df_dict[doc_non_repeat[i]]=1
            else:
                #字典裡面已經有了就加一
                df_dict[doc_non_repeat[i]]+=1
        tf_dict={}
        for i in range(len(words)):
            if words[i] not in tf_dict:
                tf_dict[words[i]]=1
            else:
                tf_dict[words[i]]+=1
        doc_num_tf.append(tf_dict)   #每個文件的term的數量統計        
        str_words=" ".join(words)   #將list轉string，以空格連接
        #corpus.append(str_words) #一個檔案的所有term存成一個string 放到 corpus這個list裡面


yy=non_repeat_term_collect[0]
for x in range(0,len(non_repeat_term_collect)):
    yy = (yy | non_repeat_term_collect[x])  #將所有文件中的term以不重複的方式存成一個list
union=yy

set(union)
union=list(union)
union.sort()

dict={}  #紀錄1095個文件出現過的term的t_index編號
fp = open("dictionary.txt", "a+", encoding='utf-8')
for i in range(len(union)):
    #df_dict[word] 的value是term 的df值
        print(i+1,union[i],"df",df_dict[union[i]],file=fp)
        dict[union[i]]=str(i+1) #取得這個term的編號
fp.close()


dict_idf={}  #term 的idf值存成dict
for word in dict:
    dict_idf[word]=math.log10(1095/df_dict[word])


doc_num_tf_list=[]  #存放每個文件裡面的term字典(字典的value為term和出現的個數) 被用來sort
for i in range(len(doc_num_tf)): 
    #doc_num_tf[i]為一個dict
    doc_num_tf_list.append(list(doc_num_tf[i]))
    doc_num_tf_list[i].sort()


tf_idf_0={}
tf_idf_1={}

for i in range(1095):
    fp = open('tfidf2\\'+str(i+1)+".txt", "a+", encoding='utf-8')
    print ("-------文件",i+1,"號出現的 term 和其 tf-idf unit vector------",file=fp) 
    tf_idf={}  #tf_idf為計算過後的字典
    count=0
    #doc_num_tf[i]為一個dict
    for word in doc_num_tf[i]:
         #取出dict 的term的tf數值
        tf_idf[word]=doc_num_tf[i][word]*(dict_idf[word])  
        count=count+((tf_idf[word])**2)
    count=math.sqrt(count)
    tf_idf_list=[]
    #tf_idf為字典
    for word in tf_idf:
        tf_idf[word]=tf_idf[word]/count #normalize後的數值
        if i==0:
            tf_idf_0[word]=tf_idf[word]
        if i==1:
            tf_idf_1[word]=tf_idf[word]
        tf_idf_list.append(word)  #sort過後的term
    tf_idf_list.sort()
    for i in range(len(tf_idf_list)):
        #dict[tf_idf_list[i]] 為term的編號
        #tf_idf_list[i]為排序過後的term
        #tf_idf[tf_idf_list[i]]為此term的if idf值
        print ("t_index",dict[tf_idf_list[i]],tf_idf_list[i],tf_idf[tf_idf_list[i]],file=fp)
    fp.close()


cosine_sim=0
for word in tf_idf_0:  #第一號文件出現的Term
    if word in tf_idf_1:  #如果第二號文件也有同一個term 兩者的tf-idf-unit vector 乘起來相加
        cosine_sim=cosine_sim+(tf_idf_0[word]*tf_idf_1[word])
print("一號文件和二號文件的相似度為",cosine_sim)

