import math
import string
from nltk.stem import  PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords

line=''
doc=[]
term=[]
df_dict={}
doc_non_repeat=[]
corpus=[]
doc_num_tf=[]
non_repeat_term_collect=[]
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
        words = [word for word in words if len(word) >=3]
        non_repeat_term_collect.append((set(words)))
        doc_non_repeat=list(set(words))   #每個文件不重複的term的list
        for i in range(len(doc_non_repeat)):
            if doc_non_repeat[i] not in df_dict:
                df_dict[doc_non_repeat[i]]=1
            else:
                df_dict[doc_non_repeat[i]]+=1
        tf_dict={}
#         tf_dict.append(doc_num)={}
        for i in range(len(words)):
            if words[i] not in tf_dict:
                tf_dict[words[i]]=1
            else:
                tf_dict[words[i]]+=1
        doc_num_tf.append(tf_dict)   #每個文件的term的數量統計
#         doc_num+=1
        term.append(words) #為各個文件內的文字存成list 
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
fp = open("dictionary.txt", "a+", encoding='utf-8-sig')
for i in range(len(union)):
    #df_dict[word] 的value是term 的df值
        print(i+1,union[i],"df",df_dict[union[i]],file=fp)
        dict[union[i]]=str(i+1) #取得這個term的編號
fp.close()



dict_idf={}  #term 的idf值存成dict

for word in dict:
    dict_idf[word]=math.log10(1095/df_dict[word])


doc_num_tf_list=[]  #存放每個文件裡面的term字典(字典為term和出現的個數)
#tf_idf=[]
for i in range(len(doc_num_tf)):
    doc_num_tf_list.append(list(doc_num_tf[i]))
    doc_num_tf_list[i].sort()

tf_idf_0={}
tf_idf_1={}

for i in range(1095):
    fp = open('tfidf2\\'+str(i+1)+".txt", "a+", encoding='utf-8')
    #print ("-------文件",i+1,"號出現的 term 和其 tf-idf unit vector------")
    print ("-------文件",i+1,"號出現的 term 和其 tf-idf unit vector------",file=fp) 
    tf_idf={}
    count=0
    for word in doc_num_tf[i]:
        #doc_num_tf[i][doc_num_tf_list[i][j]] 為tf
        tf_idf[word]=doc_num_tf[i][word]*(dict_idf[word])  #取出dic 的term的tf數值
        count=count+(tf_idf[word])**2
    count=math.sqrt(count)
    tf_idf_list=[]
    for word in tf_idf:
        tf_idf[word]=tf_idf[word]/count #normalize的數值
        if i==0:
            tf_idf_0[word]=tf_idf[word]
        if i==1:
            tf_idf_1[word]=tf_idf[word]      
        tf_idf_list.append(word)
    tf_idf_list.sort()
    for i in range(len(tf_idf_list)):
        dict[tf_idf_list[i]],tf_idf_list[i],tf_idf[tf_idf_list[i]]
        print ("t_index",dict[tf_idf_list[i]],tf_idf_list[i],tf_idf[tf_idf_list[i]],file=fp)
    fp.close()


cosine_sim=0
for word in tf_idf_0:  #第一號文件出現的Term
    if word in tf_idf_1:  #如果第二號文件也有同一個term 兩者的tf-idf-unit vector 乘起來相加
        cosine_sim=cosine_sim+(tf_idf_0[word]*tf_idf_1[word])
print(cosine_sim)


target_list=[]   
union2=[]
#         line=tfile.readline() # read the content of file and store in "line"
#         print(line)
for line in open("classification.txt",'rt', encoding='utf-8-sig'):
    target_list.append(line.split(' '))
    #將training 的classfication文件編號提出來

xx=set(target_list[0])
for i in range(len(target_list)):
#     print(target_list[i][-1])
    del target_list[i][0]
    del target_list[i][-1]
    xx = (xx | set(target_list[i]))
xx.remove('\n')  

xx=list(xx)
xx.sort()
xx = list(map(int, xx))
dict_xx={}   #training data doc id
for i in range(len(xx)):
    dict_xx[xx[i]] = xx[i]  
#將training num轉成字典

testing_doc_num={}
for i in range(1,1096):
    if i not in dict_xx:
            testing_doc_num[i]=i

zz=[]
for i in dict_xx:  #to loop all the from the training data dictionary
    zz=zz+term[dict_xx[i]-1]    #zz為all term，包括重複出現的
#將training data doc 裡面出現過的term取出(包含重複的term)

import nltk
zz=nltk.FreqDist(zz)
word_features_dict={}
# print(zz.most_common(500))   #計算出最多出現的前500 個term
word_features = list(zz.keys())[:500]
word_features.sort()
for i in range(len(word_features)):
    word_features_dict[word_features[i]]=int(i)  #word_feature_dict value為前500最多出現的Term

#將training 和testing 的data doc corpus 做處理:只留下前500個出現的term，其餘省略
term_word_feature_all=[]
for i in range(len(term)):
    term_word_feature=[]
    for j in range(len(term[i])):
        if term[i][j] in word_features_dict:  #保存前500多個term，其餘省略掉
            term_word_feature.append(term[i][j])
    term_word_feature_all.append(term_word_feature)  
    # term_word_feature_all為一個陣列，儲存每一個文件編號裡面包含的前500term

class_all_term_dict={}
for i in range(len(target_list)):
    class_all_term=[]
    for j in range(len(target_list[i])):
        #target_list[0][0]為11 [0][1]為19
        class_all_term.append(term_word_feature_all[int(target_list[i][j])-1])
        #每一個class 包含的文件 裡面的term總數相加
    for x in range(len(class_all_term)-1):    
        intersection=(class_all_term[x]+class_all_term[x+1])
    class_all_term_dict[i]=intersection
    #一個class 出現過的term存成一個陣列，存放到字典內
#         for k in range(len(term_word_feature_all[int(target_list[i][j])-1])):
#         #取出第幾號文件所存放出現的500term，target_list[i][j]-1 為第幾號文件
#             term_word_feature_all[int(target_list[i][j])-1][k] 



class_word_probability=[]  #存放每一個class 的word_probability={}字典
from collections import Counter
for i in range(len(class_all_term_dict)):
    word_probability={}  #輸入key term就可以得到term 的 prior probability (value)
    x=Counter(class_all_term_dict[i])
    for word in x:
        word_probability[word]=(x[word]+1)/(len(class_all_term_dict[i])+500)
        #Calculate Likelihood
    class_word_probability.append(word_probability)


testing_doc_class={}
#testing_doc_num 為還未歸類的文件編號，是字典
fp = open("output.txt", "a+", encoding='utf-8')
print ("Id","Value",file=fp) 
for i in testing_doc_num:
    #print (type(i))
    max_pp=0
    pp=0
    class_final=0
    for k in range(13):
        term_in_class=[]
        for j in range(len(term_word_feature_all[i-1])):
            if term_word_feature_all[i-1][j] in class_word_probability[k]:
        #如果這個term有在第k個class的 all term字典裡面
                term_in_class.append(term_word_feature_all[i-1][j])
        #將他存放到term_in_class這個list裡面
        for x in range(len(term_in_class)-1):  #x為第k個class 有出現的term
            pp=(class_word_probability[k][term_in_class[0]])
            testing_term=1/13  #每一個class 的機率為1/13
            pp=pp*(class_word_probability[k][term_in_class[x+1]])  #第幾號文本的all term(500個term)
            pp=pp*testing_term
            if pp> max_pp:
                max_pp=pp
                class_final=k
    testing_doc_class[i]=class_final+1 #testing_doc_class字典紀錄第i號文件屬於第k個class
    
    print (i,testing_doc_class[i],file=fp)
fp.close()

