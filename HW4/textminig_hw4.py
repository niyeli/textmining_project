#!/usr/bin/env python
# coding: utf-8
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

# len(term)
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
tf_idf_all={}  #tfidf_all是一個字典紀錄1095個文件裡面個別word的idf

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
    tf_idf_word={}
    for word in tf_idf:
        tf_idf[word]=tf_idf[word]/count #normalize的數值    
        tf_idf_word[word]=tf_idf[word]
        tf_idf_list.append(word)
    for x in range(0,1095):
        tf_idf_all[i]=tf_idf_word  #tf_idf_all 為一個字典，存放1095個文件
        #tf_idf_all[0]代表文件一號所有word的unit vector
        #tf_idf_all[0][word]就會列出此word 的數值    
#for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重    
    tf_idf_list.sort()
    for i in range(len(tf_idf_list)):
        dict[tf_idf_list[i]],tf_idf_list[i],tf_idf[tf_idf_list[i]]
        print ("t_index",dict[tf_idf_list[i]],tf_idf_list[i],tf_idf[tf_idf_list[i]],file=fp)
    fp.close()

def cosine_sim_num(doc1,doc2):
    cosine_sim=0
    for word in tf_idf_all[doc1-1]:  #第一號文件出現的Term
        if word in tf_idf_all[doc2-1]:  #如果第二號文件也有同一個term 兩者的tf-idf-unit vector 乘起來相加
            cosine_sim=cosine_sim+(tf_idf_all[doc1-1][word]*tf_idf_all[doc2-1][word])
    return cosine_sim

def remerge():  #remerge要從新取得原本的初始merge值和similarity
    # import numpy as np
    for i in range(1,1096):
        sim_record=[]
        for j in range(1,1096):
            sim_record.append(cosine_sim_num(i,j))
        sim_sum.append(sim_record)
    #將similarity存入sim_sum二維陣列中
    for i in range(0,1095):
        sim_sum[i][i]=0    #自己和自己相似度設為0
    for i in range(1095):
        cluster_dict[i]=[]
    #一開始1095個視為各自cluster，字典存放為空的list
# sim_dict_doc_num={}
# for i in range(1095):
#     sim_dict_doc_num[i]=sim_sum[i].index(max(sim_sum[i])) 
        #字典key為0， value就是對應最大相似度的另一個文件編號
# find_max_sim=[]
# for i in range(1095):
#     find_max_sim.append(sim_sum[i][sim_dict_doc_num[i]])
    #從第一號文件找出每號文件對應最大的相似度存為陣列
def max_sim_info():
    max_sim_now=[]
    max_sim_now.append(max(find_max_sim))#從1095個文件內找最大的相似度
    max_sim_now.append(find_max_sim.index(max(find_max_sim)))
    max_sim_now.append(sim_dict_doc_num[find_max_sim.index(max(find_max_sim))])
    # max(find_max_sim)  #sim最高的數值
    return max_sim_now

# max_sim_now  文件幾號(需+1)和幾號(需+1)目前有最大的相似度
# max_sim_now=max_sim_info()
# compare=[]
# for i in range(1,1096):
#     compare.append(cosine_sim_num(101,i))
# cluster[1]
#每一個cluster先設為一個空陣列，存在字典cluster_dict裡面
# cluster.append()  #merge
def update_act():
    cluster_dict[min(max_sim_now[1],max_sim_now[2])].append(max(max_sim_now[1],max_sim_now[2]))
    if cluster_dict[max(max_sim_now[1],max_sim_now[2])]!=[]:  #後merge的cluster為非一個文件的cluster(裡面還有先前merge的文件)
        for x in range(len(cluster_dict[max(max_sim_now[1],max_sim_now[2])])):
            cluster_dict[min(max_sim_now[1],max_sim_now[2])].append(cluster_dict[max(max_sim_now[1],max_sim_now[2])][x])
    #將cluster用字典的方式儲存，以編號小的為key，value就是目前cluster裡面的文件編號
    #這裡"還沒有包含本身"，cluster的編號就是以他為群體
    for i in range(0,1095):
        if sim_sum[max_sim_now[1]][int(i)] > sim_sum[max_sim_now[2]][int(i)]:
#             sim_sum[max_sim_now[2]][int(i)]=cosine_sim_num(max_sim_now[2]+1,int(i)+1)
            sim_sum[max_sim_now[1]][int(i)]=sim_sum[max_sim_now[2]][int(i)]
            sim_sum[i][1]=sim_sum[max_sim_now[2]][int(i)]
                    #相似度大代表比較近，cluter裡面的點分別跟外面的點的相似度取較小的          
        else:              
            sim_sum[max_sim_now[2]][int(i)]=sim_sum[max_sim_now[1]][int(i)]
            sim_sum[i][2]=sim_sum[max_sim_now[1]][int(i)]
        sim_sum[max(max_sim_now[1],max_sim_now[2])][i]=0
        sim_sum[i][max(max_sim_now[1],max_sim_now[2])]=0   
    del cluster_dict[max(max_sim_now[1],max_sim_now[2])]

sim_sum=[]
cluster_dict={}
remerge()   
for x in range(1075):
    sim_dict_doc_num={}
    for i in range(1095):
        sim_dict_doc_num[i]=sim_sum[i].index(max(sim_sum[i])) 
        #字典key為0， value就是對應最大相似度的另一個文件編號
    find_max_sim=[]
    for i in range(1095):
        find_max_sim.append(sim_sum[i][sim_dict_doc_num[i]])
    #從第一號文件找出每號文件對應最大的相似度存為陣列
    max_sim_now=max_sim_info()
    update_act()

for x in cluster_dict:
    cluster_dict[x]=[i+1 for i in cluster_dict[x]]  #文件編號為sim_sum裡面的編號+1，例如sim_sum100為文件101號
#     cluster_dict[x].append(x+1)  #加入開頭的最小的文件編號
    cluster_dict[x].sort()

fp = open('20.txt', "a+", encoding='utf-8')
for x in cluster_dict:
    print(x+1,file=fp)
    for i in range(len(cluster_dict[x])): #cluster_dict[x] 是一個list
        print(cluster_dict[x][i],file=fp)
    print('',file=fp)
fp.close()
# compare2=[]
# for i in range(1,1096):
#     compare2.append(cosine_sim_num(101,i))
# cosine_sim_num(871,815)
sim_sum=[]
cluster_dict={}
remerge()   
for x in range(1082):
    sim_dict_doc_num={}
    for i in range(1095):
        sim_dict_doc_num[i]=sim_sum[i].index(max(sim_sum[i])) 
        #字典key為0， value就是對應最大相似度的另一個文件編號
    find_max_sim=[]
    for i in range(1095):
        find_max_sim.append(sim_sum[i][sim_dict_doc_num[i]])
    #從第一號文件找出每號文件對應最大的相似度存為陣列
    max_sim_now=max_sim_info()
    update_act()

for x in cluster_dict:
    cluster_dict[x]=[i+1 for i in cluster_dict[x]]  #文件編號為sim_sum裡面的編號+1，例如sim_sum100為文件101號
#     cluster_dict[x].append(x+1)  #加入開頭的最小的文件編號
    cluster_dict[x].sort()

fp = open('13.txt', "a+", encoding='utf-8')
for x in cluster_dict:
    print(x+1,file=fp)
    for i in range(len(cluster_dict[x])): #cluster_dict[x] 是一個list
        print(cluster_dict[x][i],file=fp)
    print('',file=fp)
fp.close()

sim_sum=[]
cluster_dict={}
remerge()   
for x in range(1087):
    sim_dict_doc_num={}
    for i in range(1095):
        sim_dict_doc_num[i]=sim_sum[i].index(max(sim_sum[i])) 
        #字典key為0， value就是對應最大相似度的另一個文件編號
    find_max_sim=[]
    for i in range(1095):
        find_max_sim.append(sim_sum[i][sim_dict_doc_num[i]])
    #從第一號文件找出每號文件對應最大的相似度存為陣列
    max_sim_now=max_sim_info()
    update_act()

for x in cluster_dict:
    cluster_dict[x]=[i+1 for i in cluster_dict[x]]  #文件編號為sim_sum裡面的編號+1，例如sim_sum100為文件101號
#     cluster_dict[x].append(x+1)  #加入開頭的最小的文件編號
    cluster_dict[x].sort()

fp = open('8.txt', "a+", encoding='utf-8')
for x in cluster_dict:
    print(x+1,file=fp)
    for i in range(len(cluster_dict[x])): #cluster_dict[x] 是一個list
        print(cluster_dict[x][i],file=fp)
    print('',file=fp)
fp.close()

# len(cluster_dict)
# total1=0
# for i in cluster_dict:
#    total1= total1 + len(cluster_dict[i])
