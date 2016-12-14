# -*- coding: utf-8 -*-
import nltk
from nltk.tokenize import StanfordTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer  
from nltk.collocations import BigramCollocationFinder
import codecs

from selenium import webdriver
import time
import re
import math
from multiprocessing import Pool



unlimitpattern = [['JJ','NN'], ['RB', 'VB']]
limitpattern = [['JJ','JJ'], ['RB','JJ'], ['NN','JJ']]

#search phrase such as "Hello NEAR excellent" in serach engine and get search results
def request(keywords):
    # PROXY = "185.23.142.89:80" # IP:PORT or HOST:PORT
    # chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument('--proxy-server=http://%s' % PROXY)
    # user_agent = "'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'"
    # chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument("--user-agent=" + user_agent);
    driver = webdriver.Chrome()#(chrome_options=chrome_options)
    driver.get('https://www.google.com/xhtml')
    time.sleep(2)
    search_box = driver.find_element_by_name('q')
    res = []
    for keywordarr in keywords:
        partres = []
        for keyword in keywordarr:
            search_box.clear()
            search_box.send_keys(keyword)
            search_box.submit()
            time.sleep(30)
            results = driver.find_element_by_id('resultStats').text
            if results == "":
                partres.append(0.01)
            else:
                results = re.findall('\d+', results)
                num = ""
                for r in results:
                    num = num+r
                partres.append(int(num))
        res.append(partres)
        break
    driver.quit()   
    return res

#combine phrase with NEAR operator such as "hello word" NEAR "excellent"
def combinekeywords(wordpair,type):
    result="\""
    result+=wordpair[0]
    result+=" "
    result+=wordpair[1]
    result+="\" "
    result+="NEAR "
    if type==1:
        result+="\"excellent\""
    else:
        result+="\"poor\""
    return result
 
#compute semantic orientation hits 
def f(allreview,type):
    print("...so...")
    totallen = len(allreview)
    keywords = []
    for i in range(totallen):
        oneview = allreview[i]
        l = len(oneview)
        onekeywords = []
        for j in range(l):
            keyword = combinekeywords(oneview[j],type)
            onekeywords.append(keyword)
        keywords.append(onekeywords)
    hits = request(keywords)
    return hits

#compute every sentence semantic-orientation value
def sovalue(excelhits, poorhits, poshit, neghit):
    totallen  = len(excelhits)
    sores = []
    for i in range(totallen):
        res = 0
        partlen = len(excelhits[i])
        for j in range(partlen):
            res = res + math.log(excelhits[i][j]*neghit/(poorhits[i][j]*poshit),2)
        sores.append(res/partlen)
    return sores
              
def getPhraseByPos(dic):
    l = len(dic)
    result=[]
    i=0
    while i<l-1:
        v1 = dic[i][1]
        v2 = dic[i+1][1]
        k1 = dic[i][0]
        k2 = dic[i+1][0]
        # print([v1,v2])
        if len(v1)<2:
            i = i+1
            continue
        if len(v2)<2:
            i = i+1
            continue
        v1 = v1[0:2]
        v2 = v2[0:2]
        if [v1,v2] in unlimitpattern:
            i = i+2
            result.append([k1,k2])
        elif [v1,v2] in limitpattern:
            if i == l-2:
                result.append([k1,k2])
                break
            thirdkey = dic[i+2][0]
            v3 = dic[i+2][1]
            if (len(v3)<2):
                result.append([k1,k2])
                i = i+3
                continue
            v3 = v3[0:2]
            if v3 == 'NN':
                i = i+1
                continue
            else:
                result.append([k1,k2])
                i = i+2
        else:
            i=i+1
    return result


if __name__ == '__main__':#very important
    # res = request([["excellent"],["poor"]])
    poshit = 1510000000032
    neghit = 771000000037 
    print(poshit)
    print(neghit)
    stopword = ["-LSB-","-RSB-","-LRB-","-RRB-"]              
    tokenizer = StanfordTokenizer()
    filename = "F:/course/sentimentcode/rt-polarity.neg"
    file_object = codecs.open(filename,'r','utf-8')
    allres = []
    try:
        all_the_text = file_object.read()
        arr = tokenizer.tokenize(all_the_text)
        la = len(arr)
        correct = 0
        for line in arr:
            ax = line.split()
            wordarr = []
            for word in ax:
                if word in stopword:
                    continue
                wordarr.append(word)
            list = nltk.pos_tag(wordarr)
            result = getPhraseByPos(list)
            if len(result)==0:
                continue
            allres.append(result) 
        posres = f(allres,1)
        negres = f(allres,0)
        # with Pool(processes=2) as pool:
            # print("begin pos")
            # posresult = pool.apply_async(f, (allres,1,))
            # print("begin neg")
            # negresult = pool.apply_async(f, (allres,0,))
            # posres = posresult.get()
            # negres = negresult.get()
        print(posres)
        print(negres)
        scores = sovalue(posres, negres, poshit, neghit)
        correct=0
        for i in range(len(scores)):
            if scores[i] > 0:
                continue
            correct=correct+1
        print("correct is:")
        print(correct)
    finally:
        file_object.close()