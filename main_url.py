import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup

#importing the large spacy library
import spacy.cli
spacy.cli.download("en_core_web_lg")
nlp = spacy.load('en_core_web_lg')

from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

from collections import Counter



import re

#Initialize Flask and set the template folder to "template"
app = Flask(__name__)


def get_wiki_content(url):
    req_obj=requests.get(url)
    text=req_obj.text
    soup=BeautifulSoup(text)
    all_paras=soup.find_all("p")# the text content is in p tag
    wiki_text=""
    for para in all_paras:
        wiki_text += para.text
    return wiki_text


def top_sent(url):
    required_text=get_wiki_content(url)


    #Removes non-alphabetic characters:
    def text_strip(doc):
    
        
        #ORDER OF REGEX IS VERY VERY IMPORTANT!!!!!!
        
        doc=re.sub("(\\t)", ' ', str(doc)) #remove escape characters
        doc=re.sub("(\\r)", ' ', str(doc)) 
        doc=re.sub("(\\n)", ' ', str(doc))
        
        doc=re.sub("(__+)", ' ', str(doc))   #remove _ if it occors more than one time consecutively
        doc=re.sub("(--+)", ' ', str(doc))   #remove - if it occors more than one time consecutively
        doc=re.sub("(~~+)", ' ', str(doc))   #remove ~ if it occors more than one time consecutively
        doc=re.sub("(\+\++)", ' ', str(doc))   #remove + if it occors more than one time consecutively
        doc=re.sub("(\.\.+)", ' ', str(doc))   #remove . if it occors more than one time consecutively

        doc=re.sub("\[\d*?\]", ' ', str(doc))   # to remove the numbers in sqaure brackets(given as reference) on wikipedia page
        
        doc=re.sub(r"[<>|&©ø\[\]\'\";?~*!]", ' ', str(doc)) #remove <>()|&©ø"',;?~*!
      
        
        #doc=re.sub("(\.\s+)", ' ', str(doc)) #remove full stop at end of words(not between)
        doc=re.sub("(\-\s+)", ' ', str(doc)) #remove - at end of words(not between)
        doc=re.sub("(\:\s+)", ' ', str(doc)) #remove : at end of words(not between)
        
        try:
            url_id= re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(doc))
            repl_url = url_id.group(3)
            doc = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)',repl_url, str(doc))
        except:
            pass #there might be emails with no url in them
      
        
        doc = re.sub("(\s+)",' ',str(doc)) #remove multiple spaces
              
        
        return doc

    textt=text_strip(required_text)
    text_tokens=[i.text for i in nlp(textt)]

    word_frequencies={}
    for w1 in text_tokens:
   
        if w1.lower() not in STOP_WORDS:
          if w1.lower() not in punctuation:
            if w1 not in word_frequencies.keys():
              word_frequencies[w1]=1
            else:
              word_frequencies[w1] +=1

    NER_list=[m.text for m in nlp(textt).ents]


    for k,v in Counter(NER_list).items():
      if k in word_frequencies.keys():
        word_frequencies[k]+=v
      else:
        word_frequencies.update({k:1})


    for k in word_frequencies.keys(): #normalized frequency of each word in text(weighted frequency)
      word_frequencies[k]=word_frequencies[k]/max(word_frequencies.values())


    #Tokenize over sentence using Spacy

    sentence_token= [k for k in nlp(text_strip(required_text)).sents]
    sentence_score={}
    for sent in sentence_token:
      for word in sent:
         if word.text.lower() in word_frequencies.keys():
           if sent not in sentence_score.keys():
             sentence_score[sent]=word_frequencies[word.text.lower()]
           else:
             sentence_score[sent]+= word_frequencies[word.text.lower()]

    similarity_score = np.zeros(len(sentence_token[1:]))
    for i, doc in enumerate(sentence_token[1:]):
        similarity_score[i] = sentence_token[0].similarity(doc)

    sentence_similarity = {sentence_token[1:][i]: similarity_score[i] for i in range(len(sentence_token[1:]))}

    sentence_simi={}
    for i,j in sentence_similarity.items():
        if j<=0.88:
            sentence_simi.update({i:j})

    final_list={}
    for i in sentence_simi.keys():
      for k,v in sentence_score.items():
        if i==k:
          final_list.update({i:v})


    import operator

    sorted_final_list=sorted(final_list.items(),key=operator.itemgetter(1), reverse=True)
    summary_1 =[]
    for k,v in final_list.items():
        if v>=sorted_final_list[0:int(len(sentence_token)*0.20)][-1][1]:      # taken only 20% as wiki pages / web pages have lot of content
            summary_1.append(k)

    summary_2=[sentence_token[0]] + summary_1
    final_summary=[word.text for word in summary_2]
    summary=" ".join(final_summary)

   # from heapq import nlargest
   # summary_1=nlargest(int(len(sentence_token)*0.20),final_list,key=final_list.get) # taken only 20% as wiki pages / web pages have lot of content
   # summary_2=[sentence_token[0]] + summary_1
   # final_summary=[word.text for word in summary_2]
   # summary=" ".join(final_summary)

    return(summary)

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method=="POST":
        url=request.form.get("url")
        url_content= top_sent(url)
        
        return url_content
    return render_template('index.html') # summarized_text = ("Summarized text",url_content))
    

if __name__ == '__main__':
   app.run(debug = True)
