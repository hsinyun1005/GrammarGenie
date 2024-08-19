import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import math
import torch
import re
import json
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


def phrase2pos(phrase):
    tokens = nltk.word_tokenize(phrase)
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags

def pos2gp(pos_tag):
    search_word = None
    modified_words = []
    flag = False
    for word, tag in pos_tag:
        if flag is False:
            if tag in ['JJ', 'JJR', 'JJS']:
                search_word = word
                modified_words.append('ADJ')
                flag = True

            if tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                search_word = word
                modified_words.append('V')
                flag = True

            if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                search_word = word
                modified_words.append('N')
                flag = True
        else:
            modified_words.append(word)        
    return (search_word, ' '.join(modified_words))



def SC_retriever(pairs):
    lemmatizer = WordNetLemmatizer()
    word = pairs[0]
    grammar_p = pairs[1] + ' n'

    if pairs[1].split()[0] == 'ADJ':
        word = lemmatizer.lemmatize(word, wordnet.ADJ)
        raw_data_path = 'result/_Adjectives.json'
        sc_data_path = 'result/_Adj_SC.json'

    elif pairs[1].split()[0] == 'V':
        word = lemmatizer.lemmatize(word, wordnet.VERB)
        raw_data_path = 'result/verbs_ch02.json'
        sc_data_path = 'result/_Verb_SC.json'
    
    elif pairs[1].split()[0] == 'N':
        word = lemmatizer.lemmatize(word, wordnet.NOUN)
        raw_data_path = 'result/_Nouns.json'
        sc_data_path = 'result/_Noun_SC.json'

        # Search for corresponding Group within the Grammar Pattern 
    with open(raw_data_path) as f:
        raw_data = json.load(f)

    if grammar_p in raw_data.keys():
        for group in raw_data[grammar_p]['Groups']:
            if word in group['Members']:
                target_group = group['Group']


    # Retrieve corresponding SC 
    with open(sc_data_path) as f:
        stored_SC = json.load(f)

    if grammar_p in stored_SC.keys():
        return stored_SC[grammar_p][target_group].upper()







model_dir = f"gen4"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 256

with open('prepare_data/pattern4.json') as f:
    data = json.load(f)

def gen(text):
    inputs = ["summarize: " + text]
    inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
    output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=2, max_length=128)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    predicted = nltk.sent_tokenize(decoded_output.strip())[0]
    return predicted  


st.title("GrammarGenie")
st_text_area = st.text_input("Text partial sentence to generate the grammar pattern")
option = ['','He saw the other man as a real threat','The party called for further progress','Both the husbands and wifes consent','Voters are surprisingly uninformed','The university host a conference']
st_text_area1 = st.selectbox("Select a partial sentence",option)
if st_text_area != '':
    output = gen(st_text_area)
if st_text_area == '':
    output = gen(st_text_area1)

if st.button('generate'):
    st.markdown("__" + output + "__")
    word = output.split(' ')
    find_word = re.compile(word[0]+' '+word[1])
    find_doing = re.compile('doing something')
    find_do = re.compile('do something')
    try:
        for i in data[word[0]]:
            examp = i['example'].replace('<span class="x">','').replace('</span>','').replace('<span class="cl">','')
            if output == i['pattern']:
                st.markdown(examp)
            if output != i['pattern']:
                match_word = find_word.findall(i['pattern'])
                if match_word != []:
                    #do
                    match_do = find_do.findall(output)
                    if match_do != []:
                        find_i = re.compile(word[0]+' '+word[1]+' '+word[2])
                        match_i = find_i.findall(i['pattern'])
                        st.markdown(examp)
                    #doing
                    match_doing = find_doing.findall(output)
                    if match_doing != []:
                        find_i = re.compile(word[0]+' '+word[1]+' '+word[2])
                        match_i = find_i.findall(i['pattern'])
                        st.markdown(examp)
                    # something
                    pa_do = find_do.findall(i['pattern'])
                    pa_doing = find_doing.findall(i['pattern'])
                    if match_doing == [] and match_do == [] and pa_doing == [] and pa_do == []:
                        st.markdown(examp)
    except:
        st.markdown("__" + '-' + "__")

    op = output.split(' ')
    output = op[0]+' '+op[1]
    # Genarate SC
    pos_pairs = phrase2pos(output)
    word_gp_pairs = pos2gp(pos_pairs)
    # st.markdown(word_gp_pairs[1])
    st.markdown(output + ' ' + "__" +SC_retriever(word_gp_pairs)+ "__")


#output 是預測的grammar pattern
            
            
