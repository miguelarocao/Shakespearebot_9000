# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:59:20 2016

@author: Akshta
"""
from preprocess import dataHandler
import numpy as np
import nltk
import json
import math

filename='data/shakespeare.txt'
myData=dataHandler(filename, 1)
myData.gen_word_idx()
#myData = myHmm.myData        
with open('data/shakespeare_pos.json') as data_file:    
    pos_tag = json.load(data_file)

pos_tag_d = {}    
for key in pos_tag.keys():    
    pos_tag_d[key] = str(pos_tag[key])            
Size = 30
filename = "visual" + str(Size) + '.txt'    
f = open(filename , "w")

A_viz = np.load("trained/A_quatrain_"+ str(Size) +".npy");
O_viz = np.load("trained/O_quatrain_"+ str(Size) +".npy");
nwords = 10;

filename='data/shakespeare.txt'

viz = np.array(O_viz)
sort_viz = viz.argsort(axis = 0)[:][-nwords:]

nrows = len(sort_viz[1])
ncolumns = len(sort_viz)
mylist = np.zeros((nwords,6))

with open('data/shakespeare_syllables.json') as data_file:    
    sh_syll_d = json.load(data_file)

with open('data/spenser_syllables.json') as data_file:    
    sp_syll_d = json.load(data_file)

syll_d = sh_syll_d.copy()
syll_d.update(sp_syll_d)

f.write('Top 10 words without normalization')
f.write('\n')

sort_viz = viz.argsort(axis = 0)[:][-nwords:]

# to print the top 10 words by probability
nrows = len(sort_viz[1])
ncolumns = len(sort_viz)
mylist = np.zeros((10,6))
for i in range(1,nrows-1,1):
    for j in range(ncolumns-1,-1,-1):
        f.write(myData.idx_dict[sort_viz[j][i]])        
        f.write(', ')
    f.write('\n')
   
nrows = len(viz)
ncolumns = len(viz[1])

sum = np.sum(viz,axis = 1)

# normalization
for i in range(nrows):
    for j in range(ncolumns):
        if myData.word_dict[myData.get_idx_word(i)] > 0:
            viz[i][j] = viz[i][j]/sum[i]
        else:   
            viz[i][j] = 0;
            
f.write('\n')        
f.write('Top 10 words with normalization')
f.write('\n')

# normalized top 10 words
sort_viz = viz.argsort(axis = 0)[:][-nwords:]
nrows = len(sort_viz[1])
ncolumns = len(sort_viz)
mylist = np.zeros((10,6))
for i in range(1,nrows-1,1):
    for j in range(ncolumns-1,-1,-1):
        f.write(myData.idx_dict[sort_viz[j][i]])        
        f.write(', ')
    f.write('\n')

f.write('\n')
f.write( 'Part of Speech for top 10 words' )   
f.write('\n')

for i in range(1,nrows-1,1):
    for j in range(ncolumns-1,-1,-1):
        if pos_tag_d.has_key(myData.idx_dict[sort_viz[j][i]]):
            f.write(pos_tag_d[myData.idx_dict[sort_viz[j][i]]])
        else:
            wrd = myData.idx_dict[sort_viz[j][i]]
            tag = nltk.pos_tag([wrd])
            pos_tag_d[wrd]= tag[0][1]          
        print pos_tag_d[myData.idx_dict[sort_viz[j][i]]],
        f.write(', ')
    f.write('\n')

f.write('\n')

f.write('\n')
f.write('Number of syllable top 10 words')
f.write('\n')

for i in range(1,nrows-1,1):
    print i 
    for j in range(ncolumns-1,-1,-1):
        if syll_d.has_key(myData.idx_dict[sort_viz[j][i]]):
            tag = syll_d[myData.idx_dict[sort_viz[j][i]]]
            f.write("%d" % tag[0])
            f.write(', ')
    f.write('\n')

f.write('\n')
pos = {}
main_pos = {}
tags = list(set(pos_tag_d.values()))

# to analyse just these 5 
main_tags = [ 'Noun', 'Verb', 'Adj', 'Adverb', 'Pronoun']
f.write('\n')
f.write('\n')

for i in range(len(tags)):
    pos[tags[i]] = i

for i in range(len(main_tags)):
    main_pos[main_tags[i]] = i

nrows = len(O_viz)
ncolumns = len(O_viz[1])    
pos_arr = np.zeros((len(tags), ncolumns))
main_pos_arr = np.zeros((len(tags), ncolumns))

pos_sum = np.zeros((len(tags), 1))
main_pos_sum = np.zeros((len(tags), 1))

# sum of probabilities over all the words
for i in range(nrows):
    if pos_tag_d.has_key(myData.idx_dict[i]):
        tag = pos_tag_d[myData.idx_dict[i]]
        k = pos[tag]
        for j in range(ncolumns):
            pos_arr[k][j] = pos_arr[k][j] + O_viz[i][j]
            pos_sum[k] = pos_sum[k] + 1
            if main_pos.has_key(tag):
                main_pos_arr[main_pos[tag]][j] = main_pos_arr[main_pos[tag]][j] + O_viz[i][j]
                main_pos_sum[main_pos[tag]] = main_pos_sum[main_pos[tag]] + 1
                
sort_pos = pos_arr.argsort(axis = 0)[:][-3:]

nrows = len(sort_pos[1])
ncolumns = len(sort_pos)

pos_rev = {}
for key in pos.keys():
    pos_rev[pos[key]]= key

f.write('\n')
f.write('Most probable parts of speech without normalization')
f.write('\n')

for i in range(1,nrows-1,1):
    for j in range(ncolumns-1,-1,-1):
        print pos_rev[sort_pos[j][i]] + ',',
        f.write(pos_rev[sort_pos[j][i]])
        f.write(' ')
        f.write(str(round(pos_arr[(sort_pos[j][i]),i],2)))
        f.write(', ')
    f.write('\n')
    print '\n'

nrows = len(pos_arr)
ncolumns = len(pos_arr[1])    

norm_pos = np.zeros((nrows,ncolumns))

for i in range(nrows):
    for j in range(ncolumns):
        if pos_sum[i] == 0:
            norm_pos[i][j] = 0
        else:            
            norm_pos[i][j] = pos_arr[i][j]*100/math.sqrt(pos_sum[i])


mnrows = len(pos_arr)
mncolumns = len(pos_arr[1])    

for i in range(mnrows):
    for j in range(mncolumns):
        if main_pos_sum[i] == 0:
            main_pos_arr[i][j] = 0
        else:            
            main_pos_arr[i][j] = main_pos_arr[i][j]/main_pos_sum[i]


sort_pos = norm_pos.argsort(axis = 0)[:][-3:]
main_sort_pos = main_pos_arr.argsort(axis = 0)[:][-3:]

nrows = len(sort_pos[1])
ncolumns = len(sort_pos)

pos_rev = {}
for key in pos.keys():
    pos_rev[pos[key]]= key

f.write('\n')
f.write('Most probable Parts of Speech with normalization')
f.write('\n')

for i in range(1,nrows-1,1):
    for j in range(ncolumns-1,-1,-1):
        print pos_rev[sort_pos[j][i]] + ',',
        f.write(pos_rev[sort_pos[j][i]])
        f.write(' ')
        f.write(str(round(pos_arr[(sort_pos[j][i]),i],2)))
        f.write(', ')
    f.write('\n')

f.write('\n')

nrows = len(O_viz)
ncolumns = len(O_viz[1])    
nsyll = np.zeros((len(tags), ncolumns))
# For number of syllables
for i in range(nrows):
    if syll_d.has_key(myData.idx_dict[i]):
        syll = syll_d[myData.idx_dict[i]]
        k = syll[0]
        for j in range(ncolumns):
            nsyll[k][j] = nsyll[k][j] + O_viz[i][j]

nrows = len(O_viz)
ncolumns = len(O_viz[1])    
stress = np.zeros((nrows, ncolumns))

stress_dict = {}
stress_rev = {}
count = 0
f.write('\n')


stress_sum = np.zeros((nrows, 1))

# To analyze the stress pattern in the sequence
for i in range(nrows):
    flag = 0
    if syll_d.has_key(myData.idx_dict[i]):
        syll = syll_d[myData.idx_dict[i]]
        stress_l = syll[1]
        for key in stress_dict.keys():
            if cmp(stress_l[0], stress_dict[key]) == 0:
                k = key
                flag = 1
        if flag == 0:
            stress_dict[count] = stress_l[0]
            k = count
            count = count + 1                 
        for j in range(ncolumns):
            stress[k][j] = stress[k][j] + O_viz[i][j]                        
            stress_sum[k] = stress_sum[k] + 1  

f.write('\n')

sort_stress = stress.argsort(axis = 0)[:][-2:]

ncolumns = len(sort_stress)
nrows = len(sort_stress[1])  

f.write('\n')
f.write('Stress patterns without normalization')
f.write('\n')

for i in range(1,nrows-1,1):
    for j in range(ncolumns-1,-1,-1):
        if i > 0 and i < nrows-1:
            print stress_dict[sort_stress[j][i]],
            f.write('[')
            for item in stress_dict[sort_stress[j][i]]:
                f.write("%d" % item)
                f.write(' ')
            f.write(']')
            f.write(', ')
    f.write('\n')                
    print '\n'    



nrows = len(stress_dict)
ncolumns = len(stress[1])    

for i in range(nrows):
    for j in range(ncolumns):
        if stress_sum[i] == 0:
            stress[i][j] = 0
        else:            
            stress[i][j] = stress[i][j]/stress_sum[i]


sort_stress = stress.argsort(axis = 0)[:][-2:]


ncolumns = len(sort_stress)
nrows = len(sort_stress[1])  

f.write('\n')
f.write('Stress Patterns with normalization')
f.write('\n')


for i in range(1,nrows-1,1):
    for j in range(ncolumns-1,-1,-1):
        if i > 0 and i < nrows-1:
            print stress_dict[sort_stress[j][i]],
            f.write('[')
            for item in stress_dict[sort_stress[j][i]]:
                f.write("%d" % item)
                f.write(' ')
            f.write(']')
            f.write(', ')
    f.write('\n')                
    print '\n'    


f.close()