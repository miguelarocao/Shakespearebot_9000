#Preprocessing Classes and Functions

import numpy as np
import string
import sys
import scipy
import json

##For Language Processing
from nltk.corpus import cmudict
from hyphen import Hyphenator, dict_info
import hyphen.dictools as dicttools

class dataHandler:

    def __init__(self, filename, null_append):
        """Constructor. Calls import_data()."""
        self.filename=filename
        self.poem_length=14
        self.num_syllables=10
        self.syllable_file=filename[:-4]+"_syllables.json"

        #key is line number, string is stanza
        self.name=np.array(["quatrain","volta","couplet"])
        self.count=np.array([8,4,2])
        self.line_type=list(np.repeat(self.name,self.count))
        self.null_added = null_append

        #value is index with which current index rhymes with
        self.rhyming_pairs=[2,3,0,1,6,7,4,5,10,11,8,9,13,12]

        #dictionaries
        self.syllable_dict={} #word is key
        self.rhyming_dict={}
        self.word_dict={}
        self.idx_dict={}
        self.data_dict={}
        for item in self.name:
            self.data_dict[item]=[]

        #initialization
        self.import_data()
        self.pyhyphen_setup()

    def import_data(self):
        """Imports the poem data from given filename"""

        poem_cnt=0
        in_poem=False
        in_line_cnt=0 #keeps track of current line in poem
        rhyming_arr=[]

        #check if need to generate syllable dictionary
        syl_found=False
        try:
            temp_dict=json.load(open(self.syllable_file))
            syl_found=True
            for key,value in temp_dict.iteritems():
                self.syllable_dict[str(key)]=value
            print "Succesfully loaded syllable information from: "+self.syllable_file
        except IOError:
            print "Failed to load syllable information, "+self.syllable_file+" not found."
            print "Importing poems and generating syllable information..."


        #keeps hyphens and apostrophes
        custom_punctuation=string.punctuation.replace('-','').replace("'",'')

        in_text=False
        with open(self.filename,'r') as f:
            for line in f.readlines():
                #removes leading and trailing \n and spaces
                line=line.strip()

                #tokenize by spaces
                line=line.split(' ')

                #poem?
                if not in_poem and line!=['']:
                    in_poem=True
                    continue
                else: #in poem
                    if not in_text and line==['']:
                        continue
                    in_text=True
                    if in_text and line==['']:
                        if not syl_found:
                            print "Loaded poem "+str(poem_cnt)
                        in_poem=False
                        in_line_cnt=0
                        poem_cnt+=1
                        self.populate_rhyming(rhyming_arr)
                        rhyming_arr=[]
                        in_text=False
                        continue

                    #not a lank line
                    #remove punctuation and make lowercase
                    for i in range(len(line)):
                        line[i]=line[i].translate(None,custom_punctuation).lower()

                    #generate syllable information
                    if not syl_found:
                        self.gen_syllable_info(line)
                        print "Loaded line "+str(in_line_cnt)+" of 14"

                    #reverse
                    line.reverse()

                    if self.null_added == 1:
                        #append a 'NULL' at the end
                        line.append("NULL")

                    self.data_dict[self.line_type[in_line_cnt]].append(line)

                    rhyming_arr.append(line[0])
                    in_line_cnt+=1

        f.close()

        if in_poem:
            in_poem=False
            poem_cnt+=1

        print "Loaded "+str(poem_cnt)+" poems from "+self.filename+"."

        if not syl_found:
            json.dump(self.syllable_dict,open(self.syllable_file,'w'))
            print "Wrote syllable information to "+str(self.syllable_file)

        return

    def populate_rhyming(self,rhyming_arr):
        if len(rhyming_arr)!=14:
            raise AssertionError("populate_rhyming(): Invalid input!")

        for r in range(len(rhyming_arr)):
            rhyme=rhyming_arr[self.rhyming_pairs[r]]
            word=rhyming_arr[r]
            #check word as key
            try:
                #check if already has rhyme paired
                self.rhyming_dict[word].index(rhyme)
            except KeyError:
                #rhyme is not paired and word currently has no rhymes
                self.rhyming_dict[word]=[rhyme]
            except ValueError:
                #rhyme is paired and word already has other rhymes
                self.rhyming_dict[word].append(rhyme)


    def get_rhyme(self,word,max_depth=5):
        """Gets rhyme for input word, if rhyme doesn't exist in dict returns None.
        Never returns itself as a rhyme."""

        #probabilty of stopping at current depth
        p=0.4

        #current key
        curr_word=word

        #stops with some probability
        depth_count=0
        while True:
            try:
                rhyme_idx=np.random.randint(0,len(self.rhyming_dict[curr_word]))
            except KeyError:
                return None
            rhyme=self.rhyming_dict[curr_word][rhyme_idx]
            if (np.random.random()>p and depth_count<max_depth) or rhyme==word:
                depth_count+=1
                curr_word=rhyme
            else:
                return rhyme

    def get_all_data(self):
        """Returns all poem data, not separated by stanza type."""
        output=[]
        for mylist in self.data_dict.itervalues():
            output+=mylist
        return output

    def get_stanza_data(self,stanza=None):
        """Returns all poem data, separated by stanza type.
        Optional input specifies which type of stanza to fetch."""

        if not stanza:
            return self.data_dict
        else:
            return self.data_dict[stanza]

    def get_num_words(self):
        """Returns the number of unique words in the imported poems."""

        if not self.word_dict:
            print "get_num_words() WARNING: Please generate word dictionary first!"
            return

        return len(self.word_dict)

    def gen_word_idx(self,stanza=None):
        """Generates a dictionary of words mapped to unique indices and vice versa.
        Optinal stanza input only does this over a specific stanza"""

        #else generate
        if not stanza:
            all_words=self.get_all_data()
        else:
            all_words=self.get_stanza_data(stanza)

        word_count=0
        for word_list in all_words:
            for word in word_list:
                try:
                    self.word_dict[word]
                except KeyError:
                    self.word_dict[word]=word_count
                    self.idx_dict[word_count]=word
                    word_count+=1
        return

    def get_word_idx(self,word):
        """Returns index associated with a given word"""
        return self.word_dict[word]

    def get_idx_word(self,idx):
        """Returns word associated with a given index"""
        return self.idx_dict[idx]

    ####GENERATION
    def generate_poem(self,A,O,filename=None):
        """Generates a poem based on the transition matrix A and the
        observation matrix O.
        Optional input prints to screen by default, otherwise writes to file."""

        #word_max=10 #maximum number of words per line
        special_prob=0.2 #probability that we add special punctuation
        special_punc=['!','?',';']

        poem_arr=[] #list of lists
        for i in range(self.poem_length):
            syllable_count=0
            curr_state=0 #start state
            line=[]
            word_count=0
            new_stress=1
            while True:
                #get new state at random
                idx=range(len(A[curr_state]))
                distr=scipy.stats.rv_discrete(values=(idx,tuple(A[curr_state,:])))
                new_state=distr.rvs()

                if not line:
                    #first word
                    try:
                        word=self.get_rhyme(poem_arr[self.rhyming_pairs[i]][-1])
                        if word:
                            gen_word=False
                        #print poem_arr[self.rhyming_pairs[i]][-1]+" "+str(word)
                    except IndexError:
                        #previous rhyming line doesn't exist!
                        #select first word from rhyming dictionary at random
                        word=np.random.choice(self.rhyming_dict.keys())

                else:
                    #not first word!
                    idx=range(np.shape(O)[0]) #number of rows
                    modprob=self.get_sliced_distr(O[:,new_state],new_stress,self.num_syllables-syllable_count)
                    distr=scipy.stats.rv_discrete(values=(idx,tuple(modprob)))
                    word_idx=distr.rvs()
                    word=self.get_idx_word(word_idx)

                #update syllable info
                syl_num,syl_stress=self.syllable_dict[word]
                syllable_count+=syl_num
                success=False
                for stress in syl_stress:
                    if stress[-1]==new_stress:
                        new_stress=stress[0]^1 #invert stress for next word
                        success=True
                        break

                #special case for first word
                if not success and not line:
                    new_stress=syl_stress[0][0]^1


                line.append(word)
                word_count+=1
                #check end conditions: end state or syllable max reached
                if new_state==(len(A[0,:])-1) or syllable_count>=self.num_syllables:
                    print "Line "+str(i)+" has "+str(syllable_count)+" syllables."
                    break

            #reverse line
            line.reverse()

            #add to poem
            poem_arr.append(line)

        #Convert to text
        poem=""
        for line in poem_arr:
            line[0]=string.capwords(line[0])
            line=" ".join(line)
            if np.random.random()<special_prob:
                punctuation=special_punc[np.random.randint(len(special_punc))]
            else:
                punctuation=','
            poem+=(line+punctuation+"\n")

        poem=poem[:-2]

        if filename:
            f=open(filename,'w')
            f.write(poem)
            f.close()
        else:
            print poem

    def get_sliced_distr(self,distr,end_stress,max_syl=None):
        """Modifies the distribution such that only words ending with the provided
        last stress syllable are next.
        Optional input gives maximum syllable"""

        #since syllable labelling isn't totally accurate
        max_syl_thresh=1

        #set probabilities to 0
        for i in range(len(distr)):
            good=False
            try:
                syl_num,syl_stress=self.syllable_dict[self.get_idx_word(i)]
                #check intonation
                for stress in syl_stress:
                    if stress[-1]==end_stress:
                        distr[i]+=1e-9 #avoids division by zero
                        good=True
                        break
                #check count if necessary
                if max_syl:
                    if syl_num>(max_syl+max_syl_thresh):
                        good=False
            except KeyError:
                #happens with NULL token
                pass

            if not good:
                distr[i]=0

        #normalize
        #print distr
        distr=distr/sum(distr)
        return distr


    def pyhyphen_setup(self):
        """Sets up pyhyphen"""
        en_list=['en_CA', 'en_PH', 'en_NA', 'en_NZ', 'en_JM', 'en_BS', 'en_US',
                    'en_IE', 'en_MW', 'en_IN', 'en_BZ', 'en_TT', 'en_ZA', 'en_AU',
                    'en_GH', 'en_ZW', 'en_GB']

        for lang in en_list:
            if not dicttools.is_installed(lang): dicttools.install(lang)

        return

    def parse_word(self,word):
        """Returns syllables and stress of each syllable if exists, else None.
        First tries using NLTK cmudict, if failes then uses pyhyphen."""
        d = cmudict.dict()
        syl_stress=None
        #try:
        #    word_info=d[word.lower()][0] #no way to differentiate between different pronunciation
        #    syl_num=len(list(y for y in word_info if y[-1].isdigit()))
        #    syl_stress=list(int(y[-1]) for y in word_info if y[-1].isdigit())
        #except KeyError:
        h_en=Hyphenator('en_GB')
        syl_num=len(h_en.syllables(unicode(word)))
        if syl_num==0:
            syl_num=1

        return syl_num,syl_stress

    def gen_syllable_info(self,sentence):
        """Generates syllable information based on input sentence.
        Input should be list of words."""

        is_stress=False

        tot_syl=0
        max_word=None
        max_syl=0
        for word in sentence:
            syl_num,syl_stress=self.parse_word(word)

            #generate stressed syllables based on order
            syl_stress_order=[]
            for syl in range(syl_num):
                syl_stress_order.append(int(is_stress))
                is_stress=not(is_stress)

            #if don't match then append both, saying it can be either stress
            if syl_stress_order!=syl_stress:
                if syl_stress:
                    syl_stress=[syl_stress_order,syl_stress]
                else:
                    syl_stress=[syl_stress_order]

            self.syllable_dict[word]=[syl_num,syl_stress]

            tot_syl+=syl_num

            #keep track of longest word
            if syl_num>max_syl:
                max_word=word
                max_syl=syl_num

        if tot_syl!=self.num_syllables:
            self.syllable_dict[max_word][0]+=self.num_syllables-tot_syl

def main():
    """Proprocessing tests and examples"""

    filename='data/spenser.txt'
    data=dataHandler(filename)
##    #print data.get_all_data()
##    #print data.get_stanza_data("couplet")
#    data.gen_word_idx()
##
##    #print data.word_dict
##
##    print data.get_num_words()
##    print len(data.word_dict)
##    print data.get_word_idx('sway')
##    print data.get_idx_word(data.get_word_idx('sway'))

    #hyp_list=pyhyphen_setup()


    #print get_syllable_info("summer")

    pass

if __name__ == '__main__':
    main()
