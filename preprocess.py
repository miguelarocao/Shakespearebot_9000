#Preprocessing Classes and Functions

import numpy as np
import string

class dataHandler:

    def __init__(self, filename, null_append):
        """Constructor. Calls import_data()."""
        self.filename=filename
        self.poem_length=14

        #key is line number, string is stanza
        self.name=np.array(["quatrain","volta","couplet"])
        self.count=np.array([8,4,2])
        self.line_type=list(np.repeat(self.name,self.count))
        self.null_added = null_append

        #value is index with which current index rhymes with
        self.rhyming_pairs=[2,3,0,1,6,7,4,5,10,11,8,9,13,12]

        self.data_dict={}
        for item in self.name:
            self.data_dict[item]=[]

        self.rhyming_dict={}

        self.word_dict={}
        self.idx_dict={}

        self.import_data()

    def import_data(self):
        """Imports the poem data from given filename"""

        poem_cnt=0
        in_poem=False
        in_line_cnt=0 #keeps track of current line in poem
        rhyming_arr=[]

        with open(self.filename,'r') as f:
            for line in f.readlines():
                #removes leading and trailing \n and spaces
                line=line.strip()

                #tokenize by spaces
                line=line.split(' ')

                #poem?
                if not in_poem:
                    try:
                        int(line[0])
                        in_poem=True
                    except ValueError:
                        continue

                else: #in poem
                    if line==['']:
                        in_poem=False
                        in_line_cnt=0
                        poem_cnt+=1
                        self.populate_rhyming(rhyming_arr)
                        rhyming_arr=[]
                        continue

                    #remove punctuation and make lowercase
                    for i in range(len(line)):
                        line[i]=line[i].translate(None,string.punctuation).lower()
                        
                    if self.null_added == 1:
                        #append a 'NULL' at the end
                        line.append("NULL")

                    self.data_dict[self.line_type[in_line_cnt]].append(line)

                    rhyming_arr.append(line[-1])
                    in_line_cnt+=1

        f.close()

        if in_poem:
            in_poem=False
            poem_cnt+=1

        print "Loaded "+str(poem_cnt)+" poems from "+self.filename+"."

        #print self.rhyming_dict
        #print self.rhyming_dict['sway']
        #print self.rhyming_dict['day']

        #print self.get_rhyme('sway')

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
        """Gets rhyme for input word"""

        #probabilty of stopping at current depth
        p=0.4

        #current key
        curr_word=word

        #stops with some probability
        depth_count=0
        while True:
            rhyme_idx=np.random.randint(0,len(self.rhyming_dict[curr_word]))
            rhyme=self.rhyming_dict[curr_word][rhyme_idx]
            if np.random.random()>p and depth_count<max_depth:
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

def main():
    """Proprocessing tests and examples"""

    filename='data/shakespeare.txt'
    data=dataHandler(filename)
    #print data.get_all_data()
    #print data.get_stanza_data("couplet")
    data.gen_word_idx()

    #print data.word_dict

    print data.get_num_words()
    print len(data.word_dict)
    print data.get_word_idx('sway')
    print data.get_idx_word(data.get_word_idx('sway'))

    pass

if __name__ == '__main__':
    main()
