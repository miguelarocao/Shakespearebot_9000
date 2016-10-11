# Preprocessing Classes and Functions

import numpy as np
import string
import sys
import scipy
import json
import urllib2

##For Language Processing
from nltk.corpus import cmudict
from hyphen import Hyphenator, dict_info
import hyphen.dictools as dictools
from nltk.tag import PerceptronTagger
from nltk.data import find
import nltk


class dataHandler:
    def __init__(self, filenames, null_append, stanza="all"):
        """Constructor. Calls import_data()."""
        self.filenames = filenames
        self.poem_length = 14
        self.num_syllables = 10
        self.stanza = stanza
        self.pos_tag = None

        # key is line number, string is stanza
        self.name = np.array(["quatrain", "volta", "couplet"])
        self.stanza_count = np.array([8, 4, 2])
        self.line_type = list(np.repeat(self.name, self.stanza_count))
        self.null_added = null_append

        # Lists
        # value is index with which current index rhymes with
        self.rhyming_pairs = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 13, 12]
        self.pos_double = []  # populated in syllable_pos_setup

        # Dictionaries
        self.pos_lead = {}
        self.pos_restrict_lead = {}
        self.tag_dict = {}  # populated in syllable_pos_setup
        self.syllable_dict = {}  # word is key
        self.pos_dict = {}
        self.rhyming_dict = {}
        self.word_dict = {}
        self.idx_dict = {}
        self.data_dict = {}
        self.cmu_dict = {}
        for item in self.name:
            self.data_dict[item] = []

        # handles case where input is a single filename instead of a list
        if not isinstance(self.filenames, list):
            self.filenames = [self.filenames]

        # initialization
        self.syllable_pos_setup()
        for filename in self.filenames:
            self.import_data(filename)

    def import_data(self, filename):
        """Imports the poem data from given filename"""

        poem_cnt = 0
        in_poem = False
        in_line_cnt = 0  # keeps track of current line in poem
        rhyming_arr = []

        syllable_file = filename[:-4] + "_syllables.json"
        pos_file = filename[:-4] + "_pos.json"

        # check if need to generate syllable dictionary
        syl_found = False
        try:
            temp_dict = json.load(open(syllable_file))
            syl_found = True
            for key, value in temp_dict.iteritems():
                self.syllable_dict[str(key)] = value
            print "Succesfully loaded syllable information from: " + syllable_file
        except IOError:
            print "Failed to load syllable information, " + syllable_file + " not found."
            print "Importing poems and generating syllable information..."

        # check if need to generate POS dictionary
        pos_found = False
        try:
            temp_dict = json.load(open(pos_file))
            pos_found = True
            for key, value in temp_dict.iteritems():
                self.pos_dict[str(key)] = value
            print "Succesfully loaded POS information from: " + pos_file
        except IOError:
            print "Failed to load POS information, " + pos_file + " not found."
            print "Importing poems and generating POS information..."

        # keeps hyphens and apostrophes
        custom_punctuation = string.punctuation.replace('-', '').replace("'", '')

        in_text = False
        with open(filename, 'r') as f:
            for line in f.readlines():
                # removes leading and trailing \n and spaces
                line = line.strip()

                # tokenize by spaces
                line = line.split(' ')

                # poem?
                if not in_poem and line != ['']:
                    in_poem = True
                    continue
                else:  # in poem
                    if not in_text and line == ['']:
                        continue
                    in_text = True
                    if in_text and line == ['']:
                        if not syl_found:
                            print "Loaded poem " + str(poem_cnt)
                        in_poem = False
                        in_line_cnt = 0
                        poem_cnt += 1
                        self.populate_rhyming(rhyming_arr)
                        rhyming_arr = []
                        in_text = False
                        continue

                    # not a lank line
                    # remove punctuation and make lowercase
                    for i in range(len(line)):
                        line[i] = line[i].translate(None, custom_punctuation).lower()

                    # generate syllable information
                    if not syl_found:
                        self.gen_syllable_info(line)
                        # print "Loaded line "+str(in_line_cnt)+" of 14"

                    # generate POS information
                    if not pos_found:
                        self.gen_pos_info(line)

                    # reverse
                    line.reverse()

                    if self.null_added == 1:
                        # append a 'NULL' at the end
                        line.append("NULL")

                    # only append if the correct stanza type:
                    if self.stanza == "all" or self.stanza == self.line_type[in_line_cnt]:
                        self.data_dict[self.line_type[in_line_cnt]].append(line)
                        rhyming_arr.append(line[0])
                    else:
                        rhyming_arr.append(None)
                    in_line_cnt += 1

        f.close()

        if in_poem:
            in_poem = False
            poem_cnt += 1

        print "Loaded " + str(poem_cnt) + " poems from " + filename + "."

        if not syl_found:
            json.dump(self.syllable_dict, open(syllable_file, 'w'))
            print "Wrote syllable information to " + str(syllable_file)

        if not pos_found:
            json.dump(self.pos_dict, open(pos_file, 'w'))
            print "Wrote syllable information to " + str(pos_file)

        return

    def populate_rhyming(self, rhyming_arr):
        if len(rhyming_arr) != 14:
            raise AssertionError("populate_rhyming(): Invalid input!")

        for r in range(len(rhyming_arr)):
            if not rhyming_arr[r]:
                continue
            rhyme = rhyming_arr[self.rhyming_pairs[r]]
            word = rhyming_arr[r]
            # check word as key
            try:
                # check if already has rhyme paired
                self.rhyming_dict[word].index(rhyme)
            except KeyError:
                # rhyme is not paired and word currently has no rhymes
                self.rhyming_dict[word] = [rhyme]
            except ValueError:
                # rhyme is paired and word already has other rhymes
                self.rhyming_dict[word].append(rhyme)

    def get_rhyme(self, word, max_depth=5):
        """Gets rhyme for input word, if rhyme doesn't exist in dict returns None.
        Never returns itself as a rhyme."""

        # probabilty of stopping at current depth
        p = 0.4

        # current key
        curr_word = word

        # stops with some probability
        depth_count = 0
        while True:
            try:
                rhyme_idx = np.random.randint(0, len(self.rhyming_dict[curr_word]))
            except KeyError:
                return None
            rhyme = self.rhyming_dict[curr_word][rhyme_idx]
            if (np.random.random() > p and depth_count < max_depth) or rhyme == word:
                depth_count += 1
                curr_word = rhyme
            else:
                return rhyme

    def get_all_data(self):
        """Returns all poem data, not separated by stanza type."""
        output = []
        for mylist in self.data_dict.itervalues():
            output += mylist
        return output

    def get_stanza_data(self, stanza=None):
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

    def gen_word_idx(self, stanza=None):
        """Generates a dictionary of words mapped to unique indices and vice versa.
        Optinal stanza input only does this over a specific stanza"""

        # else generate
        if not stanza:
            all_words = self.get_all_data()
        else:
            all_words = self.get_stanza_data(stanza)

        word_count = 0
        for word_list in all_words:
            for word in word_list:
                try:
                    self.word_dict[word]
                except KeyError:
                    self.word_dict[word] = word_count
                    self.idx_dict[word_count] = word
                    word_count += 1
        return

    def get_word_idx(self, word):
        """Returns index associated with a given word"""
        return self.word_dict[word]

    def get_idx_word(self, idx):
        """Returns word associated with a given index"""
        return self.idx_dict[idx]

    ####GENERATION
    def generate_lines(self, A, O):
        """Generates a poem based on the transition matrix A and the
        observation matrix O.
        Output is a string."""

        # word_max=10 #maximum number of words per line
        special_prob = 0.15  # probability that we add special punctuation
        special_punc = ['!', '?', ';', '.']
        special_words = ['i']

        poem_arr = []  # list of lists
        for i in range(self.poem_length):
            syllable_count = 0
            curr_state = 0  # start state
            line = []
            word_count = 0
            new_stress = 1
            if self.stanza != "all" and self.line_type[i] != self.stanza:
                continue
            while True:
                # get new state at random
                idx = range(len(A[curr_state]))
                distr = scipy.stats.rv_discrete(values=(idx, tuple(A[curr_state, :])))
                curr_state = distr.rvs()

                if not line:
                    # first word
                    try:
                        relative_rhyme = (self.rhyming_pairs[i] - i) + len(poem_arr)
                        # print relative_rhyme
                        word = self.get_rhyme(poem_arr[relative_rhyme][-1])
                        if word:
                            gen_word = False
                            # print poem_arr[self.rhyming_pairs[i]][-1]+" "+str(word)
                    except IndexError:
                        # previous rhyming line doesn't exist!
                        # select first word from rhyming dictionary at random
                        word = self.get_state_rhyme(O[:, curr_state])

                else:
                    # not first word!
                    idx = range(np.shape(O)[0])  # number of rows
                    # restrict distribution
                    pos = self.pos_dict[line[-1].lower()]
                    max_syl = self.num_syllables - syllable_count
                    modprob = self.get_sliced_distr(O[:, curr_state], new_stress, max_syl, pos)
                    distr = scipy.stats.rv_discrete(values=(idx, tuple(modprob)))
                    # generate
                    word_idx = distr.rvs()
                    word = self.get_idx_word(word_idx)

                # in case word is not found
                if word is None:
                    continue

                # update syllable info
                syl_num, syl_stress = self.syllable_dict[word]
                syllable_count += syl_num
                success = False
                for stress in syl_stress:
                    if stress[-1] == new_stress:
                        new_stress = stress[0] ^ 1  # invert stress for next word
                        success = True
                        break

                # special case for first word
                if not success and not line:
                    new_stress = syl_stress[0][0] ^ 1

                # check if special word
                if word in special_words:
                    word = word.capitalize()

                # print word+" ("+self.pos_dict[word.lower()]+") ",
                line.append(word)
                word_count += 1
                # check end conditions: end state or syllable max reached
                if syllable_count >= self.num_syllables:  # curr_state==(len(A[0,:])-1) or :
                    # print "Line "+str(i)+" has "+str(syllable_count)+" syllables."
                    # print ""
                    break

            # Remove NULL
            # if curr_state==(len(A[0,:])-1):
            #    line = line[:-1]

            # reverse line
            line.reverse()

            # add to poem
            poem_arr.append(line)

        # Convert to text
        poem = ""
        for line in poem_arr:
            line[0] = string.capwords(line[0])
            line = " ".join(line)
            if np.random.random() < special_prob:
                punctuation = special_punc[np.random.randint(len(special_punc))]
            else:
                punctuation = ','
            poem += (line + punctuation + "\n")

        if self.stanza in ["all", "couplet"]:
            poem = poem[:-2]
            poem += '.'

        return poem

    def generate_naive_lines(self, A, O):
        """Generates a poem based on the transition matrix A and the
        observation matrix O.
        Output is a string."""

        # word_max=10 #maximum number of words per line
        special_prob = 0.15  # probability that we add special punctuation
        special_punc = ['!', '?', ';', '.']
        special_words = ['i']

        poem_arr = []  # list of lists
        for i in range(self.poem_length):
            curr_state = 0  # start state
            line = []
            word_count = 0
            # new_stress=1
            if self.stanza != "all" and self.line_type[i] != self.stanza:
                continue
            while True:
                # get new state at random
                idx = range(len(A[curr_state]))
                distr = scipy.stats.rv_discrete(values=(idx, tuple(A[curr_state, :])))
                curr_state = distr.rvs()

                if not line:
                    # first word
                    try:
                        relative_rhyme = (self.rhyming_pairs[i] - i) + len(poem_arr)
                        # print relative_rhyme
                        word = self.get_rhyme(poem_arr[relative_rhyme][-1])
                        if word:
                            gen_word = False
                            # print poem_arr[self.rhyming_pairs[i]][-1]+" "+str(word)
                    except IndexError:
                        # previous rhyming line doesn't exist!
                        # select first word from rhyming dictionary at random
                        word = self.get_state_rhyme(O[:, curr_state])

                else:
                    # not first word!
                    idx = range(np.shape(O)[0])  # number of rows
                    modprob = O[:, curr_state]
                    distr = scipy.stats.rv_discrete(values=(idx, tuple(modprob)))
                    # generate
                    word_idx = distr.rvs()
                    word = self.get_idx_word(word_idx)

                # check if special word
                if word in special_words:
                    word = word.capitalize()

                line.append(word)
                word_count += 1
                # check end conditions: end state
                if curr_state == (len(A[0, :]) - 1):
                    break

            line = line[:-1]
            # reverse line
            line.reverse()

            # add to poem
            poem_arr.append(line)

        # Convert to text
        poem = ""
        for line in poem_arr:
            line[0] = string.capwords(line[0])
            line = " ".join(line)
            if np.random.random() < special_prob:
                punctuation = special_punc[np.random.randint(len(special_punc))]
            else:
                punctuation = ','
            poem += (line + punctuation + "\n")

        if self.stanza in ["all", "couplet"]:
            poem = poem[:-2]
            poem += '.'

        return poem

    def get_state_rhyme(self, distr, rhyme=None):
        """Returns a randomly generated word with a rhyme sample from the
        provided state distribution."""

        distr = list(distr)

        for i in range(len(distr)):
            word = self.get_idx_word(i)

            try:
                self.rhyming_dict[word]
            except:
                # not in rhyming dict
                distr[i] = 0

        # normalize
        distr = distr / sum(distr)

        # sample
        idx = range(len(distr))  # number of rows
        distr = scipy.stats.rv_discrete(values=(idx, tuple(distr)))
        # generate
        word_idx = distr.rvs()

        return self.get_idx_word(word_idx)

    def get_sliced_distr(self, distr, end_stress, max_syl=None, curr_POS=None):
        """Modifies the distribution such that only words ending with the provided
        last stress syllable are next.
        Optional input 1 gives maximum syllable.
        Optional input 2 denotes whether or not to restrict by POS."""

        # since syllable labelling isn't totally accurate
        max_syl_thresh = 0

        distr = list(distr)

        # set probabilities to 0
        for i in range(len(distr)):
            good = False
            try:
                word = self.get_idx_word(i)
                syl_num, syl_stress = self.syllable_dict[word]
                # check intonation
                for stress in syl_stress:
                    if stress[-1] == end_stress:
                        distr[i] += 1e-9  # avoids division by zero
                        good = True
                        break
                # check count if necessary
                if max_syl:
                    if syl_num > (max_syl + max_syl_thresh):
                        good = False
                # check POS if necessary
                if curr_POS:
                    pos = self.pos_dict[word]
                    if pos == curr_POS and (curr_POS not in self.pos_double):
                        good = False
                    if pos in self.pos_lead and (curr_POS not in self.pos_lead[pos]):
                        good = False
                    if curr_POS in self.pos_restrict_lead and pos == self.pos_restrict_lead[curr_POS]:
                        good = False
            except KeyError:
                # happens with NULL token
                pass

            if not good:
                distr[i] = 0

        # normalize
        # print distr
        distr = distr / sum(distr)
        return distr

    def syllable_pos_setup(self):
        """Sets up syllables and POS tagging"""
        en_list = ['en_CA', 'en_PH', 'en_NA', 'en_NZ', 'en_JM', 'en_BS', 'en_US',
                   'en_IE', 'en_MW', 'en_IN', 'en_BZ', 'en_TT', 'en_ZA', 'en_AU',
                   'en_GH', 'en_ZW', 'en_GB']

        for lang in en_list:
            if not dictools.is_installed(lang): dictools.install(lang)

        self.cmu_dict = cmudict.dict()

        # sets up POS
        try:
            nltk.pos_tag(['test'])
            self.pos_tag = nltk.pos_tag
        except urllib2.URLError:
            PICKLE = "averaged_perceptron_tagger.pickle"
            AP_MODEL_LOC = 'file:' + str(find('taggers/averaged_perceptron_tagger/' + PICKLE))
            tagger = PerceptronTagger(load=False)
            tagger.load(AP_MODEL_LOC)
            self.pos_tag = tagger.tag

        self.tag_dict = {'NN': 'Noun', 'FW': 'Noun', 'JJ': 'Adjective', 'VB': 'Verb',
                         'IN': 'Preposition', 'CC': 'Conjunction',
                         'RP': 'Connector', 'TO': 'Connector', 'MD': 'Connector',
                         'RB': 'Adverb', 'WR': 'Wh-adverb',
                         'DT': 'DetPro', 'WD': 'DetPro', 'PD': 'DetPro', 'PR': 'DetPro', 'WP': 'DetPro',
                         'CD': 'Cardinal',
                         'EX': 'Existential there'}

        ##        self.tag_dict={'NN':'Noun', 'JJ':'Adjective','RB':'Adverb','VB':'Verb',
        ##          'IN':'Preposition','PR':'Pronoun','CC':'Conjunction',
        ##          'RP':'Particle','WR':'Wh-adverb','DT':'Determiner',
        ##          'TO':'To','MD':'Modal Aux','CD':'Cardinal', 'PD':'Predeterminer',
        ##          'WD':'Wh-determiner', 'WP':'Wh-pronoun','EX':'Existential there'}

        # POS which are allowed to happen twice in a row
        self.pos_double = []  # ['Noun','Adjective']

        # POS which can only occur sequentially
        # i.e. an Adverb must occur in fron of a verb
        self.pos_lead = {'Adverb': ['Verb'], 'Pronoun': ['Noun'], 'Adjective': ['Noun'],
                         'Preposition': ['Noun', 'Pronoun']}

        # POS which cannot occur sequentially
        # i.e. a preposition cannot come before a verb
        self.pos_restrict_lead = {'Preposition': 'Verb',}

        return

    def parse_word(self, word):
        """Returns syllables and stress of each syllable if exists, else None.
        First tries using NLTK cmudict, if failes then uses pyhyphen."""

        syl_stress = None
        try:
            word_info = self.cmu_dict[word.lower()][0]  # no way to differentiate between different pronunciation
            syl_num = len(list(y for y in word_info if y[-1].isdigit()))
            syl_stress = list(int(y[-1]) for y in word_info if y[-1].isdigit())
        except KeyError:
            h_en = Hyphenator('en_GB')
            syl_num = len(h_en.syllables(unicode(word)))
            if syl_num == 0:
                syl_num = 1

        return syl_num, syl_stress

    def gen_syllable_info(self, sentence):
        """Generates syllable information based on input sentence.
        Input should be list of words."""

        is_stress = False

        tot_syl = 0
        max_word = None
        max_syl = 0
        for word in sentence:
            syl_num, syl_stress = self.parse_word(word)

            # generate stressed syllables based on order
            syl_stress_order = []
            for syl in range(syl_num):
                syl_stress_order.append(int(is_stress))
                is_stress = not (is_stress)

            # if don't match then append both, saying it can be either stress
            if syl_stress_order != syl_stress and syl_stress:
                if syl_stress:
                    syl_stress = [syl_stress_order, syl_stress]
            else:
                syl_stress = [syl_stress_order]

            self.syllable_dict[word] = [syl_num, syl_stress]

            tot_syl += syl_num

            # keep track of longest word
            if syl_num > max_syl:
                max_word = word
                max_syl = syl_num

                # if tot_syl!=self.num_syllables:
                #    self.syllable_dict[max_word][0]+=1#self.num_syllables-tot_syl

    def gen_pos_info(self, sentence):
        """Generates parts of speech info based on input sequence.
        Populates pos_dict with this information"""

        # generate a clean list of words
        clean_words = [word.translate(None, string.punctuation).lower() for word in sentence]
        tags = self.pos_tag(clean_words)

        for i in range(len(sentence)):
            tag = tags[i][1][:2]
            word = sentence[i]
            if word == "i":
                # I often gets mislabelled
                tag = 'PR'
            try:
                self.pos_dict[word] = self.tag_dict[tag]
            except KeyError:
                print " ".join([word, tag])


def main():
    """Proprocessing tests and examples"""

    filename = 'data/best_sonnets.txt'
    data = dataHandler(filename, 1)
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

    # hyp_list=pyhyphen_setup()

    check = "And guard the shall find pleasing say".split(' ')
    for word in check:
        print data.parse_word(word)

    # data.gen_syllable_info(check)

    # print get_syllable_info("summer")

    pass


if __name__ == '__main__':
    main()
