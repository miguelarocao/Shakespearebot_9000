# HMM Class and methods

from preprocess import dataHandler
import numpy as np
import sys
from numpy import linalg as LA


class HMM:
    # HMM class

    def __init__(self, stanza="all", multi=""):
        # number of hidden states
        self.num_hidden = 30
        self.num_states = self.num_hidden + 2  # start and end state
        self.num_words = None
        self.start_idx = 0  # start state index
        self.end_idx = -1  # end state index
        # preprocess class
        self.myData = None
        self.train_data = None
        # to be filled in training
        self.A = None  # Transition Matrix: Rows->From & Columns->To
        self.O = None  # Observation Matrix: Rows->Observations, Columns->State
        self.threshold = 0.001  # threshold for fractional change in norm
        self.null_added = 1  # have nulls been appended?
        self.stanza = stanza  # stanza type

        # file saving
        self.A_file = "trained/A_" + self.stanza + "_" + str(self.num_hidden) + multi
        self.O_file = "trained/O_" + self.stanza + "_" + str(self.num_hidden) + multi
        self.word_file = "trained/word_" + self.stanza + "_" + str(self.num_hidden) + multi
        self.idx_file = "trained/idx_" + self.stanza + "_" + str(self.num_hidden) + multi
        print "Created HMM to be trained on " + stanza + " data."

    def load_data(self, filename):
        """Loads A and O previously trained on."""
        self.myData = dataHandler(filename, self.null_added, self.stanza)
        self.myData.gen_word_idx()
        self.num_words = self.myData.get_num_words()
        self.train_data = self.myData.get_all_data()

    def load_prev_trained(self):
        """Loads A and O previously trained on."""
        self.A = np.load(self.A_file + ".npy")
        self.O = np.load(self.O_file + ".npy")
        # self.myData.word_dict=np.load(self.word_file+".npy").item()
        # self.myData.idx_dict=np.load(self.idx_file+".npy").item()
        print "Training files succesfully loaded!"

    def train(self):
        """Trains the HMM using the loaded data"""

        # set seed
        np.random.seed(seed=1)

        # state transition matrix -> rows sum to 1
        # A is the transpose of what is presented in the notes
        # INCLUDES START AND END STATE: Appropriate transition probabilities set to 0
        self.A = np.zeros((self.num_states, self.num_states))
        for row in range(self.num_states):
            self.A[row, 1:] = np.random.dirichlet(np.ones(self.num_states - 1), size=1)
        # observation matrix (doesn't include start and end state)
        # INCLUDES START AND END STATE: Appropriate observations probabilities set to 0
        # Columns sum to 1
        self.O = np.zeros((self.num_words, self.num_states))
        self.O = np.transpose(self.O)
        for row in range(1, self.num_states - 1):
            self.O[row, :] = np.random.dirichlet(np.ones(self.num_words), size=1)

        # Last state -
        if self.null_added == 0:
            self.O[self.end_idx, :] = np.random.dirichlet(np.ones(self.num_words), size=1)
        else:
            self.O[self.end_idx, self.myData.get_word_idx("NULL")] = 1

        self.O = np.transpose(self.O)

        # set start values
        self.A[self.end_idx, :] = np.zeros(np.shape(self.A[self.end_idx, :]))
        # start can't go directly to end or to start again
        self.A[self.start_idx, :] += self.A[self.start_idx, self.end_idx] / (len(self.A[self.start_idx, :]) - 2)
        self.A[self.start_idx, self.end_idx] = 0
        self.A[self.start_idx, self.start_idx] = 0
        # end is guaranteed to stay in end state
        self.A[self.end_idx, self.end_idx] = 1

        # For testing
        A_test = self.A
        O_test = self.O

        A_norm_old = 0
        A_norm = LA.norm(self.A)
        O_norm_old = 0
        O_norm = LA.norm(self.O)
        count = 0
        while ((abs(A_norm - A_norm_old) / A_norm > self.threshold) or \
                       (abs(O_norm - O_norm_old) / O_norm > self.threshold)):
            A_norm_old = A_norm
            O_norm_old = O_norm
            sequence_no = 0
            A_n = np.zeros(np.shape(self.A))
            A_d = np.zeros(np.shape(self.A))
            O_n = np.zeros(np.shape(self.O))
            O_d = np.zeros(np.shape(self.O))

            for training_seq in self.train_data:
                alpha = np.zeros((self.num_states, len(training_seq) + 1))
                beta = np.zeros((self.num_states, len(training_seq) + 1))
                # expectation step
                self.e_step(alpha, beta, training_seq)
                # maximization step
                A_num, A_den, O_num, O_den = self.m_step(alpha, beta, training_seq)
                A_n += A_num
                A_d += A_den
                O_n += O_num
                O_d += O_den
                if sequence_no % 100 == 0:
                    print "count = " + str(count) + " sequence = " + str(sequence_no)
                sequence_no += 1

            self.A = self.get_division(A_n, A_d)
            self.O = self.get_division(O_n, O_d)
            A_norm = LA.norm(self.A)
            O_norm = LA.norm(self.O)
            A_test = self.A
            O_test = self.O
            print "count = " + str(count) + " A_norm = " + str(A_norm) + " A_norm_old = " + str(A_norm_old) + \
                  " O_norm = " + str(O_norm) + " O_norm_old = " + str(O_norm_old)
            count += 1

        print("Finished training")
        np.save(self.A_file, self.A)
        np.save(self.O_file, self.O)
        np.save(self.word_file, self.myData.word_dict)
        np.save(self.idx_file, self.myData.idx_dict)

    def e_step(self, alpha, beta, train):
        """Uses forward-backward approach to calculate expectation"""

        seq_len = len(train)

        # initialize alpha and beta -> based on notes
        # note that our sequence starts from 0 instead of 1,
        # so alpha and beta also shift by 1 correspondingly, with
        # alpha and beta starting from -1 and going till seq_len -1

        # for efficiency, train should be a sequence of indices of tokens

        # alpha(a, -1) = 1, if a = Start
        #               0, otherwise
        # alpha(a, 0) = P(train(0)|a) * P(a|Start)

        # forward initialisation
        alpha[self.start_idx, -1] = 1

        # forward
        for t in range(seq_len):
            x = self.myData.get_word_idx(train[t])
            for s in range(self.num_states):
                alpha[s, t] = self.O[x, s] * np.dot(alpha[:, t - 1], self.A[:, s])

            # normalize
            alpha[:, t] /= sum(alpha[:, t])

        # backwards initialisation
        # note that beta[seq_len -1] = beta[-2] is the final beta
        beta[:, seq_len - 1] = self.A[:, self.end_idx]

        # backwards
        for t in range(seq_len - 2, -2, -1):
            x = self.myData.get_word_idx(train[t + 1])
            for s in range(self.num_states):
                prod = np.multiply(self.A[s, :], self.O[x, :])
                beta[s, t] = np.dot(beta[:, t + 1], prod)

            # normalize
            beta[:, t] /= sum(beta[:, t])

        return alpha, beta

    def m_step(self, alpha, beta, train):

        A_num = np.zeros(np.shape(self.A))
        A_den = np.zeros(np.shape(self.A))
        O_num = np.zeros(np.shape(self.O))
        O_den = np.zeros(np.shape(self.O))

        double_marginal_den = np.zeros((len(train) + 1, 1))
        for j in range(-1, len(train)):
            double_marginal_den[j] = self.get_double_marginal_den(j, alpha, beta)

        triple_marginal_den = np.zeros((len(train) + 1, 1))
        for j in range(-1, len(train)):
            x = self.myData.get_word_idx(train[j])
            triple_marginal_den[j] = self.get_triple_marginal_den(j, alpha, beta, x)

        # update A
        for s_from in range(self.num_states - 1):  # from: skip end state
            # not skipping from state for verification
            for s_to in range(self.num_states):
                # compute transition for each state
                num_sum = 0  # numerator sum
                if s_to == 0:
                    den_sum = 0  # denominator sum
                for j in range(len(train)):
                    x = self.myData.get_word_idx(train[j])
                    num_sum += self.get_triple_marginal(j, alpha, beta, s_from, s_to, x, triple_marginal_den[j])
                    if s_to == 0:
                        den_sum += self.get_double_marginal(j - 1, alpha, beta, s_from, x, double_marginal_den[j - 1])

                A_num[s_from, s_to] = num_sum
                if s_to == 0:
                    A_den[s_from, :] = den_sum

        A_num[self.end_idx, self.end_idx] = 1
        A_den[self.end_idx, :] = 1

        # update O
        if self.null_added == 1:
            end_iter = self.num_states - 1
            O_num[self.myData.get_word_idx("NULL"), self.end_idx] = 1
            O_den[:, self.end_idx] = 1
        else:
            end_iter = self.num_states

        # for word in range(self.num_words):
        # no need for looping over words which don't occur in the
        # sequence - they will always have 0 numerators
        for word_id in range(len(train)):
            word = self.myData.get_word_idx(train[word_id])
            for state in range(1, end_iter):  # skip start state
                num_sum = 0  # numerator sum
                if word_id == 0:
                    den_sum = 0  # denominator sum
                for j in range(len(train)):
                    # could probably make this more efficient
                    x = self.myData.get_word_idx(train[j])
                    temp = self.get_double_marginal(j, alpha, beta, state, x, double_marginal_den[j])
                    if x == word:
                        num_sum += temp
                    if word_id == 0:
                        den_sum += temp
                    #                if den_sum==0:
                    #                    #Avoids division by 0
                    #                    den_sum=1
                O_num[word, state] = num_sum
                if word_id == 0:
                    O_den[:, state] = den_sum

        return A_num, A_den, O_num, O_den

    def get_double_marginal_den(self, j, alpha, beta):
        """Returns denominator for P(y_j=a,x_j).
        Equation (12). j >= -1."""

        # calculate denominator
        den = 0
        for s in range(self.num_states):  # from
            den += alpha[s, j] * beta[s, j]

        if den == 0:
            # to avoid division by 0
            den = 1

        # return denominator
        return den

    def get_double_marginal(self, j, alpha, beta, a, x, den):
        """Returns P(y_j=a,x_j). Equation (12). j >= -1."""

        # return probability
        return alpha[a, j] * beta[a, j] / den

    def get_triple_marginal_den(self, j, alpha, beta, x):
        """Returns denominator for P(y_j=b,y_(j-1)=a,x_j).
        Equation (13). j >= 0."""

        # calculate denominator
        den = 0
        for s1 in range(self.num_states):  # from
            for s2 in range(self.num_states):  # to
                den += alpha[s1, j - 1] * self.O[x, s2] * self.A[s1, s2] * beta[s2, j]

        if den == 0:
            # to avoid division by 0
            den = 1

        # return denominator
        return den

    def get_triple_marginal(self, j, alpha, beta, a, b, x, den):
        """Returns P(y_j=b,y_(j-1)=a,x_j). Equation (13). j >= 0."""

        # return probability
        return alpha[a, j - 1] * self.O[x, b] * self.A[a, b] * beta[b, j] / den

    def get_division(self, M_num, M_den):
        res = np.copy(M_num / M_den)
        res = np.nan_to_num(res)
        return res

    def generate_lines(self):
        return self.myData.generate_lines(self.A, self.O)

    def generate_naive_lines(self):
        return self.myData.generate_naive_lines(self.A, self.O)

    @staticmethod
    def generate_poem(hmms, filename=None):
        """Generates poem based on hmms. Input can either be a single hmm or a
        dictionary of hmms with keys 'volta', 'quatrain' and 'couplet'."""
        poem = ""

        try:
            print "Generating poem for multi-stanza types.\n"

            # quatrain
            poem += hmms["quatrain"].generate_lines()
            # volta
            poem += hmms["volta"].generate_lines()
            # couplet
            poem += hmms["couplet"].generate_lines()
            if len(hmms) > 3:
                raise AssertionError("generate_poem(): invalid input!")

        except AttributeError:
            # single input
            print "Generating poem for stanza type \"" + hmms.stanza + "\""
            poem = hmms.generate_lines()

        if filename:
            f = open(filename, 'w')
            f.write(poem)
            f.close()
        else:
            print poem

    @staticmethod
    def generate_naive_poem(hmms, filename=None):
        """Generates poem based on hmms. Input can either be a single hmm or a
        dictionary of hmms with keys 'volta', 'quatrain' and 'couplet'."""
        poem = ""

        try:
            print "Generating naive poem for multi-stanza types.\n"

            # quatrain
            poem += hmms["quatrain"].generate_naive_lines()
            # volta
            poem += hmms["volta"].generate_naive_lines()
            # couplet
            poem += hmms["couplet"].generate_naive_lines()
            if len(hmms) > 3:
                raise AssertionError("generate_poem(): invalid input!")

        except AttributeError:
            # single input
            print "Generating poem for stanza type \"" + hmms.stanza + "\""
            poem = hmms.generate_lines()

        if filename:
            f = open(filename, 'w')
            f.write(poem)
            f.close()
        else:
            print poem


def main():

    if len(sys.argv) > 2:
        print "usage: hmm.py <number of poems to generate>"
        return

    num_poems = int(sys.argv[1])

    filenames = ['data/shakespeare.txt', 'data/spenser.txt', 'data/best_sonnets.txt']

    qHMM = HMM("quatrain", "_" + str(len(filenames)))
    qHMM.load_data(filenames)
    qHMM.load_prev_trained()
    # qHMM.train()

    vHMM = HMM("volta", "_" + str(len(filenames)))
    vHMM.load_data(filenames)
    vHMM.load_prev_trained()
    # vHMM.train()

    cHMM = HMM("couplet", "_" + str(len(filenames)))
    cHMM.load_data(filenames)
    cHMM.load_prev_trained()
    # cHMM.train()

    poem_dict = {"quatrain": qHMM, "volta": vHMM, "couplet": cHMM}

    print "Generating %s poems..." % num_poems

    for i in range(num_poems):
        HMM.generate_poem(poem_dict)
        print ""

        # HMM.generate_naive_poem(poem_dict)

    #    myHMM=HMM()
    #    myHMM.load_data(filenames)
    #    myHMM.train()
    #
    #    HMM.generate_poem(myHMM)

    pass


if __name__ == '__main__':
    main()
