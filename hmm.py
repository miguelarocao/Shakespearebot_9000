#HMM Class and methods

from preprocess import dataHandler
import numpy as np
import sys

class HMM:
    #HMM class

    def __init__(self):
        #number of hidden states
        self.num_hidden=4
        self.num_states=self.num_hidden+2 #start and end state
        self.num_words=None
        self.start_idx=0 #start state index
        self.end_idx=-1 #end state index
        #preprocess class
        self.myData=None
        self.train_data=None
        #to be filled in training
        self.A=None #Transition Matrix: Rows->From & Columns->To
        self.O=None #Observation Matrix: Rows->Observations, Columns->State

    def load_data(self,filename):
        self.myData=dataHandler(filename)
        self.myData.gen_word_idx()
        self.num_words=self.myData.get_num_words()
        self.train_data=self.myData.get_all_data()

    def train(self):
        """Trains the HMM using the loaded data"""

        #state transition matrix -> rows sum to 1
        #A is the transpose of what is presented in the notes
        #INCLUDES START AND END STATE: Appropriate transition probabilities set to 0
        self.A=np.zeros((self.num_states,self.num_states))
        for row in range(self.num_states):
            self.A[row,1:]=np.random.dirichlet(np.ones(self.num_states-1),size=1)
        #observation matrix (doesn't include start and end state)
        #INCLUDES START AND END STATE: Appropriate observations probabilities set to 0
        #Columns sum to 1
        self.O=np.zeros((self.num_words,self.num_states))
        self.O=np.transpose(self.O)
        for row in range(1,self.num_states):
            self.O[row,:]=np.random.dirichlet(np.ones(self.num_words),size=1)
        self.O=np.transpose(self.O)

        #set start values
        self.A[self.end_idx,:]=np.zeros(np.shape(self.A[self.end_idx,:]))
        #start can't go directly to end or to start again
        self.A[self.start_idx,:]+=self.A[self.start_idx,self.end_idx]/(len(self.A[self.start_idx,:])-2)
        self.A[self.start_idx,self.end_idx]=0
        self.A[self.start_idx,self.start_idx]=0
        #end is guaranteed to stay in end state
        self.A[self.end_idx,self.end_idx]=1

        #For checking
        A_test = self.A
        O_test = self.O

        #repeat until convergence, single step for now
        #TO DO: Add while loop
        for training_seq in self.train_data:
            alpha=np.zeros((self.num_states,len(training_seq)+1))
            beta=np.zeros((self.num_states,len(training_seq)+1))
            #expectation step
            self.e_step(alpha,beta,training_seq)
            #maximization step
            A,O=self.m_step(alpha,beta,training_seq)
            print A
            print O
            break

    def e_step(self,alpha,beta,train):
        """Uses forward-backward approach to calculate expectation"""

        seq_len=len(train)

        #initialize alpha and beta -> based on notes
        #note that our sequence starts from 0 instead of 1,
        #so alpha and beta also shift by 1 correspondingly, with
        #alpha and beta starting from -1 and going till seq_len -1

        #for efficiency, train should be a sequence of indices of tokens

        #alpha(a, -1) = 1, if a = Start
        #               0, otherwise
        #alpha(a, 0) = P(train(0)|a) * P(a|Start)

        #forward initialisation
        alpha[self.start_idx, -1] = 1

#        #for testing - this should be the same as alpha[:, 0]
#        alpha_0 = np.zeros(np.shape(alpha[:, 0]))
#
#        x = self.myData.get_word_idx(train[0])
#        for s in range(self.num_states):
#            alpha_0[s] = self.O[x, s] * self.A[self.start_idx, s]
#
#        #normalize
#        alpha_0[:]/=sum(alpha_0[:])

        #forward
        for t in range(seq_len):
            x = self.myData.get_word_idx(train[t])
            for s in range(self.num_states):
                alpha[s,t] = self.O[x, s] * np.dot(alpha[:,t-1],self.A[:,s])

            #normalize
            alpha[:,t]/=sum(alpha[:,t])

        #backwards initialisation
        #note that beta[seq_len -1] = beta[-2] is the final beta
        beta[:, seq_len -1] = self.A[:,self.end_idx]

        #backwards
        for t in range(seq_len-2,-2,-1):
            x=self.myData.get_word_idx(train[t+1])
            for s in range(self.num_states):
                prod=np.multiply(self.A[s,:],self.O[x,:])
                beta[s,t]=np.dot(beta[:,t+1],prod)

            #normalize
            beta[:,t]/=sum(beta[:,t])

        return alpha,beta

    def m_step(self,alpha,beta,train):

        A=np.zeros(np.shape(self.A))
        O=np.zeros(np.shape(self.O))

        #update A
        for s_from in range(self.num_states-1): #from: skip end state

            #for s_to in range(1,self.num_states): #to: #skip from state
            #not skipping from state for verification
            for s_to in range(self.num_states):
                #compute transition for each state
                num_sum=0 #numerator sum
                den_sum=0 #denominator sum
                for j in range(len(train)):
                    x=self.myData.get_word_idx(train[j])
                    num_sum+=self.get_triple_marginal(j,alpha,beta,s_from,s_to,x)
                    den_sum+=self.get_double_marginal(j-1,alpha,beta,s_from,x)
                if num_sum==0 and den_sum==0:
                    #TO DO: Double check this
                    #Avoids division by 0
                    den_sum=1
                A[s_from,s_to]=num_sum/den_sum

        A[self.end_idx, self.end_idx] = 1

        #update O
        for word in range(self.num_words):
            for state in range(1,self.num_states-1): #skip start and end state
                num_sum=0 #numerator sum
                den_sum=0 #denominator sum
                for j in range(len(train)):
                    #could probably make this more efficient
                    x=self.myData.get_word_idx(train[j])
                    temp=self.get_double_marginal(j,alpha,beta,state,x)
                    if x==word:
                        num_sum+=temp
                    den_sum+=temp
                if den_sum==0:
                    #Avoids division by 0
                    den_sum=1
                O[word,state]=num_sum/den_sum

        return A,O

    def get_double_marginal(self,j,alpha,beta,a,x):
        """Returns P(y_j=a,x_j). Equation (12). j >= -1."""

        #calculate denominator
        den=0
        for s in range(self.num_states): #from
            den+=alpha[s,j]*beta[s,j]

        if den==0:
            #to avoid division by 0
            den=1

        #return probability
        return alpha[a,j]*beta[a,j]/den

    def get_triple_marginal(self,j,alpha,beta,a,b,x):
        """Returns P(y_j=b,y_(j-1)=a,x_j). Equation (13). j >= 0."""

        #calculate denominator
        den=0
        for s1 in range(self.num_states): #from
            for s2 in range(self.num_states): #to
                den+=alpha[s1,j-1]*self.O[x,b]*self.A[s1,s2]*beta[s2,j]

        if den==0:
            #to avoid division by 0
            den=1

        #return probability
        return alpha[a,j-1]*self.O[x,b]*self.A[a,b]*beta[b,j]/den


def main():

    filename='data/shakespeare.txt'

    myHmm=HMM()
    myHmm.load_data(filename)
    myHmm.train()

    pass

if __name__ == '__main__':
    main()
