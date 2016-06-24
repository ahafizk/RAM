__author__ = 'hafiz'
import numpy as np
from BaseHMM import *
from subactivities import *
from random import randint
from os.path import exists
import os
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score as score
from sklearn.externals import joblib 

class ARClassfifier(BaseHMM):
    '''
    A Discrete HMM - The most basic implementation of a Hidden Markov Model,
    where each hidden state uses a discrete probability distribution for
    the physical observations.

    Model attributes:
    - n            number of hidden states
    - m            number of observable symbols
    - A            hidden states transition probability matrix ([NxN] numpy array)
    - B            PMFs denoting each state's distribution ([NxM] numpy array)
    - pi           initial state's PMF ([N] numpy array).

    Additional attributes:
    - precision    a numpy element size denoting the precision
    - verbose      a flag for printing progress information, mainly when learning
    '''
    
    def __init__(self,n,m,A=None,B=None,pi=None,init_type='uniform',precision=np.double,verbose=False):
        '''
        Construct a new Discrete HMM.
        In order to initialize the model with custom parameters,
        pass values for (A,B,pi), and set the init_type to 'user'.

        Normal initialization uses a uniform distribution for all probablities,
        and is not recommended.
        '''
        BaseHMM.__init__(self,n,m,precision,verbose) #@UndefinedVariable

        self.A = A
        self.pi = pi
        self.B = B

        self.reset(init_type=init_type)

    def reset(self,init_type='uniform'):
        '''
        If required, initalize the model parameters according the selected policy
        '''
        if init_type == 'uniform':
            self.pi = np.ones( (self.n), dtype=self.precision) *(1.0/self.n)
            self.A = np.ones( (self.n,self.n), dtype=self.precision)*(1.0/self.n)
            self.B = np.ones( (self.n,self.m), dtype=self.precision)*(1.0/self.m)



    def mapB(self,observations,labels=None):
        '''
        This function should implement according to your training data
        '''
        length = len(observations)
        self.B_map = np.zeros( (self.n,length), dtype=self.precision)
        if labels is not None:

            for t in xrange(length):
                self.B_map[labels[t]][t] = self.B[labels[t]][observations[t]]
        else:
            #print 'testing'
            for j in xrange(self.n):
                for t in xrange(length):
                    self.B_map[j][t] = self.B[j][observations[t]]
        #print '______mapB____'
        #print self.B_map
        #print '______mapB____'
    def updatemodel(self,new_model):
        '''
        Required extension of _updatemodel. Adds 'B', which holds
        the in-state information. Specfically, the different PMFs.
        '''
        BaseHMM.updatemodel(self,new_model) #@UndefinedVariable

        self.B = new_model['B']

    def reestimate(self,stats,observations):
        '''
        Required extension of _reestimate.
        Adds a re-estimation of the model parameter 'B'.
        '''
        # re-estimate A, pi
        new_model = BaseHMM.reestimate(self,stats,observations) #@UndefinedVariable

        # re-estimate the discrete probability of the observable symbols
        B_new = self.reestimateB(observations,stats['gamma'])

        new_model['B'] = B_new

        return new_model

    def reestimateB(self,observations,gamma):
        '''
        Helper method that performs the Baum-Welch 'M' step
        for the matrix 'B'.
        '''
        # TBD: determine how to include eta() weighing
        B_new = np.zeros( (self.n,self.m) ,dtype=self.precision)

        for j in xrange(self.n):
            for k in xrange(self.m):
                numer = 0.0
                denom = 0.0
                for t in xrange(len(observations)):
                    if observations[t] == k:
                        numer += gamma[t][j]
                    denom += gamma[t][j]
                B_new[j][k] = numer/denom

        return B_new


#def get_sample_set(size=20, loops=50):
def get_sample_set(size=5, loops=10):
    
    activities = get_activities()
    subactivities = get_subactivities()
    mapping = get_mapping()

    sub_states = []
    act_labels = []
    
    for loop in range(loops):
      
       # Move next line inside the next for loop to increase
       # transition probabilities between activities
       x = randint(0, len(activities) - 1)
       for y in range(size):
           #x = randint(0, len(activities) - 1)
           activity = mapping[activities[x]]
           #print activity
           act_labels.append(x)
           s = activity[ randint(0, len(activity) - 1) ]
           sub_states.append(subactivities.index(s))

    return np.array(sub_states), np.array(act_labels)

def get_stored_hmm():
    
    #hmm_params = pickle.load(filename)

    activities = get_activities()
    subactivities = get_subactivities()

    num_subactivities = len(subactivities)
    num_activities = len(activities)

    a = np.genfromtxt('HMM_model_A', delimiter=",")
    b = np.genfromtxt('HMM_model_B', delimiter=",")
    pi = np.genfromtxt('HMM_model_pi', delimiter=",")
    
    hmm = ARClassfifier(num_activities,num_subactivities,a,b,pi,init_type='uniform',precision=np.longdouble,verbose=False)
    hmm.A = a
    hmm.B = b
    hmm.pi = pi
    return hmm
    

def getHMM():
    
    sub_states, act_labels = get_sample_set();
    print "Subactivity sequence"
    print sub_states

    print "\nActivity labels"
    print act_labels

    activities = get_activities()
    subactivities = get_subactivities()

    num_subactivities = len(subactivities)
    num_activities = len(activities)
    
    atmp = np.random.random_sample((num_activities, num_subactivities))
    row_sums = atmp.sum(axis=1)
    a = atmp / row_sums[:, np.newaxis]

    btmp = np.random.random_sample((num_activities, num_subactivities))
    row_sums = btmp.sum(axis=1)
    b = btmp / row_sums[:, np.newaxis]

    pitmp = np.random.random_sample((num_activities))
    pi = pitmp / sum(pitmp)

    print "Creating and Training HMM ..."
    hmm = ARClassfifier(num_activities,num_subactivities,a,b,pi,init_type='uniform',precision=np.longdouble,verbose=False)
    hmm.train(sub_states,labels=act_labels,iterations=1000)

    return hmm


def main():
    
    print "Creating HMM..."
    hmm = getHMM()
    print "HMM Created successfully"

    np.savetxt('HMM_model_A', hmm.A, delimiter=',') 
    np.savetxt('HMM_model_B', hmm.B, delimiter=',') 
    np.savetxt('HMM_model_pi', hmm.pi, delimiter=',') 

    #x,y = get_sample_set(8,8)
    ##x= np.array([0,0,0,1,1,1,2,2,2,3,3,3])
    #y_ = hmm.decode(x)
    
    #fb = hmm.forwardbackward(x)
    #print "-----------------", fb
    
    ### Print forward probabilities of the 4 states
    ##for z in range(1000):
        ##x,y = get_sample_set(randint(1,5),randint(1,5))
        ##alpha = hmm.calculate_alpha(x)
        ##alpha_normalized = alpha.astype('float') / alpha.sum(axis=1)[:, np.newaxis]
        ##print alpha_normalized[-1]

    #print
    #print "Pi",hmm.pi
    #print "A",hmm.A
    #print "B", hmm.B
    #print 
    #print 'Actual labels:'
    #print y
    #print 'Predicted labels'
    #print y_
    #cm = confusion_matrix(y, y_)
    #cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #print cm_normalized
    #print score(y,y_)*100

if __name__=='__main__':
    # obj = ARClassfifier(2,3)
    main()

