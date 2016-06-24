__author__ = 'hafiz'
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model import BaseHMM
try:
   import cPickle as pickle
except:
   import pickle
class HMM(BaseHMM):
    '''
    A Continuous HMM - This is a base class implementation for HMMs with
    mixtures. A mixture is a weighted sum of several continuous distributions,
    which can therefore create a more flexible general PDF for each hidden state.

    This class can be derived, but should not be used directly. Deriving classes
    should generally only implement the PDF function of the mixtures.

    Model attributes:
    - n            number of hidden states
    - m            number of mixtures in each state (each 'symbol' like in the discrete case points to a mixture)
    - d            number of features (an observation can contain multiple features)
    - A            hidden states transition probability matrix ([NxN] numpy array)
    - pi           initial state's PMF ([N] numpy array).

    Additional attributes:
    - min_std      used to create a covariance prior to prevent the covariances matrices from underflowing
    - precision    a numpy element size denoting the precision
    - verbose      a flag for printing progress information, mainly when learning
    '''
    model = None # initialize model with none
    model_file = 'randomforest.pk1'
    activity_list = None #labels for feature vectors
    activity_set = None
    total_activity = None
    transition_prob = None

    def __init__(self,n,m,d=1,A=None,means=None,covars=None,w=None,pi=None,min_std=0.01,init_type='uniform',precision=np.double,verbose=False):
        '''
        Construct a new Continuous HMM.
        In order to initialize the model with custom parameters,
        pass values for (A,means,covars,w,pi), and set the init_type to 'user'.

        Normal initialization uses a uniform distribution for all probablities,
        and is not recommended.
        '''
        BaseHMM.__init__(self,n,m,precision,verbose)

        self.d = d
        self.A = A
        self.pi = pi
        f = open(self.model_file, "rb")
        self.reset(init_type=init_type)

    def save_model(self,model_cls,cls_name):
        with open(cls_name, 'wb') as fid:
            pickle.dump(model_cls, fid)

    def load_model(self,cls_name):
        with open(cls_name, 'rb') as fid:
            self.model = pickle.load(fid)
        return self.model



    def calculate_prior(self):
        self.total = None
    # def initial_transition_prob(self):
    #     for
    def reset(self,init_type='uniform'):
        if init_type == 'uniform':
            self.pi = np.ones( (self.n), dtype=self.precision) *(1.0/self.n)
            self.A = np.ones( (self.n,self.n), dtype=self.precision)*(1.0/self.n)
            self.B = np.ones( (self.n,self.m), dtype=self.precision)*(1.0/self.m)


    def mapB(self,observations):
        '''
        Required implementation for mapB. Refer to BaseHMM for more details.
        This method highly optimizes the running time, since all PDF calculations
        are done here once in each training iteration.

        - self.Bmix_map - computesand maps Bjm(Ot) to Bjm(t).
        '''
        self.B_map = np.zeros( (self.n,len(observations)), dtype=self.precision)
        self.Bmix_map = np.zeros( (self.n,self.m,len(observations)), dtype=self.precision)
        for j in xrange(self.n):
            for t in xrange(len(observations)):
                self.B_map[j][t] = self.calculate_bjt(j, t, observations[t])

    """
    b[j][Ot] = sum(1...M)w[j][m]*b[j][m][Ot]
    Returns b[j][Ot] based on the current model parameters (means, covars, weights) for the mixtures.
    - j - state
    - Ot - the current observation
    """
    def calculate_bjt(self,j,t,Ot):

        bjt = 0
        for m in xrange(self.m):
            self.Bmix_map[j][m][t] = self.pdf(Ot)  #send this feature vector to get the emission probability
            bjt += self.Bmix_map[j][m][t]
        return bjt

    def calculate_gammamix(self,alpha,beta,observations):
        '''
        Calculates 'gamma_mix'.

        Gamma_mix is a (TxNxK) numpy array, where gamma_mix[t][i][m] = the probability of being
        in state 'i' at time 't' with mixture 'm' given the full observation sequence.
        '''
        gamma_mix = np.zeros((len(observations),self.n,self.m),dtype=self.precision)

        for t in xrange(len(observations)):
            for j in xrange(self.n):
                for m in xrange(self.m):
                    alphabeta = 0.0
                    for jj in xrange(self.n):
                        alphabeta += alpha[t][jj]*beta[t][jj]
                    comp1 = (alpha[t][j]*beta[t][j]) / alphabeta

                    bjk_sum = 0.0
                    for k in xrange(self.m):
                        bjk_sum += (self.w[j][k]*self.Bmix_map[j][k][t])
                    comp2 = (self.w[j][m]*self.Bmix_map[j][m][t])/bjk_sum

                    gamma_mix[t][j][m] = comp1*comp2

        return gamma_mix

    def updatemodel(self,new_model):
        '''
        Required extension of _updatemodel. Adds 'w', 'means', 'covars',
        which holds the in-state information. Specfically, the parameters
        of the different mixtures.
        '''
        BaseHMM.updatemodel(self,new_model)
        self.w = new_model['w']
        self.means = new_model['means']
        self.covars = new_model['covars']

    def calculate_stats(self,observations):
        '''
        Extension of the original method so that it includes the computation
        of 'gamma_mix' stat.
        '''
        stats = BaseHMM.calculate_stats(self,observations) #@UndefinedVariable
        stats['gamma_mix'] = self.calculate_gammamix(stats['alpha'],stats['beta'],observations)

        return stats

    def reestimate(self,stats,observations):
        '''
        Required extension of _reestimate.
        Adds a re-estimation of the mixture parameters 'w', 'means', 'covars'.
        '''
        # re-estimate A, pi
        new_model = BaseHMM.reestimate(self,stats,observations) #@UndefinedVariable

        # re-estimate the continuous probability parameters of the mixtures
        w_new, means_new, covars_new = self._reestimateMixtures(observations,stats['gamma_mix'])

        new_model['w'] = w_new
        new_model['means'] = means_new
        new_model['covars'] = covars_new

        return new_model

    def reestimateMixtures(self,observations,gamma_mix):
        '''
        Helper method that performs the Baum-Welch 'M' step
        for the mixture parameters - 'w', 'means', 'covars'.
        '''
        w_new = np.zeros( (self.n,self.m), dtype=self.precision)
        means_new = np.zeros( (self.n,self.m,self.d), dtype=self.precision)
        covars_new = [[ np.matrix(np.zeros((self.d,self.d), dtype=self.precision)) for j in xrange(self.m)] for i in xrange(self.n)]

        for j in xrange(self.n):
            for m in xrange(self.m):
                numer = 0.0
                denom = 0.0
                for t in xrange(len(observations)):
                    for k in xrange(self.m):
                        denom += (self._eta(t,len(observations)-1)*gamma_mix[t][j][k])
                    numer += (self._eta(t,len(observations)-1)*gamma_mix[t][j][m])
                w_new[j][m] = numer/denom
            w_new[j] = self._normalize(w_new[j])

        for j in xrange(self.n):
            for m in xrange(self.m):
                numer = np.zeros( (self.d), dtype=self.precision)
                denom = np.zeros( (self.d), dtype=self.precision)
                for t in xrange(len(observations)):
                    numer += (self._eta(t,len(observations)-1)*gamma_mix[t][j][m]*observations[t])
                    denom += (self._eta(t,len(observations)-1)*gamma_mix[t][j][m])
                means_new[j][m] = numer/denom

        cov_prior = [[ np.matrix(self.min_std*np.eye((self.d), dtype=self.precision)) for j in xrange(self.m)] for i in xrange(self.n)]
        for j in xrange(self.n):
            for m in xrange(self.m):
                numer = np.matrix(np.zeros( (self.d,self.d), dtype=self.precision))
                denom = np.matrix(np.zeros( (self.d,self.d), dtype=self.precision))
                for t in xrange(len(observations)):
                    vector_as_mat = np.matrix( (observations[t]-self.means[j][m]), dtype=self.precision )
                    numer += (self._eta(t,len(observations)-1)*gamma_mix[t][j][m]*np.dot( vector_as_mat.T, vector_as_mat))
                    denom += (self._eta(t,len(observations)-1)*gamma_mix[t][j][m])
                covars_new[j][m] = numer/denom
                covars_new[j][m] = covars_new[j][m] + cov_prior[j][m]

        return w_new, means_new, covars_new

    def normalize(self, arr):
        '''
        Helper method to normalize probabilities, so that
        they all sum to '1'
        '''
        summ = np.sum(arr)
        for i in xrange(len(arr)):
            arr[i] = (arr[i]/summ)
        return arr


    def pdf(self,feature_vector):
        '''
        This is the specific Probability Distribution Function that will be used in each
        feature vector component.
        '''
        rf = self.load_model()
        rf = RandomForestClassifier(n_estimators=100)
        emission_prob = rf.predict_proba(feature_vector)
        return emission_prob
