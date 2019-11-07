#Fabian Kaechele  06.11.2019

"""    This Fileis needed to simulate from copulas.

   """
from __future__ import division
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fmin
import sys
import math
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF




class Copula():


    def __init__(self, tau,pr,df, family):

        copula_family = ['clayton', 'frank', 'gumbel', 'normal','t','independent']
        if family not in copula_family:
            raise ValueError('The family should be clayton or frank or gumbel or normal')


        self.family = family
        self.tau = tau
        self.pr = pr
        self.df=df
        self._get_parameter()

        # set U and V to none
        self.U = None
        self.V = None

    def _get_parameter(self):
        """ calculate the parameter (theta) of copula
        """

        if self.family == 'clayton':
            self.theta = 2 * self.tau / (1 - self.tau)


        elif self.family == 'frank':
            self.theta = -fmin(self._frank_fun, -5, disp=False)[0]

        elif self.family == 'gumbel':
            self.theta = 1 / (1 - self.tau)



    def generate_uv(self, U,W,n=1000):
        """
        Generate random variables (u,v)

        Input:
            n:        Points to be generated

        Output:
            U and V:  copula

        """
        U=np.where(U==1,0.999999,U)
        U = np.where(U == 0, 0.00001, U)
        W=np.where(W==1,0.999999,W)
        W = np.where(W == 0, 0.00001, W)
        # = np.random.uniform(size=n)
        #W = np.random.uniform(size=n)

        # CLAYTON copula
        if self.family == 'clayton':

            if self.theta <= -1:
                raise ValueError('the parameter for clayton copula should be more than -1')
            elif self.theta == 0:
                raise ValueError('The parameter for clayton copula should not be 0')

            if abs(self.theta) < sys.float_info.epsilon:
                V = W
            else:
                V = U * (W ** (-self.theta / (1 + self.theta)) - 1 + U ** self.theta) ** (-1 / self.theta)

        # FRANK copula
        elif self.family == 'frank':

            if self.theta == 0:
                raise ValueError('The parameter for frank copula should not be 0')

            if abs(self.theta) > np.log(sys.float_info.max):
                V = (U < 0) + np.sign(self.theta) * U
            elif abs(self.theta) > np.sqrt(sys.float_info.epsilon):
                V = -np.log((np.exp(-self.theta * U) * (1 - W) / W + np.exp(-self.theta)
                             ) / (1 + np.exp(-self.theta * U) * (1 - W) / W)) / self.theta
            else:
                V = W

            # Gaus copula
        elif self.family == 'normal':
            V = stats.norm.cdf(stats.norm.ppf(W)*math.sqrt(1-self.pr**2)+self.pr*stats.norm.ppf(U))
            #print(pearsonr(U,V))

            # T copula
        elif self.family == 't':
            V = stats.t.cdf((stats.t.ppf(W,self.df+1) * np.sqrt(((self.df+(stats.t.ppf(U,self.df)**2))*(1 - self.pr ** 2))/(self.df+1)))+self.pr*stats.t.ppf(U,self.df),self.df)
            #print(pearsonr(U, V))

            # non copula
        elif self.family == 'non':
            alpha = 0.1
            beta = 0.9

            #V = (alpha*U**(-alpha*self.theta-1)/(W-(1-alpha)/U)+1-U**(-alpha*self.theta))**(1/(-beta*self.theta))
            # print(pearsonr(U, V))

        # GUMBEL copula
        elif self.family == 'gumbel':
            if self.theta <= 1:
                raise ValueError('the parameter for GUMBEL copula should be greater than 1')
            if self.theta < 1 + sys.float_info.epsilon:
                V = W
            else:

                u = U
                w = W
                w1 = np.random.uniform(size=n)
                w2 = np.random.uniform(size=n)
                u = (u - 0.5) * np.pi
                u2 = u + np.pi / 2;
                e = -np.log(w)
                t = np.cos(u - u2 / self.theta) / e
                gamma = (np.sin(u2 / self.theta) / t) ** (1 / self.theta) * t / np.cos(u)
                s1 = (-np.log(w1)) ** (1 / self.theta) / gamma
                s2 = (-np.log(w2)) ** (1 / self.theta) / gamma
                U = np.array(np.exp(-s1))
                V = np.array(np.exp(-s2))

        # Independent Copula
        elif self.family == 'independent':
            V = W



        self.U = U
        self.V = V
        return  U, V





    def _integrand_debye(self, t):
        """
        Integrand for the first order debye function
        """
        return t / (np.exp(t) - 1)

    def _debye(self, alpha):
        """
        First order Debye function
        """
        return quad(self._integrand_debye, sys.float_info.epsilon, alpha)[0] / alpha

    def _frank_fun(self, alpha):
        """
        optimization of this function will give the parameter for the frank copula
        """
        diff = (1 - self.tau) / 4.0 - (self._debye(-alpha) - 1) / alpha
        return diff ** 2

    def marginals(self,n):
        """
         calculate uniform distributed marginals out of other distributed marginals
         """
        a = np.random.normal(size=n)
        b = np.random.normal(size=n)
        e = ECDF(a)
        e2 = ECDF(b)
        u = e(a)
        w = e2(b)
        return u, w






