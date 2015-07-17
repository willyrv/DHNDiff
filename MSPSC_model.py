# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 23:31:19 2015

@author: willy
"""

import numpy as np
from scipy.integrate import quad
import scipy
import os
import matplotlib.pyplot as plt
from scipy.stats import kstest
from scipy.stats import chisquare

from mpl_toolkits.mplot3d.axes3d import Axes3D


class MSPSCModel:
    """
    This class allows to compute the pdf, the cdf as well as the number of
    differences under the Multiple Step Population Size Model. 
    It needs the values of t_k (the moments when population sizes change)
    the values of \lambda_k (factors of population size chances or the IICR)
    and \theta (the scaled mutation rate per base multiplied by the length
    of the sequence)
    """
    def __init__(self, t_k_values, lambda_k_values, theta):
        self.t_k_values = t_k_values
        self.lambda_k_values = lambda_k_values
        self.theta = theta
        
    def pdf(self, t, tuple_of_params=0):
        if tuple_of_params != 0:
            t_list = tuple_of_params[0]
            lambda_list = tuple_of_params[1]
        else:
            t_list = self.t_k_values
            lambda_list = self.lambda_k_values
        # find the term_n    
        term_n = len(t_list) - len(np.where(t_list-t>0)[0]) - 1
        exponent = -sum(np.true_divide(t_list[1:term_n+1]-t_list[:term_n], lambda_list[:term_n]))
        exponent -= np.true_divide(t-t_list[term_n], lambda_list[term_n])
        return np.true_divide(np.exp(exponent), lambda_list[term_n])

    def cdf(self, t, tuple_of_params):
        t_list = tuple_of_params[0]
        lambda_list = tuple_of_params[1]
        # find the term_n    
        term_n = len(t_list) - len(np.where(t_list-t>0)[0]) - 1
        exponent = -np.sum(np.true_divide(t_list[1:term_n+1]-t_list[:term_n], lambda_list[:term_n]))
        exponent -= np.true_divide(t-t_list[term_n], lambda_list[term_n])
        return 1-np.exp(exponent)

    def compute_factors_vector(self, t_list, lambda_list):
        # Computes the factor that will multiply the integral I_k in 
        # de developpment of the integral as a sum of integrals
        t_k = np.array(t_list)
        lambda_k = np.array(lambda_list)
        temp_vector = np.true_divide(t_k[1:]-t_k[:-1], lambda_k[:-1])
        temp_vector2 = np.true_divide(t_k, lambda_k)
        exponent = -np.cumsum(temp_vector) + temp_vector2[1:]
        temp_result = np.ones(len(t_k))
        temp_result[1:] = np.exp(exponent)
        return np.true_divide(temp_result, lambda_k)

    def compute_dict_integral(self, t_list, lambda_list, k_max, theta):
        # Computes all the values for the integrals and returns them in a dictionnary
        dict_integrals = {}
        # Compute the all the integrals for every interval [t_n, t_n+1]
        for i in range(len(t_list)-1):
            c = 2*theta + np.true_divide(1, lambda_list[i])
            dict_integrals[(t_list[i], 0)] = np.true_divide(np.exp(-c*t_list[i])-np.exp(-c*t_list[i+1]),c)
            for k in range(1, k_max+1):
                # We use the recursive formula for finding the other values
                dict_integrals[t_list[i], k] = np.true_divide(t_list[i]**k*np.exp(-c*t_list[i])
                -t_list[i+1]**k*np.exp(-c*t_list[i+1])+k*dict_integrals[(t_list[i], k-1)],c)
        # Now we compute the value for the last intervall [t_n, +infinity]
        c = 2*theta + np.true_divide(1, lambda_list[-1])
        dict_integrals[(t_list[-1], 0)] = np.true_divide(np.exp(-c*t_list[-1]),c)
        for k in range(1, k_max+1):
            dict_integrals[t_list[-1], k] = np.true_divide(t_list[-1]**k*np.exp(-c*t_list[-1])
                +k*dict_integrals[(t_list[-1], k-1)],c)
        return dict_integrals

    def function_F(self,t_list, lambda_list, k_max, theta):
        temp_F = np.zeros(k_max+1)
        factors = self.compute_factors_vector(t_list, lambda_list)
        dict_integrals = self.compute_dict_integral(t_list, lambda_list, k_max, theta)
        for k in range(k_max+1):
            integrals_list = np.array([dict_integrals[(i, k)] for i in t_list])
            temp_F[k] = np.true_divide(np.power(2*theta, k)*np.dot(factors, integrals_list), np.math.factorial(k))
        return temp_F

    def integrand_prob_Nk(self, t, k, theta, density_f, tuple_of_params):
        return np.exp(-2*theta*t)*np.power(t,k)*density_f(t, tuple_of_params)

    def prob_Nk(self, k, theta, density_f, tuple_of_params):
        # Here the parameters for the density f are passed in a tuple
        integral = quad(self.integrand_prob_Nk, 0, float('inf'), args=(k, theta, density_f, tuple_of_params))[0]
        return np.true_divide(np.power(2*theta,k)*integral, scipy.math.factorial(k))
    
    def log_likelihood(self, count_k, theta, density_f, tuple_of_params):
        # First we compute the vector P(N=k) for every k
        p_Nk = [self.prob_Nk(i, theta, density_f, tuple_of_params) for i in range(len(count_k))]
        return np.dot(count_k, np.log(p_Nk))
        
    ## This part is for making some test of the MLE strategy ####
    
    def log_likelihood_NMut_MPSC(self, count_k, theta, t_list, lambda_list):
        # We supose that t_list[0] > 1. We will add the value 0 at the begining of
        # t_list and the value 1 at the begining of lambda_list in order to start 
        # at the present. We do this because we assume that we always start at time
        # 0 with lambda=1
        
        # Verify that all times and lambda are positive and that times are increassing
        t_list = np.array(t_list)
        lambda_list = np.array(lambda_list)
        if sum(lambda_list>0)<len(lambda_list):
            return float('-inf')
        elif sum((t_list[1:]-t_list[:-1])>0) < (len(t_list)-1): # t_list is not increasing
            return float('-inf')
        elif min(t_list)<0:
            return float('-inf')
        else:
            t_k = np.array([0]+list(t_list))
            lambda_k = np.array([1]+list(lambda_list))
            prob_Nk = self.function_F(t_k, lambda_k, len(count_k)-1, theta)
            return np.dot(count_k, np.log(prob_Nk))
            
    def plot_log_likelihood_NMut_MPSC(self, n_obs, theta, t_list, lambda_list, pos_v1=0, pos_v2=False, domain_v1 = np.arange(0.02, 20, 0.01), domain_v2 = np.arange(0.02, 20, 0.01)):
        # Plot the likelihood of the Number of Mutations as a function of 1 or 2 
        # variables. The variable to take are population sizes at some time intervall
        # (i.e. positions of the lambda_list vector)
    
        '''    
        n_obs = 10000
        theta = 0.5
        t_list = np.array([0.01, 0.05,  0.1 ,  0.3 ,  0.5 ])
        lambda_list = np.array([2. ,  0.5,  0.1,  2. ,  1. ])
        pos_v1=0
        pos_v2=1
        domain_v1 = np.arange(0.2, 20, 0.1)
        domain_v2 = np.arange(0.2, 10, 0.1)
        '''
        # First we compute the theoretical pmf (we assume that no more than 100 
        # differences are present)
        prob_Nk = self.function_F(t_list, lambda_list, 100, theta)
        
        # Construct the observed dataset
        obs = prob_Nk * n_obs
        # top = sum(obs>1)
        count_k = obs[:sum(obs>1)]
    
        fig = plt.figure()
    
        if pos_v2:
            # We plot a surface
            X, Y = np.meshgrid(domain_v1, domain_v2)
            nb_of_rows = len(domain_v2)
            nb_of_columns = len(domain_v1)
            Z = np.zeros([nb_of_rows, nb_of_columns])
            
            for i in range(nb_of_rows):
                lambda_variable = [v for v in lambda_list]
                lambda_variable[pos_v2] = domain_v2[i]
                for j in range(nb_of_columns):
                    lambda_variable[pos_v1] = domain_v1[j]
                    Z[i][j] = self.log_likelihood_NMut_MPSC(count_k, theta, t_list, np.array(lambda_variable))
            
            ax = fig.add_subplot(1,1,1, projection='3d')
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
            #ax.plot_surface(X, Y, Z, rstride=4, cstride=4)
            #ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4)
            #ax.plot_surface(X, Y, Z)
    
        else:
            # We plot a function of one variable
            X = domain_v1
            Y = np.zeros(len(X))
            lambda_variable = np.array([v for v in lambda_list])
            for i in range(len(domain_v1)):
                lambda_variable[pos_v1] = domain_v1[i]
                Y[i] = self.log_likelihood_NMut_MPSC(count_k, theta, t_list, np.array(lambda_variable))
            
            ax = fig.add_subplot(1,1,1)
            ax.plot(X,Y)
        
        plt.show()
            
    def test_MLE_from_theory(self, n_obs, theta, t_list, lambda_list):
        # For a give picewise function lambda_k we can write the probability
        # of getting k differences (mutations) in a loci (ie P(N=k))
        # Here, from the theoretical distribution, we compute the data that we
        # should observe under the given parameters. Then, we use this data for
        # finding the lambda_k values using a MLE estrategy.
    
        # First we compute the theoretical pmf (we assume that no more than 100 
        # differences are present)
        prob_Nk = self.function_F(t_list, lambda_list, 100, theta)
        
        # Construct the observed dataset
        obs = np.round(prob_Nk * n_obs)
        # top = sum(obs>1)
        count_k = obs[:sum(obs>1)]
        
        # Define the objetive function (here, we used the right t values and we
        # estimate the lambda_list values)
        
        obj_f = lambda x: -self.log_likelihood_NMut_MPSC(count_k, theta, t_list, x)
        
        x0 = np.ones(len(t_list))
        
        res_basinh = scipy.optimize.basinhopping(obj_f, x0, niter=1000, T=2)
        res_NM = scipy.optimize.minimize(obj_f, x0, method='Nelder-Mead')
        res_Powell = scipy.optimize.minimize(obj_f, x0, method='Powell')
        res_CG = scipy.optimize.minimize(obj_f, x0, method='CG')
        res_BFGS = scipy.optimize.minimize(obj_f, x0, method='BFGS')
        #res_NewtonCG = scipy.optimize.minimize(obj_f, x0, method='Newton-CG')
        
        dict_result = {'ci':count_k, 'basinh': res_basinh, 'NM': res_NM, 'Powell': res_Powell,
        'CG': res_CG, 'BFGS':res_BFGS}
        
        return dict_result
        
    def MLE_SSPSC_NMut(self, count_k, theta):
        # Assuming a SSPSC model, estimate the parameters (alpha, T) by a maximum
        # likelihood approach
        
        obj_f = lambda x: -self.log_likelihood_NMut_MPSC(count_k, theta, np.array([x[0]]), np.array([x[1]]))
        
        x0 = np.array([1, 1])
        res_basinh = scipy.optimize.basinhopping(obj_f, x0)
        res_NM = scipy.optimize.minimize(obj_f, x0, method='Nelder-Mead')
        
        dict_result = {'basinh': res_basinh, 'NM': res_NM}
        
        return dict_result
    
    def test_MLE_SSPSC_NMut_theory(self, theta, alpha, T, n_obs=10000, max_ndif=100):
        # This is for testing the accuracy of the MLE strategy
        # The dataset is built from the theoretical distribution function
    
        # First we compute the theoretical pmf (we assume that no more than 100 
        # differences are present)
        prob_Nk = self.function_F(np.array([0, T]), np.array([1, alpha]), max_ndif, theta)
        
        # Construct the observed dataset
        obs = np.round(prob_Nk * n_obs)
        # top = sum(obs>1)
        count_k = obs[:sum(obs>1)]  
        
        return self.MLE_SSPSC_NMut(count_k, theta)
    
    def MLE_MSPSC_NMut(self, count_k, theta, t_list0, lambda_list0, 
                       fixed_T= False):
        # Assuming a MSPSC model (Multiple Step Population Size Change) model
        # we estimate the parameters (list_T, list_alpha) by a maximum
        # likelihood approach. Here we have as many values of list_T as
        # population size changes. The same is for list_lambda
        # Here, the values of t_list0 and lambda_list0 are starting points.
        
        if fixed_T:
            obj_f = lambda x: -self.log_likelihood_NMut_MPSC(count_k, theta, t_list0, np.array(x))
            x0 = lambda_list0
        else:
            obj_f = lambda x: -self.log_likelihood_NMut_MPSC(count_k, theta, np.array(x[:len(x)/2]), np.array(x[len(x)/2:]))
            x0 = list(t_list0) + list(lambda_list0)
        #obj_f = lambda x: -self.log_likelihood_NMut_MPSC(count_k, theta, np.array(fixed_t_list), np.array([x]))
        
        
        #x0 = list(lambda_list0)
        
        res_basinh = scipy.optimize.basinhopping(obj_f, x0)
        #res_NM = scipy.optimize.minimize(obj_f, x0, method='Nelder-Mead')
        
        #dict_result = {'basinh': res_basinh, 'NM': res_NM}
        dict_result = {'basinh': res_basinh}
        
        return dict_result    
        

    
##### This is for the ms commands

class MSTester:
    
    def create_ms_command(self, n_obs, t_list, lambda_list):
        # We assume that t_list[0] is always 0 and lambda_list[0] is always 1
        ms_command_base = 'ms 2 {} -T -L'.format(n_obs)
        demographic_events = ['-eN {} {}'.format(t_list[i], lambda_list[i]) for i in range(1, len(t_list))]
        return '{} {}'.format(ms_command_base, ' '.join(demographic_events))
    
    def create_ms_command_NMut(self, n_obs, theta, t_list, lambda_list):
        ms_command_base = 'ms 2 {} -t {}'.format(n_obs, theta)
        demographic_events = ['-eN {} {}'.format(t_list[i], lambda_list[i]) for i in range(1, len(t_list))]
        return '{} {}'.format(ms_command_base, ' '.join(demographic_events))

    def generate_T2_ms(self, ms_command, path2ms='./utils'):
        obs_text = os.popen(os.path.join(path2ms, ms_command)).read()
        obs_text = obs_text.split('time')[1:]
        obs = [float(i.split('\t')[1]) for i in obs_text]
        return obs

    def generate_NMut_ms(self, ms_command, path2ms='./utils'):
        obs_text = os.popen(os.path.join(path2ms, ms_command)).read()
        obs_text = obs_text.split('segsites: ')[1:]
        obs = [int(i.split('\n')[0]) for i in obs_text]
        return obs

    def compare_cdf_MPSC_MS(self, array_t, n_obs, t_list, lambda_list):
        # Given some history (t_list, lambda_list), we compare the theoretical 
        # cumulative distribution with the empirical distribution from MS
        # t_list[0] = 0 and lambda_list[0] = 1
    
        # Create a MSPSCModel object
        V_to_test = MSPSCModel(t_list, lambda_list, 1)
    
        # First we simulate the data
        t_list_ms = np.true_divide(t_list, 2)
        ms_command = self.create_ms_command(n_obs, t_list_ms, lambda_list)
        obs_T2 = self.generate_T2_ms(ms_command)
        #obs_T2 = np.true_divide(obs_T2, 2)
        obs_T2 = np.array(obs_T2)*2
        
        # We set the values of t that will be used for comparing    
        # Here we may need to extend the limit of array_t in order to get all
        # the observed values of T2 lower than array_t[-1]    
        delta = array_t[-1]-array_t[-2]
        #print max(obs_T2)
        array_t[-1] = max(array_t[-1], max(obs_T2)+delta)
        histogram = np.histogram(obs_T2, bins=array_t)
        emp_cdf = np.zeros(len(array_t))
        emp_cdf[1:] = np.cumsum(histogram[0])
        emp_cdf = np.true_divide(emp_cdf, n_obs)
        theor_cdf = np.array([V_to_test.cdf(i, (t_list, lambda_list)) for i in array_t])
        
        # Now we plot
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(array_t, theor_cdf, label='theory')
        ax.plot(array_t, emp_cdf, label='empirical')
        ax.set_ylim(0, 1.5)
        plt.legend()
        plt.show()
        t = (t_list, lambda_list)
        f_test = lambda at: np.array([V_to_test.cdf(i, t) for i in at])
        #f_test = lambda x : cdf_MPSC(x, t)
        print kstest(obs_T2, f_test)
        
        print 'Doing 100 times the KS test ...'
        rejections_count = 0
        for i in xrange(100):
            obs_T2 = self.generate_T2_ms(ms_command)
            obs_T2 = np.array(obs_T2)*2
            if kstest(obs_T2, f_test)[1]<0.05: rejections_count+=1
        print 'The number of rejections was {}'.format(rejections_count)
    
    def compare_cdf_NMut_MPSC(self, n_obs, theta, t_list, lambda_list, 
                              n_rep_chi2=100):
        # Compare the theoretical with the empirical distribution 
        # (from ms simulations)
        # of the number of differences
        msc = self.create_ms_command_NMut(n_obs, 2*theta, np.true_divide(np.array(t_list),2), lambda_list)
        obs = self.generate_NMut_ms(msc)
        
        # Make the histogram for the observed data    
        b = np.arange(-0.5, max(obs)+0.5, 1)
        h = np.histogram(obs, bins=b)[0]
        
        # Compute the theoretical distribution
        #prob_Nk = function_F(np.array(t_list)*2, lambda_list, max(obs), np.true_divide(theta,2))
        model = MSPSCModel(t_list, lambda_list, theta)
        prob_Nk = model.function_F(t_list, lambda_list, max(obs), theta)
    
        
        # Make the plot
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(obs, bins=b, color='g')
        y = np.array(prob_Nk)*n_obs
        ax.plot(b+0.5, y, 'bo')
        plt.show()
        
        # Now do a chisquare test
        top = sum(h>5)
        #top = 2
        emp_dist = np.zeros(top+1)
        emp_dist[:-1] = h[:top]
        emp_dist[-1] = n_obs - sum(emp_dist)
        
        theor_dist = np.zeros(top+1)
        theor_dist[:-1] = np.round(prob_Nk[:top]*n_obs)
        theor_dist[-1] = n_obs-sum(theor_dist)
    
        print 'Empirical and Theoretical distributions'
        print (emp_dist, theor_dist)
        print 'Chisquare test result'
        test_result = chisquare(emp_dist, theor_dist)
        print test_result
        
        print("Doing a chisquare test {} times and counting the number of\
                rejections...".format(n_rep_chi2))
        
        rejections_count = 0
        for i in xrange(n_rep_chi2):
            obs = self.generate_NMut_ms(msc)
            b = np.arange(-0.5, max(obs)+0.5, 1)
            h = np.histogram(obs, bins=b)[0]
            prob_Nk = model.function_F(t_list, lambda_list, max(obs), theta)
            
            top = sum(h>5)
            emp_dist = np.zeros(top+1)
            emp_dist[:-1] = h[:top]
            emp_dist[-1] = n_obs - sum(emp_dist)
            
            theor_dist = np.zeros(top+1)
            theor_dist[:-1] = np.round(prob_Nk[:top]*n_obs)
            theor_dist[-1] = n_obs-sum(theor_dist)        
            if chisquare(emp_dist, theor_dist)[1]<0.05: 
                rejections_count+=1
        
        print 'We rejected {} times'.format(rejections_count)
