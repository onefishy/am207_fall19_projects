import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal,multinomial
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import math
from datetime import datetime, timedelta

from rbf_keras.rbflayer import RBFLayer, InitCentersRandom
from rbf_keras.kmeans_initializer import InitCentersKMeans
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop

import keras
import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PATH_RBF_PFM="rbf_models/price_forecast_model/"


class DataHandlerPriceForcastingModel:

    def __init__(self,source_csv_file,t_in,t_out,training_percent):

        self.data = pd.read_csv(source_csv_file, header=None)
        self.config_data()


        self.t_in=t_in
        self.t_out=t_out
        self.training_percent=training_percent



        self.train_test_split_daily()






    def config_data(self):
        self.data.columns = ["time", "open", "high", "low", "close", "volume"]
        self.data =  self.data.set_index("time")
        self.data =  self.data.sort_index()
        self.data.index = pd.to_datetime( self.data.index)

        days_in_sample = [date.date() for date in self.data.index]
        days_in_sample = np.unique(np.array(days_in_sample))
        days_in_sample = [day for day in days_in_sample if self.data[
            [True if date.date() == days_in_sample[0] else False for date in self.data.index]].index.min().hour < 10]

        sample_df = self.data[[True if date.date() in days_in_sample else False for date in self.data.index]]

        self.data=sample_df

    def train_test_split_daily(self):
        days_in_sample = [date.date() for date in self.data.index]
        days_in_sample = np.unique(np.array(days_in_sample))

        training_days = int(self.training_percent * len(days_in_sample))

        validating_days =training_days+ int((1 - self.training_percent) / 2 * len(days_in_sample))

        day_train_sample = days_in_sample[0:training_days]
        day_validation_sample = days_in_sample[training_days:validating_days]
        day_test_sample = days_in_sample[validating_days:]


        self.days_in_sample=days_in_sample
        self.day_train_sample=day_train_sample
        self.day_validation_sample=day_validation_sample
        self.day_test_sample=day_test_sample

    def get_sample_split_continuos_test(self):

        return self.get_sample_split_continous(self.day_test_sample)
    def get_sample_split_continous(self,sample_period):

        sample_df = self.data[[True if date.date() in sample_period else False for date in self.data.index]]
        hours_in_day=np.unique(np.array([date.hour for date in sample_df.index]))

        series_by_hour=[]
        for day in sample_period:

            temp_day=self.data[[True if date.date() == day else False for date in self.data.index]]

            for hour in hours_in_day:

                hour_df=temp_day[[True if date.hour == hour else False for date in temp_day.index]]
                # print(hour_df)
                series_by_hour.append(hour_df)

        hour_tranche=int((self.t_in+self.t_out).seconds/(60*60))
        hour_extended=timedelta(seconds=(self.t_in+self.t_out).seconds%(60*60))
        hour_remaining=timedelta(seconds=60*60-(self.t_in+self.t_out).seconds%(60*60))


        roll=True
        tranche_start=0
        while roll==True:

            if hour_extended.seconds>0:
                #ToDo: make a function to extract intraday data by number of continuos hours
                target_tranche=series_by_hour[tranche_start:hour_tranche+1]



    def get_sample_split_by_day_test(self):

        return self.get_sample_split_by_day(self.day_test_sample)

    def get_sample_split_by_day_train(self):

        return self.get_sample_split_by_day(self.day_train_sample)

    def get_sample_split_by_day_validate(self):

        return self.get_sample_split_by_day(self.day_validation_sample)

    def get_sample_split_by_day(self,day_sample,delta_sample=timedelta(hours=1)):
        """
        This functions splists curve by only intraday data
        :return:
        """



        X_in = []
        Y_in = []

        X_out = []
        Y_out = []


        for day in day_sample:

            temp_data = self.data[[True if date.date() == day else False for date in self.data.index]]

            start_time = temp_data.index[0]
            end_time = start_time + self.t_in + self.t_out

            while end_time <= temp_data.index[-1]:
                temp_index_in = temp_data.loc[start_time:start_time + self.t_in].index
                temp_index_out = temp_data.loc[start_time:start_time + self.t_in + self.t_out].index

                X_in.append([(x - start_time).seconds / 60 for x in temp_index_in])
                X_out.append([(x - start_time).seconds / 60 for x in temp_index_out])

                Y_in.append(
                    100 * temp_data.loc[temp_index_in]["close"].values / temp_data.loc[temp_index_in]["close"].values[
                        0])
                Y_out.append(
                    100 * temp_data.loc[temp_index_out]["close"].values / temp_data.loc[temp_index_out]["close"].values[
                        0])

                start_time = start_time+delta_sample
                end_time = start_time + self.t_in + self.t_out

        return [X_in,Y_in,X_out,Y_out]




class ReducedRank():


    def __init__(self,Y,X,t_max,q,G,name="",t_in=timedelta(0),initialize_parameters=True):

        self.name=name

        self.Y=Y
        self.X=X
        self.t_max=t_max
        self.map = interp1d([0, t_max.seconds / 60], [0, 1])
        # self.map = interp1d([0, np.array(X).flatten()[-1]], [0, 1])

        self.q = q
        self.G = G
        self.h = q

        #knots
        n_knots = q - 4 #cubic polinomial
        self.knots=[i/(1+n_knots) for i in range(1,n_knots+1)]
        self.S = [self.spline_basis(self.map(s)) for s in X]
        self.spline_time=[self.map(s)for s in X]
        self.curves=len(self.S)



        if initialize_parameters==True:
            self.intialize_parameters()
        self.are_curves_classified = False
        self.curves_classified=[[] for i in range(len(X))]
        self.coefficients_representation = [[] for i in range(len(X))]

        self.t_in=t_in
        if t_in.seconds>0:
            X_in = [np.array(x)[np.array(x) <= t_in.seconds / 60] for x in X]
            X_out=[np.array(x)[np.array(x) > t_in.seconds / 60] for x in X]
            self.S_in=[self.spline_basis(self.map(s)) for s in X_in]
            self.S_out=[self.spline_basis(self.map(s)) for s in X_out]

            self.spline_time_in = [self.map(s) for s in X_in]

            self.curves_classified_in = [[] for i in range(len(X_in))]


    def likelihood(self):
      """
      We need the hidden value of  Z to compute the loglikelihhod
      https://pdfs.semanticscholar.org/817b/0871a996c1164dbe3dcd3e4054f7ce69dc3d.pdf?_ga=2.130886031.1404632557.1573172925-616252560.1572394910
      equation (5)
      :return:
      """



    def functional_representation(self,k,S_i):
        [m, A, c, Q, sigma, k_, pi] = self.unroll_parameters()
        c=self.parameters["C"][:,k]
        y=np.matmul(S_i, m + np.matmul(A, c))
        return y

    def get_coefficients_representaiton(self,k,i):
        [m, A, c, Q, sigma, k_, pi] = self.unroll_parameters()
        c = self.parameters["C"][:, k]

        S_i=self.S[i]
        Y_i=self.Y[i]


        coefficients =  m + np.matmul(A, c)+self.d_i_E_step(S_i,Y_i)
        return coefficients


    def spline_basis(self,t):
        """

        :param t:
        :param tau:
        :return:
        """
        knots=self.knots
        S = np.zeros((len(t), 4 + len(knots)))

        for n_row, t_value in enumerate(t):
            basis = [1, t_value, t_value ** 2, t_value ** 3]
            for knot in knots:
                # epsilon=t[(knot+1)*base_percentile]
                basis.append(max((t_value - knot) ** 3, 0))

            S[n_row, :] = np.array(basis)

        return S


    def weighted_curves_residuals(self):

        [m, A, c, Q, sigma, k, pi] = self.unroll_parameters()

        residuals=[]
        Y=self.Y
        for counter,S_i in enumerate(self.S):
            res=Y[counter]-self.expected_functional_representation(S_i)
            residuals.extend(res)

        return np.array(residuals)

    def MSE(self):

        residuals=self.weighted_curves_residuals()

        return (residuals**2).sum()

    def expected_functional_representation(self, S_i):
        [m, A, c, Q, sigma, k, pi] = self.unroll_parameters()
        C=self.parameters["C"]
        clusters = C.shape[1]
        expectation = 0
        for i in range(clusters):
            c = C[:, i]
            expectation = expectation + np.matmul(S_i, m + np.matmul(A, c)) * pi[i]

        return expectation
    def intialize_parameters(self):


        C = np.eye(self.h, self.G)
        pi = np.ones(self.G) / self.G

        k = 0



        nu=[]
        for counter,S_i in enumerate(self.S):
            model=sm.OLS(self.Y[counter],S_i).fit()
            nu.append(model.params)
        nu=np.array(nu)
        kmeans=KMeans(n_clusters=self.G).fit(nu)
        self.centroids=kmeans.cluster_centers_
        m=self.centroids.mean(axis=0)
        centroids_distance=[centroid-m for centroid in self.centroids]

        u, s, A =np.linalg.svd(centroids_distance)

        for counter,distance in enumerate(centroids_distance):

            C[:,counter]=np.matmul(np.linalg.inv(A),distance)


        Q = np.eye(self.q)
        sigma=.01

        self.parameters = {
            "m": m,
            "A": A,
            "C": C,
            "sigma": sigma,
            "Q": Q,
            "k": k,
            "pi": pi
        }



    def unroll_parameters(self):
        m = self.parameters["m"]
        A = self.parameters["A"]
        Q = self.parameters["Q"]
        sigma = self.parameters["sigma"]
        k = self.parameters["k"]
        c = self.parameters["C"][:, k]
        pi = self.parameters["pi"]

        return [m, A, c, Q, sigma, k, pi]


    def do_E_step(self):

        D={}
        covar_d={}


        clusters=self.parameters["C"].shape[1]
        for k in range(clusters):
            d_i = []
            covar_d_i = []
            for counter,S_i in enumerate(self.S):
                self.parameters["k"]=k
                d=self.d_i_E_step(self.S[counter],self.Y[counter])
                d_i.append(d)

                covar=self.d_i_covar_E_step(self.S[counter],self.Y[counter])
                covar_d_i.append(covar)

            D[k]=d_i
            covar_d[k]=covar_d_i

        self.D=D
        self.covar_d=covar_d

    def do_M_step(self,fit_A=False):


        pi_estimates = self.pi_M_step()
        self.parameters["pi"] = pi_estimates

        pi_given_i_M_step=self.pi_given_i_M_step()
        self.pi_given_i=pi_given_i_M_step

        error = self.MSE()

        Q_estimate = self.Q_M_step()
        self.parameters["Q"] = Q_estimate

        error = self.MSE()
        for i in range(10):
            m_estimate = self.m_estimate_M_step()
            self.parameters["m"] = m_estimate.reshape(-1)

            # update c
            error = self.MSE()

            # A=self.A_estimate_M_step_numeric()
            # self.parameters["A"] = A

            XX = self.parameters["C"].copy()


            for k in range(self.G):
                self.parameters["k"]=k
                XX[:, k] = self.c_estimate_M_step().reshape(1, -1)

            self.parameters["C"] = XX.copy()


            error = self.MSE()


            if fit_A:
                XX =  self.parameters["A"].copy()

                for a_col in range( self.parameters["A"].shape[1]):
                    XX[:, a_col] = self.a_m_estimate_M_step(m_column=a_col).reshape(1, -1)
                    self.parameters["A"] = XX


                self.parameters["A"] = XX
                error = self.MSE()







        sigma_estimate=np.sqrt(self.sigma_estimate_M_step())[0][0]
        self.parameters["sigma"] = sigma_estimate
        error = self.MSE()


    def d_i_E_step(self,S_i, Y_i):
        # https://perso.uclouvain.be/michel.verleysen/papers/dsi05sd.pdf
        # (6)

        [m, A, c, Q, sigma, k, pi] = self.unroll_parameters()
        try:
            first = np.matmul(np.transpose(S_i), S_i) + sigma * sigma * np.linalg.inv(Q)
            second = Y_i - np.matmul(S_i, m) - np.matmul(S_i, np.matmul(A, c))
            third = np.linalg.inv(first)
        except:

            raise

        d = np.matmul(np.matmul(third, np.transpose(S_i)), second)

        return d

    def d_i_covar_E_step(self,S_i, Y_i):
        # https://perso.uclouvain.be/michel.verleysen/papers/dsi05sd.pdf
        # (6)

        [m, A, c, Q, sigma, k, pi] = self.unroll_parameters()
        first = np.matmul(np.transpose(S_i), S_i) + sigma * sigma * np.linalg.inv(Q)

        third = np.linalg.inv(first)

        covar = sigma * sigma * np.linalg.inv(first)

        return covar

    def f_y_given_z(self,S_i, Y_i):
        [m, A, c, Q, sigma, k, pi] = self.unroll_parameters()

        N = len(S_i)

        mean = np.matmul(S_i, m) + np.matmul(S_i, np.matmul(A, c))
        covar = (sigma ** 2) * np.eye(N) + np.matmul(S_i, np.matmul(Q, np.transpose(S_i)))
        return multivariate_normal.pdf(Y_i, mean, covar, allow_singular=True)

    def pi_given_i_M_step(self):
        [m, A, c, Q, sigma, k, pi] = self.unroll_parameters()

        S = self.S
        Y = self.Y


        n_k = len(pi)

        pi_estimates = []

        for i in range(self.curves):
            S_i = S[i]
            Y_i = Y[i]
            estimates = []
            for k in range(n_k):
                self.parameters["k"] = k
                estimates.append(self.f_y_given_z(S_i, Y_i) * pi[k])

            if np.array(estimates).sum() == 0:
                estimates = pi
            else:
                estimates = np.array(estimates) / (np.array(estimates).sum())

            pi_estimates.append(estimates)

        return pi_estimates

    def pi_M_step(self):
        # https://perso.uclouvain.be/michel.verleysen/papers/dsi05sd.pdf
        # (7)


        pi_estimates=self.pi_given_i_M_step()
        return np.array(pi_estimates).sum(axis=0) / np.array(pi_estimates).sum()

    def Q_M_step(self):

        # https://perso.uclouvain.be/michel.verleysen/papers/dsi05sd.pdf
        # (8)

        [m, A, c, Q, sigma, k, pi] = self.unroll_parameters()
        n_k = len(pi)

        e_di = 0
        counter = 0
        for i in range(self.curves):

            for k in range(n_k):

                e_di =e_di+ self.covar_d[k][i]
                counter = counter + 1

        Q_estimate = e_di / counter

        return Q_estimate

    def m_estimate_M_step(self):
        # https://perso.uclouvain.be/michel.verleysen/papers/dsi05sd.pdf
        # (9)
        [m, A, c, Q, sigma, k, pi] = self.unroll_parameters()

        S=self.S
        Y=self.Y

        n_k = len(pi)
        product_sum = 0
        product_sum_square = 0

        for i in range(self.curves):

            S_i = S[i]
            Y_i = Y[i]

            weighted_curve = 0
            for k in range(n_k):
                pi_i=self.pi_given_i[i][k]

                d_i = self.D[k][i].reshape(-1, 1)
                weighted_curve = weighted_curve + np.matmul(pi_i * S_i, np.matmul(A, c).reshape(-1, 1) + d_i)

            difference = Y_i.reshape(-1, 1) - weighted_curve

            product_sum = product_sum + np.matmul(np.transpose(S_i), difference)
            product_sum_square = product_sum_square + np.matmul(np.transpose(S_i), S_i)

        return np.matmul(np.linalg.inv(product_sum_square), product_sum)

    def c_estimate_M_step(self):
        # https://perso.uclouvain.be/michel.verleysen/papers/dsi05sd.pdf
        # (10)
        [m, A, c, Q, sigma, k, pi] = self.unroll_parameters()
        S=self.S
        Y=self.Y
        second_term_sum = 0
        inverse_term = 0
        A_tranpose = np.transpose(A)
        for i in range(self.curves):
            pi_i = self.pi_given_i[i][k]
            S_i = S[i]
            S_i_transpose = np.transpose(S_i)
            Y_i = Y[i]
            d_i = self.D[k][i].reshape(-1, 1)
            weighted_curve = Y_i.reshape(-1, 1) - np.matmul(S_i, m).reshape(-1, 1) - np.matmul(S_i, d_i).reshape(-1, 1)
            weigthed_product = pi_i * np.matmul(np.transpose(A), np.matmul(S_i_transpose, weighted_curve))
            second_term_sum = second_term_sum + weigthed_product

            inverse_term = inverse_term + pi_i * np.matmul(A_tranpose, np.matmul(S_i_transpose, np.matmul(S_i, A)))

        return np.matmul(np.linalg.inv(inverse_term), second_term_sum)

    def a_m_estimate_M_step(self, m_column):
        # https://perso.uclouvain.be/michel.verleysen/papers/dsi05sd.pdf
        # (11)
        [m, A, c, Q, sigma, k, pi] = self.unroll_parameters()
        S=self.S
        Y=self.Y
        n_k = len(pi)

        first_term = 0

        second_third_product = 0

        for i in range(self.curves):
            S_i = S[i]
            Y_i = Y[i]
            S_i_transpose = np.transpose(S_i)

            for k in range(n_k):

                pi_i = self.pi_given_i[i][k]
                d_i = self.D[k][i].reshape(-1, 1)
                c = self.parameters["C"][:, k]
                c_km = c[m_column]

                fourth_term = 0

                for l in range(len(c)):
                    if l != m_column:
                        fourth_term = fourth_term + c[l] * np.matmul(S_i, A[:, l]).reshape(-1, 1)

                third_term = Y_i.reshape(-1, 1) - np.matmul(S_i, m).reshape(-1, 1) - fourth_term - np.matmul(S_i, d_i)

                second_third_product = second_third_product + np.matmul(pi_i * c_km * S_i_transpose, third_term)

                first_term = first_term + pi_i * c_km * c_km * np.matmul(S_i_transpose, S_i)

        first_term_inv = np.linalg.inv(first_term)

        return np.matmul(first_term_inv, second_third_product)

    def A_estimate_M_step_numeric(self):

        [m, A, c, Q, sigma, k, pi] = self.unroll_parameters()

        def get_a_numeric(A_flat):

            [m, A, c, Q, sigma, k, pi] = self.unroll_parameters()
            A=A_flat.reshape(int(np.sqrt(len(A_flat))),int(np.sqrt(len(A_flat))))
            self.parameters["A"]=A

            residuals=self.weighted_curves_residuals()

            #get contrain



            return self.A_contrain(A,Q,sigma)+(residuals**2).sum()

        res=minimize(get_a_numeric,A.flatten(),method='BFGS',options={'gtol': 1e-6, 'disp': True})
        A=res.x.reshape(A.shape[0],A.shape[1])

        return A

    def A_contrain(self,A,Q,sigma):

        total_val=0
        for counter, S in enumerate(self.S):
            S = self.S[counter]
            S_t = np.transpose(S)
            A_t = np.transpose(A)

            SIGMA = np.matmul(np.matmul(S, Q), S_t)
            SIGMA = SIGMA + np.eye(SIGMA.shape[0]) * sigma * sigma
            SIGMA_INV = np.linalg.inv(SIGMA)

            constrain = np.matmul(np.matmul(np.matmul(np.matmul(A_t, S_t), SIGMA_INV), S), A)

            total_val=total_val+(constrain.sum()-constrain.shape[0])**2

        return total_val

    def C_estimate_M_step_numeric(self):

        def get_C_numeric(C_flat):
            C=C_flat.reshape(self.parameters["C"].shape[0],self.parameters["C"].shape[1])
            self.parameters["C"] = C

            residuals = self.weighted_curves_residuals()


            contrain=np.array([abs(c) for c in C.sum(axis=1)]).sum()


            print(contrain)
            return (residuals ** 2).sum()+contrain*1e2

        C=self.parameters["C"].flatten()
        res = minimize(get_C_numeric, C.flatten(), method='Nelder-Mead', options={'gtol': 1e-6, 'disp': True})
        C = res.x.reshape(self.parameters["C"].shape[0], self.parameters["C"].shape[1])
        return C

    def sigma_estimate_M_step(self):
        [m, A, c, Q, sigma, k, pi] = self.unroll_parameters()
        S=self.S
        Y=self.Y
        n_k = len(pi)

        counter = 0
        sum_terms = 0
        for i in range(self.curves):
            S_i = S[i]
            Y_i = Y[i]
            S_i_transpose = np.transpose(S_i)

            for k in range(n_k):
                d_i = self.D[k][i].reshape(-1, 1)
                d_i_cov = self.covar_d[k][i]
                c = self.parameters["C"][:, k]

                y_i_bar = Y_i.reshape(-1, 1) - np.matmul(S_i, m).reshape(-1, 1) \
                          - np.matmul(S_i, np.matmul(A, c)).reshape(-1, 1) \
                          - np.matmul(S_i, d_i).reshape(-1, 1)

                y_i_bar_tranpose = np.transpose(y_i_bar)
                first_term = np.matmul(y_i_bar_tranpose, y_i_bar)
                second_term = np.matmul(np.matmul(S_i, d_i_cov), S_i_transpose)

                second_term = 0
                sum_terms = sum_terms + pi[k] * (first_term + second_term)
                counter = counter + 1
        return sum_terms / counter

    def set_parameters(self,parameters):

        self.parameters=parameters

    def functional_clasification(self,curve_i):
        """
        classifies a curve according to https://dial.uclouvain.be/pr/boreal/object/boreal%3A20030/datastream/PDF_01/view
        H.47
        :param curve_i:
        :return:
        """
        [m, A, c, Q, sigma, k, pi] = self.unroll_parameters()

        posterior=pd.DataFrame(index=range(self.G),columns=["error"])

        S_i=self.S[curve_i]
        S_i_t=np.transpose(S_i)
        Y_i=self.Y[curve_i]
        n=len(S_i)


        for k in posterior.index:

            x=Y_i-self.functional_representation(k,S_i)
            SIGMA=np.matmul(np.matmul(S_i, Q), S_i_t)+np.eye(n) * sigma * sigma
            SIGMA_inv=np.linalg.inv(SIGMA)
            posterior.loc[k]=np.matmul(np.matmul(np.transpose(x),SIGMA_inv),x)


        classification=posterior[posterior.error==posterior.error.min()].index.values[0]

        self.curves_classified[curve_i]=self.functional_representation(classification,S_i)
        self.coefficients_representation[curve_i]=self.get_coefficients_representaiton(classification,curve_i)

        if self.t_in.seconds>0:
            S_i_in=self.S_in[curve_i]
            self.curves_classified_in[curve_i] = self.functional_representation(classification, S_i_in)



        return classification
    def classify_curves(self):
        """
        assignes curves to clusters
        :return:
        """
        classification=[]
        for curve_i,S_i in enumerate(self.S):

            classification.append(self.functional_clasification(curve_i))

        self.curves_class=classification
        self.are_curves_classified=True

    def plot_initial_curves(self):
        S=self.S
        for i in range(self.curves):
            x_range = self.X[i]
            plt.plot(x_range, self.expected_functional_representation(S_i=S[i]),
                     label="expected functional form")
            plt.plot(x_range, self.Y[i], label="Observed", color="grey", alpha=.5)

        plt.show()

    def plot_curve_train(self,curve_i,k):
        plt.plot(self.Y[curve_i])
        plt.plot(self.functional_representation(k,self.S[curve_i]))
        plt.show()

    def plot_curves_by_cluster(self):
        if self.are_curves_classified==False:
            self.classify_curves()

        columns=2
        fig, ax = plt.subplots(int(self.G/columns)+1, columns)

        fig.suptitle("Clustered curves : "+ self.name, fontsize=14)
        for curve_i,y_i in enumerate(self.Y):

            row=int(self.curves_class[curve_i]/(columns))
            col=self.curves_class[curve_i]%(columns)
            sns.lineplot(self.spline_time[curve_i],self.curves_classified[curve_i],
                                     color="blue",alpha=.5,ax=ax[row,col])



            sns.lineplot(self.spline_time[curve_i], self.Y[curve_i],
                         color="grey", alpha=.5, ax=ax[row, col],dashes=True)

            if self.t_in.seconds>0:
                sns.lineplot(self.spline_time_in[curve_i], self.  curves_classified_in[curve_i],
                             color="yellow", alpha=.5, ax=ax[row, col])





            ax[row,col].set_title("Cluster: "+ str(self.curves_class[curve_i]))

        plt.show()

    def fit(self,steps=20,fit_A=False):
        self.error=[]

        self.plot_initial_curves()
        for i in range(steps):

            self.do_E_step()
            self.do_M_step(fit_A)
            error = self.MSE()
            print("Error after iteration "+ str(i),error)
            self.error.append(error)

            if i%10==0:
                self.plot_initial_curves()





    def plot_error(self):
        plt.plot(self.error)
        plt.show()

class Trainer():
    """
    implements:
    https://perso.uclouvain.be/michel.verleysen/papers/ffm07sd.pdf
    particular case of
    http://faculty.marshall.usc.edu/gareth-james/Research/fpc.pdf

    """


    def __init__(self,Y,X,t_max,q,G,name="",t_in=timedelta(0),initialize_parameters=True):
        """

        :param Y:
        :param X:
        :param t_max:
        :param q:
        :param G:
        """


        self.X=X
        self.Y=Y
        self.curves=len(X)
        self.t_max=t_max

        self.map = interp1d([0, t_max.seconds/60], [0, 1])

        #defaul parameters

        self.q=q
        self.G=G

        self.name=name

        self.model=ReducedRank(Y,X,t_max,q,G,name,t_in,initialize_parameters)


    def fit(self,steps=20,fit_A=False):
        self.model.fit(steps,fit_A)

    def get_basis_matrix(self):

        return self.model.spline_time,self.model.S



class RBF():

    def __init__(self,x,y):
        """

        :param x: list of input vectors
        :param y: list of output vectors
        """

        self.x=np.array(x)
        self.y=np.array(y)

    def fit(self,hidden_units=3,load_model=False,model_name="no_name"):


        if len(self.x)>0:
            input_shape=self.x[0].shape[0]
            output_shape=self.y[0].shape[0]



            if load_model==True:
                print("loading model "+model_name)
                model = keras.models.load_model(PATH_RBF_PFM+model_name+".h5", custom_objects={'RBFLayer': RBFLayer})
            else:
                print("Training model " + model_name)
                model=Sequential()

                if len(self.x)>hidden_units:

                    rbflayer=RBFLayer(hidden_units,
                                      initializer=InitCentersKMeans(self.x),
                                      input_shape=(input_shape,))
                else:
                    rbflayer = RBFLayer(hidden_units,
                                        initializer=InitCentersRandom(self.x),
                                        input_shape=(input_shape,))

                model.add(rbflayer)
                model.add(Dense(output_shape))
                model.compile(loss='mean_squared_error',
                              optimizer=RMSprop())


                if self.x.shape[0]*.1<1:
                    #sample is very small to train

                    model=[]

                else:

                    history=model.fit(self.x, self.y,
                              batch_size=20,
                              epochs=5000,
                              verbose=False,
                              validation_split=0.1)




                    plt.plot(history.history['loss'])
                    plt.plot(history.history['val_loss'])
                    plt.title('model loss'+model_name)
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'val'], loc='upper left')
                    plt.show()

                    if model_name!="no_name":
                        pass
                        # model.save(PATH_RBF_PFM+model_name+".h5")



            self.model = model
        else:
            self.model=[]




class PriceForecastingModel():

    def __init__(self,TrainerSpaceIn,TrainerSpaceOut):
        """

        :param TrainerSpaceIn:
        :param TrainerSpaceOut:
        """

        self.trainer_space_in=TrainerSpaceIn
        self.trainer_space_out=TrainerSpaceOut

        self.frequency_table=self.get_frequency_table()




        #Initialize RBFN
        self.out_coef_dim=self.trainer_space_out.q
        self.in_coef_dim = self.trainer_space_in.q
        self.n_clusters_out=self.trainer_space_out.G

        self.initialize_rbf()

    def get_frequency_table(self):

        curves_classification_in = self.trainer_space_in.model.curves_class
        curves_classification_out = self.trainer_space_out.model.curves_class

        frequency_table = pd.DataFrame(index=range(self.trainer_space_in.G), columns=range(self.trainer_space_out.G))
        frequency_table = frequency_table.fillna(0)
        for curve_i in range(len(curves_classification_in)):
            curve_in = curves_classification_in[curve_i]
            curve_out = curves_classification_out[curve_i]
            frequency_table.loc[curve_in][curve_out] = 1 + frequency_table.loc[curve_in][curve_out]

        frequency_table = frequency_table.div(frequency_table.sum(axis=1), axis=0)
        return frequency_table

    def fit(self,hidden_nodes=3,force_train=False):

        for cluster in self.local_rbf.keys():

            model_name = "cluster_rbf_" + str(cluster)
            if os.path.exists(BASE_DIR+"/"+PATH_RBF_PFM+model_name+".h5"):
                if force_train == True:
                    load_model = False
                else:
                     load_model=True
            else:
                load_model=False




            self.local_rbf[cluster].fit(hidden_units=hidden_nodes,
                                        load_model=load_model,
                                        model_name=model_name,
                                        )



    def initialize_rbf(self):

        local_rbf={}
        x={i:[] for i in range(self.n_clusters_out)}
        y={i:[] for i in range(self.n_clusters_out)}

        for curve_i in range(self.trainer_space_out.curves):

            curve_out_cluster=self.trainer_space_out.model.curves_class[curve_i]

            coefficients_in=self.trainer_space_in.model.coefficients_representation[curve_i]
            coefficients_out=self.trainer_space_out.model.coefficients_representation[curve_i]

            x[curve_out_cluster].append(coefficients_in)
            y[curve_out_cluster].append(coefficients_out)

        for i in range(self.n_clusters_out):
            local_rbf[i]=RBF( x[i],y[i])

        self.local_rbf=local_rbf
        self.rbf_x=x
        self.rbf_y=y





    def predict_curve(self,S_i_out,coefficients_in,cluster_in):
        """
        Predicts curve by predicting coefficients
        :param S_i_out: spline basis for the full out space
        :param coefficients_in: last observed coefficients for the in space
        :param cluster_in:  cluster that the in-space was assigned by method. ReducedRank.functional_clasification(self,curve_i):
        :return: curve in price
        """

        prediction = np.array([0 for i in range(S_i_out.shape[0])])
        fre_table = self.frequency_table

        try:
            for column in fre_table.columns:

                if fre_table.loc[cluster_in][column]==0 or isinstance(self.local_rbf[column].model,list):
                    prediction=prediction
                else:
                    predicted_coefficients = self.predict_coefficients(coefficients_in, column)
                    predicted_curve = np.matmul(S_i_out, predicted_coefficients)
                    prediction = prediction + fre_table.loc[cluster_in][column] * predicted_curve
        except Exception as e:
            raise

        return prediction

    def predict_coefficients(self,coefficients_in,cluster_out):
        """
        Predicts coefficients using RBF trained model
        :param coefficients_in:
        :param cluster_out:
        :return:
        """


        local_rbfs=self.local_rbf
        coefficients_prediction=local_rbfs[cluster_out].model.predict(coefficients_in.reshape(1,-1))

        if np.isnan(coefficients_prediction).any():
            error="SDF"

        return coefficients_prediction.reshape(-1)


    def test_prediction(self,trainer_in_online,trainer_out_online,online=False,plot=True):


        spline_basis_out = trainer_out_online.model.S
        spline_time_out = trainer_out_online.model.spline_time
        y_out =trainer_out_online.model.Y

        curves_class_in = trainer_in_online.model.curves_class
        coefficients_in = trainer_in_online.model.coefficients_representation
        y_in=trainer_in_online.Y
        columns = 2

        all_predictions=self.curves_prediction(spline_basis_out,coefficients_in,
                          curves_class_in
                         )

        strategy_df=pd.DataFrame(index=range(len(all_predictions)+1),columns=["strategy_value","asset_return",
                                                                              "strategy_return","strategy_position",
                                                                              "asset_side"])
        for curve_i,curve_predicted in enumerate(all_predictions):



            asset_return=y_out[curve_i][-1]/y_in[curve_i][-1]-1
            strategy_df.loc[curve_i ]["asset_return"]=asset_return

            strategy_df.loc[curve_i]["asset_side"] = np.sign(asset_return)

            try:


                if np.isnan(curve_predicted).any() or curve_predicted.sum()==0:
                    #error in prediction does not predict
                    strategy_df.loc[curve_i]["strategy_position"] = 0
                else:
                    if curve_predicted[-1] > curve_predicted[len(y_in[curve_i])]:
                        # open long position
                        strategy_df.loc[curve_i]["strategy_position"] = 1

                    else:
                        strategy_df.loc[curve_i]["strategy_position"] = -1
                        # open short position

            except:
                raise




            strategy_df.strategy_return=strategy_df.asset_return*strategy_df.strategy_position

            bets=strategy_df[strategy_df.strategy_position!=0]
            bets=bets.dropna(how="all")
            correct_forecast=abs(bets.asset_side+bets.strategy_position).sum()/2
            correct_forecast=correct_forecast/len(strategy_df.index)


        self.correct_forecast_test=correct_forecast
        self.all_predictions_test=all_predictions
        self.strategy_df=strategy_df

        return strategy_df,all_predictions,correct_forecast


    def curves_prediction(self,spline_basis_out,coefficients_in,
                          curves_class_in
                          ):


        all_predictions=[]


        for curve_i, curve_class in enumerate(spline_basis_out):


            predicted_curve = self.predict_curve(spline_basis_out[curve_i], coefficients_in[curve_i],
                                                 curves_class_in[curve_i])




            all_predictions.append(predicted_curve)

        return all_predictions

    def plot_predictions_training(self,curves_index=[1,2],trainer_space_out="",trainer_space_in=""):
        """
        Plots prediction curves from the training sample
        :param curves_index: list with indices of the curves that wants to be ploted.
        :return: figure with actual curve as projections
        """

        if trainer_space_out=="":
            trainer_space_out=self.trainer_space_out
        if trainer_space_in=="":
            trainer_space_in=self.trainer_space_in

        curves_class_out=[trainer_space_out.model.curves_class[i] for i in curves_index]
        spline_basis_out = [trainer_space_out.model.S[i] for i in curves_index]
        spline_time_out=[trainer_space_out.model.spline_time[i] for i in curves_index]
        curve_classified_out=[trainer_space_out.model.curves_classified[i] for i in curves_index]
        y_out=[trainer_space_out.model.Y[i] for i in curves_index]

        curves_class_in =[ trainer_space_in.model.curves_class[i] for i in curves_index]
        coefficients_in=[trainer_space_in.model.coefficients_representation[i] for i in curves_index]


        columns=2
        fig, ax = plt.subplots(int((len(curves_class_out) + 1)/columns), columns)
        fig.suptitle("Training Curves Prediction", fontsize=14)

        for curve_i,curve_class in enumerate(coefficients_in):
            row = int(curve_i / (columns))
            col = curve_i % (columns)


            predicted_curve=self.predict_curve(spline_basis_out[curve_i],coefficients_in[curve_i],
                                               curves_class_in[curve_i])


            sns.lineplot(spline_time_out[curve_i], curve_classified_out[curve_i],
                         color="blue", alpha=.5, ax=ax[row, col], label="Observed smoothed curve")

            sns.lineplot(spline_time_out[curve_i], predicted_curve,
                         color="red", alpha=.5, ax=ax[row, col],label="Predicted Curve")

            sns.scatterplot(spline_time_out[curve_i], y_out[curve_i],
                         color="yellow", alpha=.5, ax=ax[row, col], label="Observed Prices")

        plt.legend(loc="upper left")
        plt.show()




class ModelBackTest():


    def __init__(self,t_in_list,t_out_list,q_spline_knots,G_clusters,rbf_hidden):
        self.t_in_list=t_in_list
        self.t_out_list=t_out_list
        self.q_spline_knots=q_spline_knots
        self.G_clusters=G_clusters
        self.rbf_hidden=rbf_hidden

        self.source_csv_file="SPY_1M.csv"
        self.training_percent = .6

    def validate(self):

        models={}
        model_count=0
        best_model=0
        best_forecast=0
        total_models_to_fit=len(self.t_in_list)*len(self.t_out_list)*len(self.q_spline_knots)*len(self.G_clusters)*len(self.rbf_hidden)
        for t_in in self.t_in_list:
            for t_out in self.t_out_list:
                for q in self.q_spline_knots:
                    for G in self.G_clusters:
                        for hidden_nodes in self.rbf_hidden:
                            print("Fitting model: "+ str(model_count) +" out of "+ str(total_models_to_fit))

                            try:
                                [strategy_df, all_predictions, forecast,model_objects]=self.train_test_model(t_in, t_out, q, G,hidden_nodes, str(model_count))

                                if forecast>best_forecast:
                                    best_forecast=forecast
                                    best_model=model_count
                                error_train = ""
                            except Exception as e:
                                error_train=e
                                forecast="none"
                                model_objects={}

                            models[model_count]={"t_in":t_in,"t_out":t_out,"q":q,"G":G,"error":error_train,
                                                 "forecast":forecast,"model_objects":model_objects}
                            model_count=model_count+1

        self.best_model=best_model
        self.models=models



    def train_test_model(self,t_in,t_out,q,G,hidden_nodes,model_name):

        data_handler_pfm = DataHandlerPriceForcastingModel(source_csv_file=self.source_csv_file,
                                                           t_in=t_in, t_out=t_out,
                                                           training_percent=self.training_percent)

        [X_in, Y_in, X_out, Y_out] = data_handler_pfm.get_sample_split_by_day_train()
        curves_index = [i for i in range(4)]

        # fit in models
        trainer_in = Trainer(Y=Y_in, X=X_in, t_max=t_in, q=q, G=G, name="In Space " +model_name)
        trainer_in.fit(5, fit_A=True)
        trainer_in.model.plot_error()
        trainer_in.model.plot_curves_by_cluster()

        # fit out coefficients
        trainer_out = Trainer(Y=Y_out, X=X_out, t_max=t_in + t_out, q=q, G=G, name="Out Space" +model_name, t_in=t_in)
        trainer_out.model.parameters = trainer_in.model.parameters
        trainer_out.fit(10, fit_A=True)
        trainer_out.model.plot_error()
        trainer_out.model.plot_curves_by_cluster()

        forcasting_model = PriceForecastingModel(trainer_in, trainer_out)
        forcasting_model.fit(hidden_nodes=hidden_nodes)
        forcasting_model.plot_predictions_training(curves_index=curves_index)

        [X_in_v, Y_in_v, X_out_v, Y_out_v] = data_handler_pfm.get_sample_split_by_day_validate()

        trainer_in_online = Trainer(Y=Y_in_v, X=X_in_v, t_max=t_in, q=q, G=G, name="In Space Online" +model_name,
                                    initialize_parameters=False)
        trainer_in_online.model.parameters = trainer_in.model.parameters
        # fit to obtaine coefficient representaiton clustering algorithm is needed
        trainer_in_online.fit(1, fit_A=True)
        trainer_in_online.model.classify_curves()
        trainer_in_online.model.plot_curves_by_cluster()

        trainer_out_online = Trainer(Y=Y_out_v, X=X_out_v, t_max=t_in + t_out, q=q, G=G, name="Out Space Online"+model_name,
                                     t_in=t_in, initialize_parameters=False)
        trainer_out_online.model.parameters = trainer_out.model.parameters
        trainer_out_online.model.classify_curves()

        [strategy_df, all_predictions, correct_forecast] = forcasting_model.test_prediction(trainer_in_online,
                                                                                            trainer_out_online)

        model_objects={"trainer_in":trainer_in,"trainer_out":trainer_out,"forecasting_model":forcasting_model,
                       "trainer_in_online":trainer_in_online,"trainer_out_online":trainer_out_online}

        return [strategy_df, all_predictions, correct_forecast,model_objects]

# plt.close('all')
# t_in_list=[timedelta(hours = 3)]
# t_out_list=[timedelta(minutes = 30),timedelta(minutes = 60)]
# q_spline_knots=[5,7]
# G_clusters=[2,3,4,5]
# rbf_hidden=[2,3,4]
# back_test=ModelBackTest(t_in_list=t_in_list,t_out_list=t_out_list,
#                         q_spline_knots=q_spline_knots,G_clusters=G_clusters,
#                         rbf_hidden=rbf_hidden)
# sns.set()
# back_test.validate()
# a=5


# t_in_list=[timedelta(hours = 3)]
# t_out_list=[timedelta(minutes = 30)]
# q_spline_knots=[7]
# G_clusters=[5]
# rbf_hidden=[3]
# back_test=ModelBackTest(t_in_list=t_in_list,t_out_list=t_out_list,
#                         q_spline_knots=q_spline_knots,G_clusters=G_clusters,
#                         rbf_hidden=rbf_hidden)
# sns.set()
# back_test.validate()



















