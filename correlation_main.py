"""   Performs a calculation of the rank correlation of two industries

Input: empirical return vectors
Output: csv with rank correlations.

   """

# Fabian Kaechele 06.11.2019

import numpy as np
import scipy
from statsmodels.distributions.empirical_distribution import ECDF
import sys
import pandas as pd


def makecopula(z):
    """    This function calculates the copula of given data
    Input:  df4: Marginals
    Output: first: [0,1]-scaled marginals

       """



    a = z[Industry1]
    b = z[Industry2]

    e = ECDF(a)
    e2 = ECDF(b)
    u = e(a)
    v = e2(b)
    for i in range(0, len(u)):
        if u[i] >= 1:
            u[i] = 1 - sys.float_info.epsilon
        if u[i] <= 0:
            u[i] = 0 + sys.float_info.epsilon
        if v[i] >= 1:
            v[i] = 1 - sys.float_info.epsilon
        if v[i] <= 0:
            v[i] = 0 + sys.float_info.epsilon
    first = np.transpose(np.vstack([u, v]))
    return first

# window size
size=500


# rolling parameter
rolling=20


# dates
start1=19260701

# Alpha-Value
alphatotal=0.05
# how often should the window be rolled
runs=1200





# read in data - select which dataset should be read
df = pd.read_csv("C:/Users/fkaec/Desktop/Master/Masterarbeit/04_Sonstiges/5_Industry_Portfolios_Daily1.csv",dtype=str)
#df = pd.read_csv("C:/Users/fkaec/Desktop/Master/Masterarbeit/04_Sonstiges/12_Industry_Portfolios_Daily1.csv",dtype=str)
#df = pd.read_csv("C:/Users/fkaec/Desktop/Master/Masterarbeit/04_Sonstiges/49_Industry_Portfolios_Daily1.csv",dtype=str)
df.rename(columns={ df.columns[0]: "Date" }, inplace = True)
types_dict = {'Date': float, 'Cnsmr': float, 'Manuf': float, 'HiTec': float, 'Hlth': float, 'Other': float}
#types_dict = {'Date': float, 'NoDur': float, 'Durbl': float, 'Manuf': float, 'Enrgy': float, 'Chems': float,'BusEq': float,'Telcm': float,'Utils': float,'Shops': float,'Hlth': float,'Money': float,'Other': float}
#types_dict = {'Date': float, 'Agric': float, 'Food ': float, 'Soda ': float, 'Beer ': float, 'Smoke': float,'Toys ': float,'Fun  ': float,'Books': float,'Hshld': float,'Clths': float,'Hlth ': float,'MedEq': float,'Drugs': float,	'Chems': float,	'Rubbr': float,	'Txtls': float,	'BldMt': float,	'Cnstr': float,	'Steel': float,	'FabPr': float,	'Mach ': float, 	'ElcEq': float,	'Autos': float,	'Aero ': float, 	'Ships': float,	'Guns ': float, 	'Gold ': float, 	'Mines': float,	'Coal ': float, 	'Oil  ': float,  	'Util ': float, 	'Telcm': float,	'PerSv': float,	'BusSv': float,	'Hardw': float,	'Softw': float,	'Chips': float,	'LabEq': float,	'Paper': float,	'Boxes': float,	'Trans': float,	'Whlsl': float,	'Rtail': float,	'Meals': float,	'Banks': float,	'Insur': float,	'RlEst': float,	'Fin  ': float,  	'Other': float}



for col, col_type in types_dict.items():
    df[col] = df[col].astype(col_type)

# Select columns



#Industry=('Cnsmr', 'Manuf', 'HiTec', 'Hlth', 'Other')
Industry=('Manuf', 'Cnsmr')
#Industry=('NoDur', 'Durbl', 'Manuf', 'Enrgy', 'Chems','BusEq','Telcm','Utils','Shops','Hlth','Money','Other')
#Industry=('Oil  ','Txtls','Clths','Toys ','Smoke','Food ','Beer ','Soda ','Coal ','Banks', 'Insur', 'Fin  ')

ablehnungtotal=np.zeros(shape=(runs,2))

for Industry1 in Industry:
    for Industry2 in Industry:
        if Industry1!=Industry2:
            if Industry.index(Industry1) < Industry.index(Industry2) :
                df1=df[['Date',Industry1,Industry2]]

                # Select correct time frame

                df2=df1.loc[(df1.Date>=start1)]


                cors=np.zeros(shape=runs)

                i=0
                j=i+size


                h=0
                while i<(runs*rolling):
                    # Select time frame
                    df3 = df2[i:j]

                    # Create Copula
                    first = makecopula(df3)
                    # Calculate correlation
                    pr2, rest = scipy.stats.pearsonr(first[:, 0], first[:, 1])

                    cors[h]=pr2


                    # next window
                    i=i+rolling
                    j=j+rolling
                    h=h+1

                # Save csv
                name = str(Industry1) + '_'+str(Industry2)+'_corr.csv'
                np.savetxt(name, cors, delimiter=",")
