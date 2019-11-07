"""    This File performs the test to check wheter the empirical data follows a specific copula model or not
Input: Empirical reterun vectors
Output: csv-files with rejection frequencies of the test, comparing data to specified copula model

   """

# Fabian Kaechele 06.11.2019

import numpy as np
import scipy
from statsmodels.distributions.empirical_distribution import ECDF
import sys
import pandas as pd
import functionstog1

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

# Window size
size=500

# Anzahl Fenster
Fenster=8

# rolling parameter
rolling=20


# dates
start1=20070101
family2='t'

# Alpha-Value
alphatotal=0.05

# how often should the window be rolled
runs=130
# how often should the test be repeated for each setting
wdh=100


# necessary values, but not important
Fenster=Fenster+1
family1='normal'
dof=3
mix=0
dof1=3
dof2=dof
pr1=0
pr2=0
tau1=pr1
tau2=pr2
n1=20
n2=size



# read in data - select which dataset should be read
#df = pd.read_csv("C:/Users/fkaec/Desktop/Master/Masterarbeit/04_Sonstiges/5_Industry_Portfolios_Daily1.csv",dtype=str)
df = pd.read_csv("C:/Users/fkaec/Desktop/Master/Masterarbeit/04_Sonstiges/12_Industry_Portfolios_Daily1.csv",dtype=str)
#df = pd.read_csv("C:/Users/fkaec/Desktop/Master/Masterarbeit/04_Sonstiges/49_Industry_Portfolios_Daily1.csv",dtype=str)
df.rename(columns={ df.columns[0]: "Date" }, inplace = True)
#types_dict = {'Date': float, 'Cnsmr': float, 'Manuf': float, 'HiTec': float, 'Hlth': float, 'Other': float}
types_dict = {'Date': float, 'NoDur': float, 'Durbl': float, 'Manuf': float, 'Enrgy': float, 'Chems': float,'BusEq': float,'Telcm': float,'Utils': float,'Shops': float,'Hlth': float,'Money': float,'Other': float}
#types_dict = {'Date': float, 'Agric': float, 'Food ': float, 'Soda ': float, 'Beer ': float, 'Smoke': float,'Toys ': float,'Fun  ': float,'Books': float,'Hshld': float,'Clths': float,'Hlth ': float,'MedEq': float,'Drugs': float,	'Chems': float,	'Rubbr': float,	'Txtls': float,	'BldMt': float,	'Cnstr': float,	'Steel': float,	'FabPr': float,	'Mach ': float, 	'ElcEq': float,	'Autos': float,	'Aero ': float, 	'Ships': float,	'Guns ': float, 	'Gold ': float, 	'Mines': float,	'Coal ': float, 	'Oil  ': float,  	'Util ': float, 	'Telcm': float,	'PerSv': float,	'BusSv': float,	'Hardw': float,	'Softw': float,	'Chips': float,	'LabEq': float,	'Paper': float,	'Boxes': float,	'Trans': float,	'Whlsl': float,	'Rtail': float,	'Meals': float,	'Banks': float,	'Insur': float,	'RlEst': float,	'Fin  ': float,  	'Other': float}


for col, col_type in types_dict.items():
    df[col] = df[col].astype(col_type)

# Select columns
#Industry=('Cnsmr', 'Manuf', 'HiTec', 'Hlth', 'Other')
Industry=('Manuf', 'Hlth')
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
                t1 = np.zeros(shape=(Fenster, Fenster, wdh))
                t2 = np.zeros(shape=(Fenster, Fenster, wdh))
                i=0
                j=i+size
                h=0

                while i<(runs*rolling):

                    df3 = df2[i:j]

                    # Create Copula and calculate pr
                    first = makecopula(df3)
                    pr2,rest=scipy.stats.pearsonr(first[:,0],first[:,1])

                    # Clear old data
                    unten = 0
                    oben = 0
                    mitte = 0
                    abgelehnt = np.zeros(shape=(wdh))

                    # how often
                    for n in range(0,wdh):

                        # Create copula and test
                        firstxx,second = functionstog1.generateCopulas(family1, family2, n1, n2,  tau1, tau2, pr1, pr2, dof1, dof2,  mix)
                        t1, t2, abgelehnt,DD,G = functionstog1.test(alphatotal,first, second, size, size, t1, t2, Fenster, n, abgelehnt)


                        # symmetrical rejection or not
                        if DD!=None:

                            if G[DD] >((DD + 2) ** 2 - 1):
                                #Gebe Position wie gewohnt an
                                H=G[DD]-(DD+2)**2
                                # Zeilennummer G[DD] ist index Fensterzahl bei ablehnung
                                helper3 = np.floor(H / (DD + 2))
                                # Spaltennummer
                                helper4 = (H/ (DD + 2) - helper3) * (DD + 2)
                                # Einsortieren
                                if (helper3/(DD+2))<= (1/3) and (helper4/(DD+2))<=(1/3):
                                    unten=unten+1
                                elif (helper3/(DD+2))>= (2/3) and (helper4/(DD+2))>=(2/3):
                                    oben=oben+1
                                else:
                                    mitte=mitte+1


                    # Save values in array
                    ablehnungtotal[h, 0] = sum(abgelehnt)/wdh
                    ablehnungtotal[h, 1] = (oben + mitte + unten) / wdh

                    # Nextwindow
                    i=i+rolling
                    j=j+rolling
                    h=h+1

                # Save csv
                name = str(Industry1) + '_'+str(Industry2)+'.csv'
                np.savetxt(name, ablehnungtotal, delimiter=",")
