"""    This File performs the test based on data read in.
Input: Data of returns
Output: Grafics displaying where copulas are time-varying

   """

# Fabian Kaechele

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import sys
import pandas as pd
import functionstog1
from arch import arch_model
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def makecopula(z):
    """    This function calculates the copula of given data
    Input:  df4: Marginals
    Output: first: [0,1]-scaled marginals

       """
    if Garchfitting == 1:
        model = arch_model(z[[Industry1]], vol='GARCH', power=2.0, p=1, q=1)
        fixed_res = model.fit(disp='off')
        a = fixed_res.resid
        model1 = arch_model(z[Industry2], vol='GARCH', power=2.0, p=1, q=1)
        fixed_res1 = model1.fit(disp='off')
        b = fixed_res1.resid

    else:
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

# max number of windows for test
Fenster=8

# rolling parameter
rolling=2


# dates
start1=20031112
start2=20090701

# Alpha-Value
alphatotal=0.05

# how often should we roll
runs=500
Fenster=Fenster+1
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



Industry=('Cnsmr', 'Manuf', 'HiTec', 'Hlth', 'Other')
#Industry=('Hlth', 'Other')
#Industry=('NoDur', 'Durbl', 'Manuf', 'Enrgy', 'Chems','BusEq','Telcm','Utils','Shops','Hlth','Money','Other')
+Industry=('Oil  ','Txtls','Clths','Toys ','Smoke','Food ','Beer ','Soda ','Coal ','Banks', 'Insur', 'Fin  ')
k=0
for Industry1 in Industry:
    for Industry2 in Industry:
        if Industry1!=Industry2:
            if Industry.index(Industry1) < Industry.index(Industry2) :
                df1=df[['Date',Industry1,Industry2]]

                # Select correct time frame
                df2=df1.loc[(df1.Date>=start1)]
                df3=df1.loc[(df1.Date>=start2)]
                df4 = df2[0:size]

                # Create first copula
                first = makecopula(df4)

                # create arrays to save data in later
                abgelehnt = np.zeros(shape=(runs))
                t1 = np.zeros(shape=(Fenster, Fenster, runs))
                t2 = np.zeros(shape=(Fenster, Fenster, runs))
                unten=0
                oben=0
                mitte=0
                i=0
                j=i+size
                h=0
                fig, ax = plt.subplots(1)
                errorboxes = []


                while i<(runs*rolling):

                    df5 = df3[i:j]

                    # Create second copula
                    second = makecopula(df5)
                    # test
                    t1, t2, abgelehnt,DD,G = functionstog1.test(alphatotal,first, second, size, size, t1, t2, Fenster, h, abgelehnt)

                    # create plot
                    # DD=Fensterzahl, G[DD]=wie vieltes
                    if DD!=None:
                        # only symetrical
                        if G[DD]<=((DD+2)**2-1):
                            # wide of box
                            helper=1/(DD+2)
                            # number of row
                            helper1=np.floor(G[DD]/(DD+2))
                            # number of column
                            helper2=(G[DD]/(DD+2)-helper1)*(DD+2)
                            x=helper*helper1
                            y=helper*helper2


                            rect = Rectangle((x, y), helper, helper)
                            errorboxes.append(rect)
                            # asymmetrical
                        if G[DD] >((DD + 2) ** 2 - 1):

                            H=G[DD]-(DD+2)**2
                            # number of row
                            helper3 = np.floor(H / (DD + 2))
                            # number of column
                            helper4 = (H/ (DD + 2) - helper3) * (DD + 2)
                            # sort
                            if (helper3/(DD+2))<= (1/3) and (helper4/(DD+2))<=(1/3):
                                unten=unten+1
                            elif (helper3/(DD+2))>= (2/3) and (helper4/(DD+2))>=(2/3):
                                oben=oben+1
                            else:
                                mitte=mitte+1


                    # next window
                    i=i+rolling
                    j=j+rolling
                    h=h+1



                # Add collection to axes
                pc = PatchCollection(errorboxes, facecolor='k', alpha=0.005, edgecolor='k')
                ax.add_collection(pc)
                ax.set_xlabel(Industry1)
                ax.set_ylabel(Industry2)
                plt.title (str(start1)+ ' vs. '+str(start2)+ ' with window size='+str(size)+', runs='+str(runs)+', rolling='+str(rolling))
                gesamtablehnung=sum(abgelehnt)/runs
                # Plot
                plt.show()
                name = str(start1) +'_'+str(Industry1)+str(Industry2)+ '.png'
                fig.savefig(name)
                bild=(unten/runs*100,mitte/runs*100, oben/runs*100)
                # this is for plotting purpose
                index = (1,3,5)
                fig2=plt.bar(index, bild)
                plt.xlabel('Rejection area', fontsize=8)
                plt.ylabel('No of rejections', fontsize=8)
                plt.ylim(0,100)
                plt.xticks(index, ('lower tail','middle','upper tail'), fontsize=8, rotation=11)
                plt.title('Unsymetric rejections in percentage '+str(Industry1)+str(Industry2))#
                name1 = str(start1) + '_' + str(Industry1) + str(Industry2) + '_unsym.png'
                plt.savefig(name1)
                plt.show()
