# Fabian Kaechele

""" use the functions out of this file to perform overall test with overlappng cells
                        """
import numpy as np
import scipy

import pandas as pd

import statsmodels
from scipy import stats

def sort(a,n,Fenster):
    """ Sortiert Datenpunkte in Fenster ein und berechnet Haeufigkeiten. Wird von generateCopula() aufgerufen
            Input:  a: Matrix aus u,v
                    n: Stichprobengroesse
                    Fenster: Anzahl Fenster in x,y-Richtung
            Output: p: Matrix mit Haeufigkeiten
                    """

# Leere Matrix mit Whk. erzeugen
    p = np.zeros(shape = (Fenster, Fenster))
    overlap=(0.1)
    df=pd.DataFrame(a)#{'u':[a[:,0]],'v':[a[:,1]]})
    df.columns = ['u', 'v']
    dfhelp=pd.DataFrame(np.zeros(shape=(n,Fenster*Fenster)))
    df=pd.concat([df,dfhelp], axis=1)

    k=0
    for i in range(0, Fenster):
        lowu = i * (1 / Fenster * (1 - 0.5 * overlap))
        highu = (i + 1) * 1 / Fenster * (1 + 0.5 * overlap)
        for j in range(0, Fenster):
            lowv = j * (1 / Fenster * (1 - 0.5 * overlap))
            highv = (j + 1) * 1 / Fenster * (1 + 0.5 * overlap)
            df[k]=np.where((df['u']>=lowu)&(df['u']<=highu)&(df['v']>=lowv )&(df['v']<=highv),1,0)
            k=k+1

    k=0
    for i in range(0,Fenster):
        for j in range(0, Fenster):
            p[i,j]=df[k].sum()
            k=k+1

    p = p / n
    return p


def test(alphatotal,first,second,n1,n2,t1,t2,Fenster,h,abgelehnt):
    """ Berechnet Teststatistik fuer jedes Fenster und vergleicht mit kritischem Wert
        Erzeugt Detailansicht t1/t2 je Testdurchlauf mit Ablehnungswahrscheinlichkeit je Fenster - nach fdr correction
        Erzeugt Vektor h mit ablehnung(1) oder annahme von H0 fuer jeden run

                Input:  alphatotal:
                        first: Copula1
                        second: Copula2
                        n1: Sample Size Copula1
                        n2: Sample Size Copula2
                        t1: 3-dim. Array aus 0/1 der fuer jeden Testdurchlauf festhaelt wo Test angenommen/abgelehnt
                        t2: 3-dim. Array aus 0/1 der fuer jeden Testdurchlauf festhaelt wo Test angenommen/abgelehnt
                        Fenster: Anzahl Fenster in x,y-Richtung
                        h: Laufvariable fuer Anzahl runs
                        abgelehnt: 1 x runs Vektor mit Gesamttestentscheidung

                Output: t1: 3-dim. Array aus 0/1 der fuer jeden Testdurchlauf festhaelt wo Test angenommen/abgelehnt
                        t2: 3-dim. Array aus 0/1 der fuer jeden Testdurchlauf festhaelt wo Test angenommen/abgelehnt
                        abgelehnt: 1 x runs Vektor mit Gesamttestentscheidung
                        DD: Fensterzahl bei Ablehnung
                        G: wie vieltes Fenster lehnt ab
                        """

    Total='Nullhypothese kann nicht abgelehnt werden.'

#Bonferroni Adjustment alpha
    B = np.zeros(Fenster-2)
    G = np.zeros(Fenster - 2)



    for r in range(2,Fenster):
        p1 = sort(first, n1, r)
        p2 = sort(second, n1, r)
# Pruefe gleiche Zellen gegeinander. pij vs pij

        for i in range(0,r):
            for j in range (0,r):
                c= p1[i,j]
                d= p2[i,j]
                if c*n1<=0 or d*n2<=0:
                    T= 0
                elif n1==0:
                    T = ((c - d) ** 2) / ( d * (1 - d) / n2)
                elif n2==0:
                    T = ((c - d) ** 2) / ((c * (1 - c) / n1))
                else:
                    T = ((c-d)**2)/((c*(1-c)/n1 + d*(1-d)/n2))

                p_value = 1-scipy.stats.f.cdf(T, 1,min(n1,n2)-5)

                t1[i,j,h]=p_value

# Pruefe pij vs pji (exchangeability of copula)

        for i in range(0, r):
            for j in range(0, r):
                c = p1[i, j]
                d = p2[j, i]
                if c*n1 <= 0 or d*n1 <= 0:
                    T = 0
                elif n1== 0:
                    T = ((c - d) ** 2) / (d * (1 - d) / n2)

                elif n2 == 0:
                    T = ((c - d) ** 2) / ((c * (1 - c) / n1))

                else:
                    T = ((c - d) ** 2) / ((c * (1 - c) / n1 + d * (1 - d) / n2))

                p_value = 1-scipy.stats.f.cdf(T,1,min(n1,n2)-5)

                t2[i,j,h]=p_value

# Trage Ergebnis in Ergebnisvektor h ein
#build array with p-values
        A=np.zeros(2*(r**2)-r)
        k=0
        for i in range(0, r):
            for j in range(0, r):
                A[k]=t1[i,j,h]
                k=k+1
        for i in range(0, r):
            for j in range(0, r):
                if i!=j:
                    A[k]=t2[i,j,h]
                    k=k+1

        reject, pvals = statsmodels.stats.multitest.fdrcorrection(A, alpha=0.05, method='indep',is_sorted=False)
# smalest p-value of test
        B[r-2]=np.amin(pvals)
# position of this value
        Ghelp=list(pvals)
        G[r-2]=Ghelp.index(np.amin(pvals))


#Gesamtvektor bauen
    DD=None
    if any(B<alphatotal):
        abgelehnt[h]=1
#Höchste Fensterzahl für Ablehnung = DD
        AA=np.asarray(np.where((B < 0.05)))
        D=AA[0]
        DD=D[-1]

    return t1,t2,abgelehnt,DD,G

def Chitest(alphatotal, first,second , n1, n2, Fenster, h, abgelehnt):
    """    Berechnet Teststatistik fuer jedes Fenster und vergleicht mit kritischem Wert
            Erzeugt Detailansicht t1/t2 je Testdurchlauf mit Ablehnungswahrscheinlichkeit je Fenster - nach fdr correction
            Erzeugt Vektor h mit ablehnung(1) oder annahme von H0 fuer jeden run -Chi² test

                Input:  alphatotal:
                        first: erste copula
                        second: zweite copula
                        n1: Sample Size Copula1
                        n2: Sample Size Copula2
                        Fenster: Anzahl Fenster in x,y-Richtung
                        h: Laufvariable fuer Anzahl runs
                        abgelehnt: 1 x runs Vektor mit Gesamttestentscheidung

                Output: abgelehnt: 1 x runs Vektor mit Gesamttestentscheidung
    """
    p1 = sort(first, n1, Fenster)
    p2 = sort(second, n2, Fenster)
    T=0
    for i in range(0, Fenster):
        if T==None:
            break
        for j in range(0, Fenster):
            c = p1[i, j]*n1
            x=p1[i, j]
            y=p2[i, j]
            d = p2[i, j]*n2
            if c  <= 5 or d  <= 5:
                T = None
                break
            else:
                T = T+(((c - d) ** 2) / (d))

    if T!=None:
        p_value = scipy.stats.chi2.cdf(T, (((Fenster-1)**2)-1))

        if p_value >1- alphatotal:
            abgelehnt[h] = 1
    else:
        abgelehnt[h]=None
    return abgelehnt