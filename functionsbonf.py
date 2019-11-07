# Fabian Kaechele

""" use the functions out of this file to perform the baseline test, without the selection step and Bonferoni adjustment
                        """
import numpy as np
import scipy
from statsmodels.distributions.empirical_distribution import ECDF
import sys
import copulafunc
import random


#import matplotlib.pyplot as plt

def generateCopulas(family1, family2,n1,n2,h, tau1,tau2,pr1,pr2,df1,df2,runs, Fenster, mix):
    """ Erzeugt gewuenschte Copulas durch bedinge Inversionsmethode und plottet Copulas
        Input:  family1: Typ der ersten Copula
                familiy2: Typ der zweiten Copula
                n1: Sample Size Copula 1
                n2:Sample SIze Copula 2
                h: Laufvariable fuer Anzahl runs
                tau1:
                tau2:
                pr1:
                pr2:
                df1:
                df2:
        Output: p1: Matrix mit Haeufigkeiten der ersten Copula
                p2: Matrix mit Haeufigkeiten der zweiten Copula
       """

    schieben=0

# Erzuege erstes Copula-Objekt
    one = copulafunc.Copula(tau1,pr1,df1,family1)
    U, W = marginals(n1)
    u,v = one.generate_uv(U,W, n1)

# Schieben des Grid

    if schieben != 0:
        u = u + schieben
        v = v + schieben
        for i in range(0, n1):
            if u[i] >= 1:
                u[i] = 1 - sys.float_info.epsilon
            if u[i] <= 0:
                u[i] = 0 + sys.float_info.epsilon
            if v[i] >= 1:
                v[i] = 1 - sys.float_info.epsilon
            if v[i] <= 0:
                v[i] = 0 + sys.float_info.epsilon


# # Grafik erzeugen
    # if runs==1 and n1==1000:
    #     fig = plt.figure()
    #     fig.add_subplot(2,2,1)
    #     plt.scatter(u,v,marker='.',color='blue')
    #     plt.ylim(0,1)
    #     plt.xlim(0,1)
    #     if family1=='frank':
    #         plt.title( 'Frank Copula \u03C4 = {}'.format(round(one.tau,2)))
    #     elif family1=='gumbel':
    #         plt.title( 'Gumbel Copula \u03C4 = {}'.format(round(one.tau,2)))
    #     elif family1=='clayton':
    #         plt.title( 'Clayton Copula \u03C4 = {}'.format(round(one.tau,2)))
    #     elif family1=='normal':
    #         plt.title('Gaus Copula \u03C1 = {}'.format(round(one.pr, 2)))
    #     elif family1=='t':
    #         titles= 't-Copula \u03C1 = {}'.format(round(one.pr, 2))
    #         degrees=str(df1)
    #         plt.title(titles +', df = '+degrees)
    #     elif family1 == 'independent':
    #         plt.title('Independence Copula')

    a = np.transpose(np.vstack([u, v]))

# Einsortieren von U,V
    p1 = sort(a,n1, Fenster)


# Erzeuge zweites Copula-Objekt
    U, W= marginals(n2)
    #U = np.random.uniform(size=n2)
   # W = np.random.uniform(size=n2)
    two = copulafunc.Copula(tau2,pr2,df2,family2)
    u,v = (1-mix)*np.array(two.generate_uv(U,W, n2))+mix*np.array(one.generate_uv(U,W, n2))

# Schieben des Grid
    if schieben !=0:
        u = u + schieben
        v = v + schieben
        for i in range(0, n1):
            if u[i] >= 1:
                u[i] = 1 - sys.float_info.epsilon
            if u[i] <= 0:
                u[i] = 0 + sys.float_info.epsilon
            if v[i] >= 1:
                v[i] = 1 - sys.float_info.epsilon
            if v[i] <= 0:
                v[i] = 0 + sys.float_info.epsilon

# # Grafik erzeugen
    # if runs==1 and n2==1000:
    #     fig.add_subplot(2,2,2)
    #     plt.scatter(u,v,marker='.',color='blue')
    #     plt.ylim(0,1)
    #     plt.xlim(0,1)
    #     if mix != 1 and mix!= 0:
    #         plt.title('Mixed Copula')
    #     elif mix==1:
    #         if family1 == 'frank':
    #             plt.title('Frank Copula \u03C4 = {}'.format(round(one.tau, 2)))
    #         elif family1 == 'gumbel':
    #             plt.title('Gumbel Copula \u03C4 = {}'.format(round(one.tau, 2)))
    #         elif family1 == 'clayton':
    #             plt.title('Clayton Copula \u03C4 = {}'.format(round(one.tau, 2)))
    #         elif family1 == 'normal':
    #             plt.title('Gaus Copula \u03C1 = {}'.format(round(one.pr, 2)))
    #         elif family1 == 't':
    #             titles = 't-Copula \u03C1 = {}'.format(round(one.pr, 2))
    #             degrees = str(df1)
    #             plt.title(titles + ', df = ' + degrees)
    #         elif family1 == 'independent':
    #             plt.title('Independence Copula')
    #     elif family2 == 'frank':
    #         plt.title('Frank Copula \u03C4 = {}'.format(round(two.tau, 2)))
    #     elif family2 == 'gumbel':
    #         plt.title('Gumbel Copula \u03C4 = {}'.format(round(two.tau, 2)))
    #     elif family2 == 'clayton':
    #         plt.title('Clayton Copula \u03C4 = {}'.format(round(two.tau, 2)))
    #     elif family2 == 'normal':
    #         plt.title('Gaus Copula \u03C1 = {}'.format(round(two.pr, 2)))
    #     elif family2=='t':
    #         titles = 't-Copula \u03C1 = {}'.format(round(two.pr, 2))
    #         degrees = str(df2)
    #         plt.title(titles + ', df = ' + degrees)
    #     elif family2 == 'independent':
    #         plt.title('Independence Copula')
    #
    #     plt.show()

    a = np.transpose(np.vstack([u, v]))
        # Einsortieren von U,V
    p2= sort(a,n2, Fenster)

    return p1,p2
def sort(a,n,Fenster):
    """ Sortiert Datenpunkte in Fenster ein und berechnet Häufigkeiten. Wird von generateCopula() aufgerufen
            Input:  a: Matrix aus u,v
                    n: Stichprobengröße
                    Fenster: Anzahl Fenster in x,y-Richtung
            Output: p: Matrix mit Häufigkeiten
                    """

# Leere Matrix mit Whk. erzeugen
    p = np.zeros(shape = (Fenster, Fenster))

# u,v in Zielkoordinaten wandeln
    for i in range(0,n):
        for j in range(0,2):
            a[i,j]= np.floor(a[i,j]*Fenster)

# Matrix p füllen
    for ab in range(0, Fenster):
        for ac in range(0, Fenster):
            for i in range(0, n):
                    if (a[i, 0] == ab) and (a[i, 1] == ac):
                        p[ab, ac]= p[ab, ac]+1

    p = p / n
    return p

def test(alphatotal,p1,p2,n1,n2,t1,t2,Fenster,h,abgelehnt):
    """ Berechnet Teststatistik fuer jedes Fenster und vergleicht mit kritischem Wert
        Erzeugt Detailansicht t1/t2 je Testdurchlauf mit Ablehnungswahrscheinlichkeit je Fenster
        Erzeugt Vektor h mit ablehnung(1) oder annahme von H0 fuer jeden run

                Input:  alphatotal:
                        p1: Fenster x Fenster Matrix mit Haeufigkeiten
                        p2: Fenster x Fenster Matrix mit Haeufigkeiten
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
                        """

    Total='Nullhypothese kann nicht abgelehnt werden.'

#Bonferroni Adjustment alpha

    alphabonf = alphatotal / ((Fenster ** 2)*2-Fenster)
    #print(alphabonf)

# Pruefe gleiche Zellen gegeinander. pij vs pij

    for i in range(0,Fenster):
        for j in range (0,Fenster):
            c= p1[i,j]
            d= p2[i,j]
            if c<=0 or d<=0:
                T= 0
            elif n1==0:
                T = ((c - d) ** 2) / ( d * (1 - d) / n2)
            elif n2==0:
                T = ((c - d) ** 2) / ((c * (1 - c) / n1))
            else:
                T = ((c-d)**2)/((c*(1-c)/n1 + d*(1-d)/n2))

            p_value = 1-scipy.stats.f.cdf(T, 1,min(n1,n2)-5)

            if (p_value*(2*(Fenster**2)-Fenster)) <alphatotal:

                Total = 'Nullhypothese wird abgelehnt!'
                t1[i,j,h]=t1[i,j,h]+1

            else:
                t1[i, j,h] = t1[i, j,h]

    # Pruefe pij vs pji (exchangeability of copula)

    for i in range(0, Fenster):
        for j in range(0, Fenster):
            c = p1[i, j]
            d = p2[j, i]
            if c <= 0 and d <= 0:
                T = 0
            elif n1== 0:
                T = ((c - d) ** 2) / (d * (1 - d) / n2)

            elif n2 == 0:
                T = ((c - d) ** 2) / ((c * (1 - c) / n1))

            else:
                T = ((c - d) ** 2) / ((c * (1 - c) / n1 + d * (1 - d) / n2))

            p_value = 1-scipy.stats.f.cdf(T,1,min(n1,n2)-5)


            if (p_value*(2*(Fenster**2)-Fenster)) < alphatotal:
                Total = 'Nullhypothese wird abgelehnt!'
                t2[i,j,h]=t2[i,j,h]+1

            else:
                t2[i, j,h] = t2[i, j,h]


# Trage Ergebnis in Ergebnisvektor h ein
    if Total == 'Nullhypothese wird abgelehnt!':
        abgelehnt[h]=1
    return t1,t2,abgelehnt


def marginals(n):
    """ Festlegen welche marginals verwendet werden.
                    Input:  n=      Größe

                    Output: u,v=    mit eigener empirischer Verteilungsfunktion transformierte Daten entsprechend marginals in Größe n
                            """


#Standart t
    np.random.seed(random.randrange(1, 100000, 2))
    a = np.random.standard_t(df=3,size=n)
    np.random.seed(random.randrange(1, 100000, 2))
    b = np.random.standard_t(df=3,size=n)
# GARCH
    ω = 0.01
    α = [0.15]
    β = [0.8]
    #a = garch.garch(ω, α, β, n)
    #b = garch.garch(ω, α, β, n)



    #fixed_res = model.fit(disp='off')
    #a=fixed_res.resid
    #model = arch_model(b)
    #fixed_res = model.fit(disp='off')
    #b= fixed_res.resid

# calculate u,w
    e = ECDF(a)
    e2 = ECDF(b)
    u = e(a)
    w = e2(b)

    return u, w
