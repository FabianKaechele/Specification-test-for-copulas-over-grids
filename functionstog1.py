# Fabian Kaechele

""" functions for basic overall test
                        """
import numpy as np
import scipy
from statsmodels.distributions.empirical_distribution import ECDF
import sys
import copulafunc
import statsmodels
from scipy import stats
import random
import garch

import matplotlib.pyplot as plt


def generateCopulas(family1, family2,n1,n2, tau1,tau2,pr1,pr2,df1,df2, mix):
    """ Erzeugt gewuenschte Copulas durch bedinge Inversionsmethode und plottet Copulas
        Input:  family1: Typ der ersten Copula
                familiy2: Typ der zweiten Copula
                n1: Sample Size Copula 1
                n2:Sample SIze Copula 2
                tau1: shape parameter copula 1
                tau2: shape parameter copula 2
                pr1: shape parameter copula 1
                pr2: shape parameter copula 2
                df1: Degrees of freedom copula 1
                df2: Degrees of freedom copula 1
                mix: mixing parameter
        Output: first: values of first copula u,v
                second: values of first copula u,v"""

    schieben=0

# Erzuege erstes Copula-Objekt
    one = copulafunc.Copula(tau1,pr1,df1,family1)
    U, W = marginals(n1)
    u,v = one.generate_uv(U,W, n1)

# Schieben des Grid

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



#    Grafik erzeugen
    runs=1
    if runs==1 and n1==3000:
        fig = plt.figure()
        plt.scatter(u,v,marker='.',color='blue',s=1)
        plt.ylim(0,1)
        plt.xlim(0,1)
#
#         x = u
#         y = v
#
#         # fig = plt.figure()
#         #
#         # gs = GridSpec(4, 4)
#         #
#         # # ax_joint = fig.add_subplot(gs[1:4, 0:3])
#         # # ax_marg_x = fig.add_subplot(gs[0, 0:3])
#         # # ax_marg_y = fig.add_subplot(gs[1:4, 3])
#         #
#         # ax_joint.scatter(x, y,marker='.',color='blue',s=0.5)
#         # ax_marg_x.hist(x,color='blue')
#         # ax_marg_y.hist(y, orientation="horizontal",color='blue')
#
#         # Turn off tick labels on marginals
#         # plt.setp(ax_marg_x.get_xticklabels(), visible=False)
#         # plt.setp(ax_marg_y.get_yticklabels(), visible=False)
#         #
#         # # Set labels on joint
#         # ax_joint.set_xlabel('u')
#         # ax_joint.set_ylabel('v')
#         #
#         # # Set labels on marginals
#         # ax_marg_y.set_xlabel('marginal u')
#         # ax_marg_x.set_ylabel('marginal v')
#         # #plt.show()
#
        font = {'fontname': 'Times New Roman', 'size':'20'}
#
#
#
#         #
#         #
        if family1=='frank':
            plt.title( 'Frank Copula \u03C4 = {}'.format(round(one.tau,2)), **font)
        elif family1=='gumbel':
            plt.title( 'Gumbel Copula \u03C4 = {}'.format(round(one.tau,2)), **font)
        elif family1=='clayton':
            plt.title( 'Clayton Copula \u03C4 = {}'.format(round(one.tau,2)), **font)
        elif family1=='normal':
            plt.title('Gaus Copula \u03C1 = {}'.format(round(one.pr, 2)), **font)
        elif family1=='t':
            titles= 't-Copula \u03C1 = {}'.format(round(one.pr, 2), **font)
            degrees=str(df1)
            plt.title(titles +', df = '+degrees, **font)
        elif family1 == 'independent':
            plt.title('Independence Copula', **font)
        plt.savefig('abc.png', dpi=800)
    first = np.transpose(np.vstack([u, v]))

# Erzeuge zweites Copula-Objekt
    U, W= marginals(n2)
    two = copulafunc.Copula(tau2,pr2,df2,family2)
    u, v = (1 - mix) * np.array(two.generate_uv(U, W, n2)) + mix * np.array(one.generate_uv(U, W, n2))
# Schieben des Grid

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


    # Grafik erzeugen
    # if runs==1 and n2==1000:
    #     fig.add_subplot(2,2,2)
    #    # plt.hist(v)
    #     plt.scatter(u,v,marker='.',color='blue',s=4)
    #    #plt.ylim(0,1)
    #     #plt.xlim(0,1)
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

    second = np.transpose(np.vstack([u, v]))


    return first, second
def sort(b,n,Fenster):
    """ Sortiert Datenpunkte in Fenster ein und berechnet Häufigkeiten. Wird von generateCopula() aufgerufen
            Input:  a: Matrix aus u,v
                    n: Stichprobengröße
                    Fenster: Anzahl Fenster in x,y-Richtung
            Output: p: Matrix mit Häufigkeiten
                    """
    a=np.copy(b)
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


    B = np.zeros(Fenster-2)
    G = np.zeros(Fenster - 2)



    for r in range(2,Fenster):
        p1 = sort(first, n1, r)
        p2 = sort(second, n1, r)
# Pruefe gleiche Zellen gegeinander. pij vs pij
        #print('Test: pij vs pij')
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
        #print('Test: pij vs pji')
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
        A=np.zeros(2*(r**2))
        k=0
        for i in range(0, r):
            for j in range(0, r):
                A[k]=t1[i,j,h]
                k=k+1
        for i in range(0, r):
            for j in range(0, r):
                if i==j:
                    A[k]=1
                    k = k + 1
                else:
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
    T1=0
    g=0

    for i in range(0, Fenster):
        if T==None:
            break
        for j in range(0, Fenster):
            c = p1[i, j]*n1
            d = p2[i, j]*n2
            if d<5:
                if (g/(Fenster**2)>0.2 or d<=2):
                    T = None
                    break
                g=g+1
            T = T+(((c - d) ** 2) / (d+c))


    if T!=None:
        df=((Fenster-1)**2)

        p_value = scipy.stats.chi2.cdf(T, df)

        if p_value > 1- alphatotal:
            abgelehnt[h] = 1
    else:
        abgelehnt[h]=None
    return abgelehnt