
import multiprocessing as mp



# Fabian Kaechele

import numpy as np
import functionstog1
import functionstog1bonf
import functionstog1gedreht
import functionstog1chiaggregated
import functionsoverlaptog
import functionsbonf
import functionsbh
import time

# Set values for parallel computing
numbercores=40
runs=np.zeros(shape=(1,numbercores))
#runs=np.full((1,numbercores),25)
runs[0,:]=25
runs=runs.astype(int)
runs=runs.tolist()
runs = [val for sublist in runs for val in sublist]
start = time.time()

def simu(runs):



    # Parameter for Simulation
    # Alpha Value
    alphatotal=0.05
    # Kendalls tau for archim. copula 1
    tau1 = 0.3
    # Kendalls tau for archim. copula 2
    tau2 = 0.9
    # pearson corr for t/gauss copula 1
    pr1= tau1
    # pearson corr for t/gauss copula 2
    pr2= tau2
    # degrees of freedom for t-copula 1
    df1=3
    # degrees of freedom for t-copula 2
    df2=3
    # Typ copula 1
    family1='normal'
    # Typ copula 2
    family2='t'
    # number of runs
    runs=runs





    # Start simulation
    print('Simulationsergebnisse fuer '+ family1+'-Copula (p='+str(pr1)+ ') und '+family2+'-Copula (p='+str(pr2)+') mit '+str(runs)+' Durchgaengen zum Niveau Alpha='+str(alphatotal))

    lauf=0
    # Create result array
    ergebnisges=np.zeros(shape=(5, 5,17))

    # for window in range(4,16):
    for window in (5,8,10,15):
        Fenster=window
        ergebnis = np.zeros(shape=(5, 5))
        lauf1=1

        # Set mixing values (for copula mixing)
        for m in (0,1):
            mix=m


            for n in (250,500,1000):

                n1=n
                n2=n

                abgelehnt = np.zeros(shape=(runs))
                t1 = np.zeros(shape=(Fenster, Fenster, runs))
                t2 = np.zeros(shape=(Fenster, Fenster, runs))



                # run test runs-times and save result
                for h in range(0,runs):

                # Create copulas and select variant of testing

                # Standart Settings

                    first,second = functionstog1.generateCopulas(family1, family2, n1, n2,  tau1, tau2, pr1, pr2, df1, df2,  mix)
                    t1, t2, abgelehnt,DD,G= functionstog1.test(alphatotal, first,second, n1, n2, t1, t2, Fenster, h, abgelehnt)

                # Standart test with Bonf-Adjustemnt
                    t1, t2, abgelehnt,DD,G= functionstog1bonf.test(alphatotal, first,second, n1, n2, t1, t2, Fenster, h, abgelehnt)

                # Standart test with rotated clayton
                    # first,second = functionstog1gedreht.generateCopulas(family1, family2, n1, n2,  tau1, tau2, pr1, pr2, df1, df2,  mix)
                    # t1, t2, abgelehnt,DD,G= functionstog1gedreht.test(alphatotal, first,second, n1, n2, t1, t2, Fenster, h, abgelehnt)

                # Overlapping
                    #t1, t2, abgelehnt,DD,G= functionsoverlaptog.test(alphatotal, first,second, n1, n2, t1, t2, Fenster, h, abgelehnt)

                # Chi-test variantes
                    #abgelehnt = functionstog1chiaggregated.Chitest(alphatotal, first,second, n1, n2, Fenster, h, abgelehnt)
                    #abgelehnt = functionsoverlaptog1.Chitest(alphatotal, first,second, n1, n2, Fenster, h, abgelehnt)


                # Only baseline test different alpha adjustments
                    #p1, p2 = functionsbh.generateCopulas(family1, family2,n1,n2,h, tau1,tau2,pr1,pr2,df1,df2,runs, Fenster, mix)
                    #t1, t2, abgelehnt = functionsbh.test(alphatotal, p1, p2, n1, n2, t1, t2, Fenster, h, abgelehnt)
                    #t1, t2, abgelehnt = functionsbh.test(alphatotal, p1, p2, n1, n2, t1, t2, Fenster, h, abgelehnt)


                    list(abgelehnt)
                    abgelehnt1=[x for x in abgelehnt if x is not None]



                # Create array with results for table
                if n == 250:
                    ergebnis[0, 0] = n
                    ergebnis[0, lauf1] = sum(abgelehnt1) / (len(abgelehnt1)) * 100
                elif n == 500:
                    ergebnis[1, 0] = n
                    ergebnis[1, lauf1] = sum(abgelehnt1) / (len(abgelehnt1)) * 100
                elif n == 1000:
                    ergebnis[2, 0] = n
                    ergebnis[2, lauf1] = sum(abgelehnt1) / (len(abgelehnt1)) * 100
                elif n == 5000:
                    ergebnis[3, 0] = n
                    ergebnis[3, lauf1] = sum(abgelehnt1) / (len(abgelehnt1)) * 100
                elif n == 10000:
                    ergebnis[4, 0] = n
                    ergebnis[4, lauf1] = sum(abgelehnt1) / (len(abgelehnt1)) * 100

            lauf1=lauf1+1

        ergebnisges[:,:,lauf]=ergebnis
        lauf=lauf+1
        # print results
        print('Simulationsergebnisse fuer ' + str(Fenster) + ' Fenster.')
        print(ergebnis)
    return ergebnisges







# Main
if __name__ == '__main__':
    pool=mp.Pool(processes=numbercores)
    ergebnisgesges=(pool.map(simu, runs))
    pool.close()
    pool.join()
    ergebnisend=sum(ergebnisgesges)/numbercores

# Save data in csv
    for k in range(0, 5):
        a = ergebnisend[:, k, :]
        name = str(k) + '.csv'
        np.savetxt(name, a, delimiter=",")

# Analyze run time
    ende = time.time()
    print('Dauer in Sekunden: ' + str(ende - start))