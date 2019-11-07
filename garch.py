# Fabian Kaechele  06.11.2019

""" This file contains a function to create data out of a GARCH(1,1)-process
                          """
import numpy as np

def garch(ω, α, β, n_out=1000):
    """ Creates data out of a GARCH(1,1)-process
    Input: Garch Parameters, smaple size
    Output: Data out of desired GARCH-Process
                              """
    p = len(α)
    q = len(β)


    # Größe Einbrennphase
    n_pre = 1000
    n = n_pre + n_out


    # Sample noise
    ɛ = np.random.standard_normal(size=n)#st.t.rvs(df=5,size=n))
         #st.t.var(df=5))#np.random.normal(size=n)


    y = np.zeros(n)
    σ = np.zeros(n)

    #Start-Values
    for k in range(np.max([p, q])):
        σ[k] = (np.random.standard_normal())

    #Create GARCH data

    for k in range(np.max([p, q]), n):
        α_term = sum([α[i] * y[k-i-1]**2 for i in range(p)])
        β_term = sum([β[i] * ɛ[k-i-1]**2 for i in range(q)])
        σ[k] = np.sqrt(ω + α_term + β_term)
        y[k] = σ[k] * ɛ[k]


    return y[n_pre:]

