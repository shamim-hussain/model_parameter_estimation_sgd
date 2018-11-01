# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:54:52 2018

@author: snirj
"""

import numpy as np

def arburg(x, n):
    ep=x[1:]
    em=x[0:-1]
    Dk=np.sum(ep**2)+np.sum(em**2)
    ae=np.zeros(shape=(n+1,), dtype=float)
    ri=np.zeros(shape=(n,), dtype=float)
    ae[0] = 1.
    ne=np.sum(x*x)/len(x)
    for k in range(n):
        g=-2*np.sum(ep*em)/(Dk+1e-4)
        if g>=1. or g<=-1.:
            ae[:]=0.
            ri[:]=0.
            ne=0.
            break
        ri[k]=g
        ep,em=(ep+g*em)[1:],(em+g*ep)[0:-1]
        Dk=(1-g**2)*Dk-ep[0]-em[-1]
        ae[0:k+2] += g*np.flip(ae[0:k+2],0)
        ne=(1-g**2)*ne
    return ae, ne, ri

arburgv=np.vectorize(arburg, signature='(m),()->(n),(),(p)')