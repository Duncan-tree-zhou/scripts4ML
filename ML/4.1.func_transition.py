import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
from scipy import stats


if __name__ == "__main__":
    pai1 = np.array([0.25,0.61,0.14])
    # pai1 = np.array([0.21,0.68,0.11])
    pai = np.mat(pai1)
    P1 = np.array([[0.65,0.28,0.07],[0.15,0.67,0.18],[0.12,0.36,0.52]])
    P =  np.mat(P1)
    print 'pai=%s'%pai
    print 'P=%s'%P
    print 'pai*P=%s'%(pai*P)
    print 'pai*P^2=%s'%(pai*P*P)
    print 'pai*P^3=%s'%(pai*P*P*P)
    print 'pai*P^4=%s'%(pai*P*P*P*P)
    print 'pai*P^5=%s'%(pai*P*P*P*P*P)
    print 'pai*P^6=%s'%(pai*P*P*P*P*P*P)
    print 'pai*P^7=%s'%(pai*P*P*P*P*P*P*P)
    print 'pai*P^8=%s'%(pai*P*P*P*P*P*P*P*P)
    print 'pai*P^9=%s'%(pai*P*P*P*P*P*P*P*P*P)
    print 'pai*P^10=%s'%(pai*P*P*P*P*P*P*P*P*P*P)
    print 'pai*P^11=%s'%(pai*P*P*P*P*P*P*P*P*P*P*P)
    print 'pai*P^12=%s'%(pai*P*P*P*P*P*P*P*P*P*P*P*P)