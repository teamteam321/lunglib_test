import numpy as np
import math


def axe_len(k,spacing):
    x = k[0]
    y = k[1]
    z = k[2]
    
    # not sure but close enough for real ellipsoid
    constant = 4.494653677675146

    C = np.vstack([x,y,z])
    COV = np.cov(C)
    eivalue, eivector = np.linalg.eig(COV)
    
    x_len = constant*math.sqrt(eivalue[0]) * spacing[0]
    y_len = constant*math.sqrt(eivalue[1]) * spacing[1]
    z_len = constant*math.sqrt(eivalue[2]) * spacing[2]
    
    return (x_len,y_len,z_len), eivector
    #eivalue = np.sort(eivalue)[::-1]
    #axe_lens = (4.473 * math.sqrt(eivalue[0]),4.588877802 * math.sqrt(eivalue[1]), 4 * math.sqrt(eivalue[2]))
    