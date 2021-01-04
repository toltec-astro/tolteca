from scipy.interpolate import interp1d as interp
import numpy as np

#These elevation models were generated with the am code
#https://www.cfa.harvard.edu/~spaine/am/
#UMass folks can use it on gtx - supporting LMT data in /data/wilson/am
#The npz files referenced below contain evaluations at the LMT
#between 20 and 80 degree elevations in steps of 2 degrees.
#This class linearly interpolates for the input elevation and then
#provides interpolating functions with frequency (in GHz) as the input.

class LMTAtmosphere:

    def __init__(self,
                 path='./',
                 quartile=50.,
                 elevation=70.):

        #atmosphere quartile
        if(quartile==50):
            npzFile=path+'amLMT50.npz'
        elif(quartile==25):
            npzFile=path+'amLMT25.npz'
        elif(quartile==75):
            npzFile=path+'amLMT75.npz'
        else:
            raise ValueError('Only quartiles 25, 50 and 75 are supported.')

        #elevation check
        if(elevation > 80 or elevation < 20):
            raise ValueError('Elevation must be between 20 and 80 deg.')

        self.npzFile = npzFile
        self.elevation = elevation

        #grab the saved data
        df = np.load(self.npzFile)
        el = df['el']
        self.f_GHz = df['atmFreq'][:,0]
        T_atm = df['atmTRJ']
        atm_tx = df['atmTtx']

        #deal with elevation match first
        i = np.searchsorted(el,elevation)
        if(abs(el[i]-elevation) < 0.001):
           self.T = interp(self.f_GHz, T_atm[:,i])
           self.tx = interp(self.f_GHz, atm_tx[:,i])
        else:
           #got to manually interpolate over elevation
           Thigh = T_atm[:,i]
           Tlow = T_atm[:,i-1]
           txhigh = atm_tx[:,i]
           txlow = atm_tx[:,i-1]
           TatEl = (Tlow*(el[i]-elevation) + Thigh*(elevation-el[i-1]))/ \
                   (el[i]-el[i-1])
           txatEl = (txlow*(el[i]-elevation) + txhigh*(elevation-el[i-1]))/ \
                    (el[i]-el[i-1])

           self.T = interp(self.f_GHz, TatEl)
           self.tx = interp(self.f_GHz, txatEl)
        

    
