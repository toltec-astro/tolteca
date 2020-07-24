import numpy as np
import matplotlib as mpl
import matplotlib.axes as axs
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
import netCDF4
import glob
from astropy.table import Table
from scipy.stats import sigmaclip
#from photutils import DAOStarFinder
from astropy.table import Table
import subprocess
from scipy import signal
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
from matplotlib.patches import Ellipse
import re

class obs:
    def __init__(self,obsnum=None,nrows=None,ncols=None,path=None,
                 index=None,sampfreq=None,order=None,transpose=None):

        self.obsnum = str(obsnum)
        self.nrows = nrows
        self.ncols = ncols
        self.path = path
        self.index = index
        self.sampfreq = sampfreq
        self.obsnum = obsnum
        
        self.pnames = ['x','y','fwhmx','fwhmy','amps','snr']
        self.nc_pnames = ["amplitude", "FWHM_x", "FWHM_y", "offset_x", "offset_y"] #, "bolo_name"]

        if self.path != None:
            self.get_obs_data(order, transpose)
        elif self.index != None:
            self.from_index(order, transpose)

    def from_index(self, order, transpose):
        self.ncs = []
        self.nws = []

        for i in range(len(self.index['nw_names'])):
            nw = self.index['nw_names'][i]
            self.nws.append(str(nw))
            dx,dy,df = self.get_design(nw)

            f = self.index['nw_path'][nw]['path']
            print('reading in  %s' % (f))
            try:
                nc = ncdata(f,self.obsnum,self.nrows,self.ncols, nw,
                            self.sampfreq, order, transpose,dx, dy, df)
                self.ncs.append(nc)
            except:
                print('file not found %s ' % (f))
                self.ncs.append(-1)

        
    def get_obs_data(self,order,transpose):
        self.beammap_files = np.sort(glob.glob(self.path+str(self.obsnum)+'/*.nc'))
        self.raw_files = np.sort(glob.glob(self.path[:-6]+'/data/'+str(self.obsnum)+'/*processed.nc'))
        
        print('Getting data from %i files for obsnum %s' %(len(self.beammap_files),self.obsnum))

        self.ncs = list(np.ones(13)*-1)
        self.nws = list(np.ones(13)*-1)

        #self.tdets = 0
        for f in self.beammap_files:
            nw = re.findall(r'\d+', f)   
            self.nws[int(nw[-1])] = nw[-1]
            print('on nw ' + self.nws[int(nw[-1])])
            dx,dy,df = self.get_design(int(self.nws[int(nw[-1])]))
            nc = ncdata(f,self.obsnum,self.nrows,self.ncols,self.nws[int(nw[-1])],self.path,self.sampfreq,order,transpose,dx,dy,df)
            self.ncs[int(nw[-1])] = nc
            #self.tdets = self.tdets + self.ncs[int(nw[-1])].ndets

        '''while -1 in self.ncs:
            self.ncs.remove(-1)

        while -1 in self.nws:
            self.nws.remove(-1)
        '''

        '''
        self.p = {}
        self.nw_arr = np.ones(self.tdets)*-99
        for i in range(len(self.pnames)):
            self.p[self.pnames[i]] = np.ones(self.tdets)*-99
            
        m = 0
        for i in range(len(self.ncs)):
            for j in range(self.ncs[i].ndets):
                self.nw_arr[m] = self.nws[i]
                for k in self.pnames:
                    self.p[k][m] = self.ncs[i].p[k][j]
                m = m + 1
        '''
                
    
    def get_design(self,nw,path='default'):
        '''self.designed = Table.read('/Users/mmccrackan/toltec/data/designed_1100.asc',format='ascii.commented_header')
        
        i = np.where(self.designed['nw'] == nw)[0]
        
        if i!=list():
            angle = np.pi/2.
            dy0 = -self.designed['y']
            dx0 = self.designed['x']
            df0 = self.designed['f']
            
            dx = np.cos(angle)*dx0 + np.sin(angle)*dy0
            dy = -np.sin(angle)*dx0 + np.cos(angle)*dy0 
    
            #dx = (7.5*10**-4)*dx + 12.3
            #dy = (7.*10**-4)*dy + 14.5
            
            #df = (0.88*10**3)*df0 +2.*10**1
            #df = df[i]
            df = df0[i]
            
    
            dx = -dx[i]/np.max(dx[i])
            dx = dx - np.mean(dx)
            
            dy = dy[i]/np.max(dy[i])
            dy = dy - np.mean(dy)
            
            return dx,dy,df
        else:
        '''
        return 0,0,0
        
    def obs_limit(self,lim_p,lim,lim_type,refresh=True):
        for nc in self.ncs:
            nc.limit(lim_p,lim,lim_type,refresh)
        
    def obs_scatter(self,x,y,c='k',gi=False,nf=True,marker='s',s=10,log=False):
        
        tdets = np.zeros(len(self.ncs)+1)
        for i in range(len(self.ncs)):
            tdets[i+1] = self.ncs[i].ndets
            
        tx = np.zeros(int(np.sum(tdets)))
        ty = np.zeros(int(np.sum(tdets)))
        tc = np.zeros(int(np.sum(tdets)))
        ts = np.zeros(int(np.sum(tdets)))
        
        cum_tdets = np.cumsum(tdets)
                
        for i in range(1,len(self.ncs)):
            tx[int(cum_tdets[i-1]):int(cum_tdets[i])] = self.ncs[i-1].p[x]
            ty[int(cum_tdets[i-1]):int(cum_tdets[i])] = self.ncs[i-1].p[y]
            
            if c in self.ncs[i].pnames:
                tc[int(cum_tdets[i-1]):int(cum_tdets[i])] = self.ncs[i-1].p[c]
            else:
                tc = c
            if s in self.ncs[i].pnames:
                ts[int(cum_tdets[i-1]):int(cum_tdets[i])] = self.ncs[i-1].p[s]
            else:
                ts = s
        
        if c in self.ncs[0].pnames:
            if log == True:
                tc = np.log10(tc)
        
        if nf == True:
            plt.figure()
        plt.scatter(tx,ty,c=tc,s=ts)
        plt.xlabel(x)
        plt.ylabel(y)
        if c in self.ncs[0].pnames:
            if log == True:
                plt.colorbar(label='log ' + c)
            else:
                plt.colorbar(label=c)
        else:
            plt.colorbar()
                
            
    def obs_hist(self,x,bins=10,nf=True,log=False):        
        
        tdets = np.zeros(len(self.ncs)+1)
        for i in range(len(self.ncs)):
            tdets[i+1] = self.ncs[i].ndets
            
        tx = np.zeros(int(np.sum(tdets)))
        
        cum_tdets = np.cumsum(tdets)
        
        for i in range(1,len(self.ncs)):
            tx[int(cum_tdets[i-1]):int(cum_tdets[i])] = self.ncs[i-1].p[x]      
            
        if nf == True:
            plt.figure()
            
        plt.hist(tx,bins)

        if log == True:
            plt.yscale('log', nonposy='clip')
            
        plt.xlabel(x)
        if log == True:
            plt.ylabel('N')
        else:
            plt.ylabel('N')

             
class ncdata:
    def __init__(self, ncfile_name,obsnum,nrows,ncols,nw,sampfreq,order,transpose,dx,dy,df):
        self.ncfile_name = ncfile_name
        self.obsnum = str(obsnum)
        self.nrows = nrows
        self.ncols = ncols
        self.nw = nw
        #self.path = path
        self.sampfreq = sampfreq
        self.dx = dx
        self.dy = dy
        self.df = df
        
        #self.beammap_files = np.sort(glob.glob(self.path+str(obsnum)+'/*toltec'+nw+'.nc'))
        #self.raw_files = np.sort(glob.glob(self.path[:-6]+'/data/'+str(obsnum)+'/toltec'+nw+'*.nc'))

        self.pnames = ['x','y','fwhmx','fwhmy','amps','snr']
        self.nc_pnames = ["amplitude", "FWHM_x", "FWHM_y", "offset_x", "offset_y"] #, "bolo_name"]
        
        self.map_names = ['x_onoff', 'r_onoff', 'x_off', 'r_off', 'x_on', 'r_on', 'xmap', 'rmap']
        
        self.get_nc_data(order,transpose)
        
    def __getstate__(self):
        # this is to allow pickling of this object
        # so that this can be held in the redis cache.
        state = self.__dict__.copy()
        del state['ncfile']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.ncfile = netCDF4.Dataset(self.ncfile_name)

    def get_nc_data(self,order,transpose):
        self.ncfile = netCDF4.Dataset(self.ncfile_name)
        
        self.ndets = len(self.ncfile.dimensions['ndet'])
        
        self.indices = list(range(self.ndets))
        self.bad_indices = []

        self.get_params()
        self.make_maps(order,transpose)
        self.get_f()
        #self.scale_designed()
        self.x_snr, self.x_snr_amp, self.x_std = self.get_snr(map_type='x')
        self.r_snr, self.r_snr_amp, self.r_std = self.get_snr(map_type='r')

    def scale_designed(self,gi=False):
        
        if gi == False:
            self.dx = np.max(self.p['x'])/2.*self.dx + np.median(self.p['x'])
            self.dy = np.max(self.p['y'])/2.*self.dy + np.median(self.p['y'])
        else:
            self.dx = (np.max(self.p['x'][self.indices]) - np.min(self.p['x'][self.indices]))/4.*self.dx + np.median(self.p['x'][self.indices])
            self.dy = (np.max(self.p['y'][self.indices]) - np.min(self.p['y'][self.indices]))/2.*self.dy + np.median(self.p['y'][self.indices])
    
    def make_maps(self,order,transpose):      
        self.make_x_onoff_map(order)
        self.make_r_onoff_map(order)
        self.make_x_off_map(order)
        self.make_r_off_map(order)
        self.make_xmap(order)
        self.make_rmap(order) 
        
        self.x_on = self.x_onoff * self.x_off
        self.r_on = self.r_onoff * self.r_off
        
        if transpose == True:
            self.x_onoff = np.flip(np.transpose(self.x_onoff,(1, 0, 2)),axis=1)
            self.r_onoff = np.flip(np.transpose(self.r_onoff,(1, 0, 2)),axis=1)
            self.x_off = np.flip(np.transpose(self.x_off,(1, 0, 2)),axis=1)
            self.r_off = np.flip(np.transpose(self.r_off,(1, 0, 2)),axis=1)
            self.x_on = np.flip(np.transpose(self.x_on,(1, 0, 2)),axis=1)
            self.r_on = np.flip(np.transpose(self.r_on,(1, 0, 2)),axis=1)
            self.xmap = np.flip(np.transpose(self.xmap,(1, 0, 2)),axis=1)
            self.rmap = np.flip(np.transpose(self.rmap,(1, 0, 2)),axis=1)
            
            #x_tmp = -self.p['x']
            #self.p['x'] = -self.p['y']
            #self.p['y'] = x_tmp   

    def make_x_onoff_map(self,order='C'):
        tmp = np.array(self.ncfile['x_onoff'][:])
                    
        self.x_onoff = np.zeros([self.nrows,self.ncols,self.ndets])
        print('Generating x_onoff map')

        for i in range(self.ndets):
            self.x_onoff[:,:,i] = np.reshape(tmp[:,i],(self.nrows,self.ncols),order=order)
        self.x_onoff[::2,:,:]= np.flip(self.x_onoff[::2,:,:],axis=1)
    
    def make_r_onoff_map(self,order='C'):
        tmp = np.array(self.ncfile['r_onoff'][:])
                    
        self.r_onoff = np.zeros([self.nrows,self.ncols,self.ndets])
        print('Generating r_onoff map')

        for i in range(self.ndets):
            self.r_onoff[:,:,i] = np.reshape(tmp[:,i],(self.nrows,self.ncols),order=order)
        self.r_onoff[::2,:,:]= np.flip(self.r_onoff[::2,:,:],axis=1)
        
    def make_x_off_map(self,order='C'):
        tmp = np.array(self.ncfile['x_off'][:])
                    
        self.x_off = np.zeros([self.nrows,self.ncols,self.ndets])
        print('Generating x_off map')

        for i in range(self.ndets):
            self.x_off[:,:,i] = np.reshape(tmp[:,i],(self.nrows,self.ncols),order=order)
        self.x_off[::2,:,:]= np.flip(self.x_off[::2,:,:],axis=1)
        
    def make_r_off_map(self,order='C'):
        tmp = np.array(self.ncfile['r_off'][:])
                    
        self.r_off = np.zeros([self.nrows,self.ncols,self.ndets])
        print('Generating r_off map')

        for i in range(self.ndets):
            self.r_off[:,:,i] = np.reshape(tmp[:,i],(self.nrows,self.ncols),order=order)
        self.r_off[::2,:,:]= np.flip(self.r_off[::2,:,:],axis=1)
        
    def make_xmap(self,order='C'):
        try:
            tmp = np.array(self.ncfile['xmap'][:])
                        
            self.xmap = np.zeros([self.nrows,self.ncols,self.ndets])
            print('Generating xmap')
    
            for i in range(self.ndets):
                self.xmap[:,:,i] = np.reshape(tmp[:,i],(self.nrows,self.ncols),order=order)
            self.xmap[::2,:,:]= np.flip(self.xmap[::2,:,:],axis=1)
        
        except:
            self.xmap = np.zeros([self.nrows,self.ncols,self.ndets])

        
    def make_rmap(self,order='C'):
        try:
            tmp = np.array(self.ncfile['rmap'][:])
                        
            self.rmap = np.zeros([self.nrows,self.ncols,self.ndets])
            print('Generating rmap')
    
            for i in range(self.ndets):
                self.rmap[:,:,i] = np.reshape(tmp[:,i],(self.nrows,self.ncols),order=order)
            self.rmap[::2,:,:]= np.flip(self.rmap[::2,:,:],axis=1)
        except:
            self.rmap = np.zeros([self.nrows,self.ncols,self.ndets])

        
    def get_params(self):
        self.p = {}
        
        for i in range(len(self.pnames)):
            self.p[self.pnames[i]] = np.ones(self.ndets)*-99
        
        for i in range(self.ndets):
            map_fit = 'map_fits'+str(i)
            self.p['x'][i] = self.ncfile[map_fit].getncattr('offset_x')
            self.p['y'][i] = self.ncfile[map_fit].getncattr('offset_y')
            self.p['fwhmx'][i] = self.ncfile[map_fit].getncattr('FWHM_x')
            self.p['fwhmy'][i] = self.ncfile[map_fit].getncattr('FWHM_y')
            self.p['amps'][i] = self.ncfile[map_fit].getncattr('amplitude')
                        
    def get_f(self):
        '''try:
            nc_f = netCDF4.Dataset(self.raw_files[0])
            self.f = np.array(nc_f['Header.Toltec.ToneFreq'][:]) + np.array(nc_f['Header.Toltec.LoFreq'])
            self.f = self.f[0]/10**6
        except:
            self.f = np.zeros(self.ndets)
        '''
        
        try:            
            self.f = (self.ncfile['tone_freq'][:] +  self.ncfile['LoFreq'])/10**6.
        except:
            self.f = np.zeros(self.ndets)

        
    def get_snr(self,map_type,sigma_cut=3):
        snr = np.zeros(self.ndets)
        snr_amp = np.zeros(self.ndets)
        std = np.zeros(self.ndets)

        for i in range(self.ndets):
            delta_std = 1
            limit = 1e-5
            if map_type == 'x':
                x_onoff = np.array(self.x_onoff[:,:,i])
            elif map_type == 'r':
                x_onoff = np.array(self.r_onoff[:,:,i])
            iters = 0
            while ((abs(delta_std) >=limit) or iters<10):
                std_old = np.std(x_onoff)
                mean_old = np.mean(x_onoff)
                x_onoff = x_onoff[np.abs(x_onoff - mean_old) < sigma_cut*std_old]
                delta_std = (np.std(x_onoff) - std_old)/std_old
                iters = iters + 1
          
            if map_type == 'x':  
                snr[i] = np.max(self.x_onoff[:,:,i])/np.std(x_onoff)
            elif map_type == 'r':
                snr[i] = np.max(self.r_onoff[:,:,i])/np.std(x_onoff)
            snr_amp[i] = self.p['amps'][i]/np.std(x_onoff)
            std[i] = np.std(x_onoff)
            
        return snr, snr_amp, std
        
    def limit(self,lim_p,lim,lim_type,refresh=True):
        for i in range(len(self.pnames)):
            if lim_p == self.pnames[i]:
                self.lim_ind = i
        
        self.lim_p = self.pnames[self.lim_ind]
                
        if ('lim_p_temp' not in dir(self)) or (refresh==True):
            print('refreshing')
            self.lim_p_temp = copy.deepcopy(self.p)
            self.indices = list(range(self.ndets))
            self.bad_indices = []
        
        else:
            self.lim_p_temp = copy.deepcopy(self.lim_p)
            
        if lim_type == '>':
            self.tid = np.where(self.p[self.lim_p] > lim)[0]
        elif lim_type == '<':
            self.tid = np.where(self.p[self.lim_p] < lim)[0]
        elif lim_type == '=':
            self.tid = np.where(self.p[self.lim_p] == lim)[0]
        elif lim_type == '!=':
            self.tid = np.where(self.p[self.lim_p] != lim)[0]
            
        for i in range(self.ndets):
            if (i not in self.tid) and (i in self.indices):
                self.indices.remove(i)
            if (i not in self.tid) and (i not in self.bad_indices):
                self.bad_indices.append(i)
                    
        print('Len of indices ' + str(len(self.indices)))
        print('Len of bad indices ' + str(len(self.bad_indices)))
        
    def plot_psd(self,det,map_type,use_welch=True,scan_index='None',axes='None',plot=True):
        if map_type == 'x':
            map_to_plot = np.ravel(self.x_onoff[:,:,det])
            dataname = 'Xs'
        elif map_type == 'r':
            map_to_plot = np.ravel(self.r_onoff[:,:,det])
            dataname = 'Rs'
    
        if scan_index != 'None':
            maxpix = scan_index
        else:
            maxpix = np.where(map_to_plot == np.max(map_to_plot))[0]
    
        nc = netCDF4.Dataset(self.raw_files[1])
        
        xs = np.array(nc[dataname][int(self.ncfile['si'][maxpix]):int(self.ncfile['ei'][maxpix]),det])
        xs = xs[:]
        
        if use_welch == False:
            npts = len(xs) 
            hann = 0.5 - 0.5 * np.cos(np.linspace(0, 2.0 * np.pi / npts * (npts - 1),npts))
            n2 = 512
            while n2 < npts:
                n2 = 2*n2
            
            fft = np.fft.fft(xs*hann,n=n2)
            Pxx_den = np.abs(fft)**2.
            Pxx_den = Pxx_den[0:int(len(Pxx_den)/2)+1]/self.sampfreq/1024
            f = np.linspace(0,sf/2.,len(Pxx_den))
        
        else:
            f, Pxx_den = signal.welch(xs,sf,nperseg=128,detrend=False)#,nfft=n2)
        
        
        if plot == True:
            plt.semilogy(f,np.sqrt(Pxx_den))
            plt.xlabel('freq (Hz)')
            plt.ylabel('PSD (V/Hz^1/2)')
            
        return np.mean(Pxx_den)

    def nc_scatter(self,x,y,marker='.',c='b',s=5,gi=False,nf=True,interactive=True,use_ellipse=False,plot_design = True,plot_nearest=False,bi=False,three_d=False,alpha=1):
        if nf == True:
            if three_d == True:
                fig = plt.figure(figsize=(10,8))
                ax = plt.axes(projection='3d')
            else:
                fig,ax = plt.subplots()
            
        if gi == True:
            ind_to_plot = self.indices
        elif bi == False:
            ind_to_plot = list(range(self.ndets))
        else:
            ind_to_plot = self.bad_indices
        
        if three_d == False:    
            if c in self.pnames:
                if s in self.pnames:
                    plt.scatter(self.p[x][ind_to_plot],self.p[y][ind_to_plot],edgecolor=self.p[c][ind_to_plot],marker=marker,s=self.p[s][ind_to_plot],zorder=2,facecolor=self.p[c][ind_to_plot],alpha=alpha)
                else:
                    plt.scatter(self.p[x][ind_to_plot],self.p[y][ind_to_plot],c=self.p[c][ind_to_plot],marker=marker,s=s,zorder=2,alpha=alpha)
                    plt.colorbar()
            else:
                if s in self.pnames:
                    plt.scatter(self.p[x][ind_to_plot],self.p[y][ind_to_plot],edgecolor=c,marker=marker,s=self.p[s][ind_to_plot],zorder=2,facecolor=c,alpha=alpha)
                else:
                    plt.scatter(self.p[x][ind_to_plot],self.p[y][ind_to_plot],edgecolor=c,marker=marker,s=s,zorder=2,facecolor=c,alpha=alpha)
        
        std_dx = 1#0.5**0.5#np.std(self.dx)
        std_dy = 1#0.5**0.5#np.std(self.dy)
        std_df = 1#np.sqrt(1000)#0.0001**0.5#np.std(self.df)
        
        
        #self.dx = self.dx/np.max(self.dx)
        #self.dx = self.dx + np.min(self.dx)
        #self.dy = self.dy/np.max(self.dy)
        #self.dy = self.dy + np.min(self.dy)
        
        n_matched = []
        if plot_design == True:
            if three_d == False:
                plt.scatter(self.dx,self.dy,zorder=0,c='k',s=3,alpha=0.1)
            else:
                ax.scatter(self.dx,self.dy,self.df,zorder=0,c='k',s=3,alpha=0.1)
            if plot_nearest==True:
                if three_d == False:
                    for i in ind_to_plot:
                        #dist = np.sqrt((self.dx - self.p[x][i])**2. + (self.dy - self.p[y][i])**2.)
                        #dist = np.sqrt(((self.dx - self.p[x][i])/self.dx)**2. + ((self.dy - self.p[y][i])/self.dy)**2. + ((self.df - self.f[i])/self.df)**2.)
                        dist = np.sqrt(((self.dx - self.p[x][i])/std_dx)**2. + ((self.dy - self.p[y][i])/std_dy)**2. + ((self.df - self.f[i])/std_df)**2.)
                        j = np.where(dist == np.min(dist))[0][0]                    
                        plt.plot([self.dx[j],self.p[x][i]],[self.dy[j],self.p[y][i]],c='r',zorder=1)
                        plt.scatter(self.dx[j],self.dy[j],c='k',marker='s',s=7,zorder=2)
                        n_matched.append(j)
                else:
                    for i in ind_to_plot:
                        #dist = np.sqrt((self.dx - self.p[x][i])**2. + (self.dy - self.p[y][i])**2.)
                        #dist = np.sqrt(((self.dx - self.p[x][i])/self.dx)**2. + ((self.dy - self.p[y][i])/self.dy)**2. + ((self.df - self.f[i])/self.df)**2.)
                        dist = np.sqrt(((self.dx - self.p[x][i])/std_dx)**2. + ((self.dy - self.p[y][i])/std_dy)**2. + ((self.df - self.f[i])/std_df)**2.)
                        j = np.where(dist == np.min(dist))[0][0]                    
                        ax.plot([self.dx[j],self.p[x][i]],[self.dy[j],self.p[y][i]],[self.f[i],self.df[j]],c='r',zorder=1)
                        ax.scatter(self.dx[j],self.dy[j],self.df[j],c='k',marker='s',s=7,zorder=2)
                        ax.scatter(self.p['x'][i],self.p['y'][i],self.f[i],c='m',marker='s',s=7,zorder=2)
                        n_matched.append(j)
        
        found_dets = 0            
        for j in np.unique(n_matched):
            if n_matched.count(j) == 2:
                found_dets +=1
                
        print('A total of ' + str(int(found_dets)) + ' have likely been identified correctly')
            
        plt.axis('equal')
        plt.xlabel(x)
        plt.ylabel(y)
        
    def plotmap(self,det,sx,sy,map_type,nf=True,s='fwhmx',update=False,gi=False,use_ellipse=False,c='r',marker='x',plot_nearest=False,bi=False,three_d=False):            
        if nf == True:
            fig = plt.figure(figsize=(10,8))
            
        if gi == True:
            ind_to_plot = self.indices
        elif bi == False:
            ind_to_plot = list(range(self.ndets))
        else:
            ind_to_plot = self.bad_indices
            
        if map_type == 'x':
            onoff = self.x_onoff
            off = self.x_off
            on = self.x_on
        elif map_type == 'r':
            onoff = self.r_onoff
            off = self.r_off
            on = self.r_on
            
        gs = gridspec.GridSpec(2, 4)
        ax1 = plt.subplot(gs[0, 0])
        onoffi = plt.imshow(onoff[:,:,det])
        plt.colorbar(onoffi,fraction=0.046, pad=0.04)
        
        e = Ellipse(xy=(self.p['x'][det],self.p['y'][det]),width=2.*np.log(2)*self.p['fwhmx'][det],height=2.*np.log(2)*self.p['fwhmy'][det],facecolor='None',edgecolor='r')
        ax1.add_artist(e)

        plt.title(map_type + ' On/Off Map')
        plt.xlabel('x')
        plt.ylabel('y')
        
        ax2 = plt.subplot(gs[0, 1])
        offi = plt.imshow(off[:,:,det])
        plt.colorbar(offi,fraction=0.046, pad=0.04)
        plt.title(map_type + ' Off Map')
        plt.xlabel('x')
        plt.ylabel('y')
        
        ax3 = plt.subplot(gs[1, 0])
        oni = plt.imshow(on[:,:,det])
        plt.colorbar(oni,fraction=0.046, pad=0.04)
        plt.title(map_type + ' On Map')
        plt.xlabel('x')
        plt.ylabel('y')
        
        ax4 = plt.subplot(gs[1, 1],aspect='equal',adjustable='box',autoscale_on=False)
        try:
            self.plot_psd(det,map_type=map_type,use_welch=False,scan_index='None',axes='None')
            plt.title('On Source Pixel PSD')
            plt.axis('equal')
        except:
            print('No raw files found for PSD')
        
        plt.tight_layout()
            
        if update == False:
            ax5 = plt.subplot(gs[:, 2:])
            self.nc_scatter(sx,sy,gi=gi,bi=bi,c=c,s=s,marker=marker,nf=False,interactive='off',use_ellipse=False,plot_nearest=plot_nearest,three_d=three_d)
            z = plt.scatter(self.p[sx][det],self.p[sy][det],marker='x',edgecolor='m',facecolor='None',s=50)
            plt.xlabel(sx)
            plt.ylabel(sy)
        
        print('Frequency (MHZ) ' + str(self.f[det]))
        if map_type == 'x':
            print('On/Off Map SNR ' + str(self.x_snr[det]))
        elif map_type == 'r':
            print('On/Off Map SNR ' + str(self.r_snr[det]))
        
        if update == False:
            return plt.gcf(),ax1,ax2,ax3,ax4,ax5,z
        else:
            return plt.gcf(),ax1,ax2,ax3,ax4

    def map_update(self,det,map_type,sx,sy,c,marker='x',gi=False,s='fwhmx',plot_nearest=False,bi=False,three_d=False):
        if gi == True:
            ind_to_plot = self.indices
        elif bi == False:
            ind_to_plot = list(range(self.ndets))
        else:
            ind_to_plot = self.bad_indices
  
        self.fig,self.ax1,self.ax2,self.ax3,self.ax4,self.ax5,self.z = self.plotmap(det=ind_to_plot[det],gi=gi,bi=bi,marker=marker,nf=True,map_type=map_type,sx=sx,sy=sy,s=s,c=c,plot_nearest=plot_nearest,three_d=three_d)
        
        def update_plotmaps(event):            
            if event.inaxes == self.ax5:
                clicked_dist = np.sqrt((event.xdata - self.p[sx][ind_to_plot])**2 + (event.ydata - self.p[sy][ind_to_plot])**2)
                i = np.where(clicked_dist == np.min(clicked_dist))[0][0]
                            
                self.z.remove()
                self.z = self.ax5.scatter(self.p[sx][ind_to_plot[i]],self.p[sy][ind_to_plot[i]],marker='x',color='m',s=50)
                if event.button == 1:
                        self.fig,self.ax1,self.ax2,self.ax3,self.ax4 = self.plotmap(det=ind_to_plot[i],sx=sx,sy=sy,gi=gi,bi=bi,marker=marker,nf=False,map_type=map_type,s=s,update=True,c=c,plot_nearest=plot_nearest,three_d=three_d)
                else:
                        self.plot_xr_maps(i,nf=True)
                self.fig.suptitle('Maps for detector ' + str(ind_to_plot[i]))

                plt.show()
        
        cid = self.fig.canvas.mpl_connect('button_press_event', update_plotmaps)

    def plot_xr_maps(self,det,normed=False,nf=True):
        si = np.array(self.ncfile['si'][:])
        ei = np.array(self.ncfile['ei'][:])
            
        if nf == True:                                
            fig = plt.figure(figsize=(12,8))
        gs = gridspec.GridSpec(2, 2)
        ax = plt.subplot(gs[1, 0])
        
        if normed == True:
            ax.plot(np.ravel(self.xmap[:,:,det])[1:] - np.ravel(self.xmap[:,:,det])[1],label='Normalized by first value')
            plt.legend(frameon=False,fontsize='small')
        else:
            ax.plot(np.ravel(self.xmap[:,:,det])[1:])

        plt.xlabel('Pixel')
        plt.ylabel('Xs')
        
        ax2 = plt.subplot(gs[0, 0])
        pc = plt.imshow(self.xmap[:,:,det])
        plt.colorbar(pc,label='Xs')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('X map')
        
        ax3 = plt.subplot(gs[0, 1])
        pc = plt.imshow(self.rmap[:,:,det])
        plt.colorbar(pc,label='Rs')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('R map')
        
        ax4 = plt.subplot(gs[1, 1])
        
        if normed == True:
            ax4.plot(np.ravel(self.rmap[:,:,det])[1:] - np.ravel(self.rmap[:,:,det])[1],label='Normalized by first value')
            plt.legend(frameon=False,fontsize='small')
        else:
            ax4.plot(np.ravel(self.rmap[:,:,det])[1:])

        plt.xlabel('Pixel')
        plt.ylabel('Rs')
        
        plt.tight_layout()
        
    def xplot(self,dets,nf=True,exclude=None,gi=True,ratio=False,lim=1):
        if nf == True:                                
            fig = plt.figure(figsize=(12,8)) 
        if gi == True:
            ind_to_plot = self.indices
        else:
            ind_to_plot = list(range(self.ndets))
        
        norm = plt.Normalize()
        colors = plt.cm.jet(norm(self.f[dets]))
        print(len(colors),len(dets))
        
        if ratio == False:
            gs = gridspec.GridSpec(1, 2)
            ax1 = plt.subplot(gs[0, 0])
        j=0
        for i in dets:
            if ratio == False:
                y = np.ravel(self.xmap[:,:,i])
            else:
                y = np.ravel(self.xmap[:,:,i])/np.ravel(self.rmap[:,:,i])
            if exclude !=None:
                plt.plot(y[y!=exclude])
            else:
                plt.plot(y,c=colors[j])
            j = j + 1
        plt.xlabel('pixel')
        if ratio == False:
            plt.ylabel('Xs')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        else:
            plt.ylabel('Xs/Rs')
            plt.ylim(-lim,lim)

        if ratio == False:
            ax2 = plt.subplot(gs[0, 1])
            j=0
            for i in dets:
                y = np.ravel(self.rmap[:,:,i])
                if exclude !=None:
                    plt.plot(y[y!=exclude])
                else:
                    plt.plot(y,c=colors[j])
                j = j + 1  
            plt.xlabel('pixel')
            plt.ylabel('Rs')
            plt.suptitle('Variation with pixel/time for nw ' + self.nw)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            #plt.tight_layout()
        
    def psd_pixel_plot(self,dets,nf=True,gi=False,bi = False):                
        if nf == True:                                
            fig = plt.figure(figsize=(12,8))
            
        if gi == True:
            ind_to_plot = self.indices
        elif bi == False:
            ind_to_plot = list(range(self.ndets))
        else:
            ind_to_plot = self.bad_indices
        
        gs = gridspec.GridSpec(1, 2)
        ax1 = plt.subplot(gs[0, 0])
        
        for i in dets:
            y = np.ravel(self.x_off[:,:,i])
            plt.semilogy(y)
        plt.xlabel('Pixel')
        plt.ylabel('Mean PSD value for Xs Off map')
            
        ax2 = plt.subplot(gs[0, 1])
        
        for i in dets:
            y = np.ravel(self.r_off[:,:,i])
            plt.semilogy(y)
        plt.xlabel('Pixel')
        plt.ylabel('Mean PSD value for Rs Off map')
        
    def xrvt(self,dets,chunksize=1,rng=None,start=0,ratio=False,vs_Rs=False,lim=1.0):
        plt.figure(figsize=(10,8))
        
        if ratio == False and vs_Rs == False:
            gs = gridspec.GridSpec(1, 2)
            ax1 = plt.subplot(gs[0, 0])
        
        nc = netCDF4.Dataset(self.raw_files[1])
        
        if rng == None:
            nt = len(nc.dimensions['time'])
        else:
            nt = rng
        nsteps = int(nt/chunksize)
        
        time = np.linspace(0,nt,nsteps)
        
        m = 0
        for i in dets:
            m = m + 1
            print('on det %i/%i For Xs' % (m,len(dets)))
            if ratio == False:
                if chunksize == 1:
                    Xs = nc['Xs'][start:start+rng,i]
                else:
                    Xs = nc['Xs'][::chunksize,i]
            if ratio == True and vs_Rs == False:
                if chunksize == 1:
                    Xs = nc['Xs'][start:start+rng,i]/nc['Rs'][start:start+rng,i]
                else:
                    Xs = nc['Xs'][::chunksize,i]/nc['Rs'][::chunksize,i]
            if ratio == False and vs_Rs == True:
                if chunksize == 1:
                    Rs = nc['Rs'][start:start+rng,i]
                else:
                    Rs = nc['Rs'][::chunksize,i]
                
            if vs_Rs == False:
                plt.plot(time,Xs)
            else:
                plt.plot(Rs,Xs,'.')
                plt.plot(Rs[0],Xs[0],'x',markersize=10,c='k')
        
        if ratio == False and vs_Rs == False:
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.ylabel('Xs')
            plt.xlabel('time index')
        if ratio == True and vs_Rs == False:
            plt.xlabel('time index')
            plt.ylabel('Xs/Rs')
            plt.ylim(-lim,lim)
        else:
            plt.xlabel('Rs')
            plt.ylabel('Xs')
        
        if ratio == False and vs_Rs == False:
            ax2 = plt.subplot(gs[0, 1])
            m = 0
            for i in dets:
                m = m + 1
                print('on det %i/%i for Rs' % (m,len(dets)))
                if chunksize == 1:
                    Rs = nc['Rs'][start:start+rng,i]
                else:
                    Xs = nc['Rs'][::chunksize,i]
                    
                plt.plot(time,Rs)
            
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.xlabel('time index')
            plt.ylabel('Rs')
        
            plt.suptitle('Range = %i | Chunk = %i' % (rng,chunksize))
    
    def ampvf(self,gi=False,bi=False,nf=True,plot_f=False,snr=True):
        if nf == True:                                
            fig = plt.figure(figsize=(12,8))
            
        if gi == True:
            ind_to_plot = self.indices
        elif bi == False:
            ind_to_plot = list(range(self.ndets))
        else:
            ind_to_plot = self.bad_indices
        
        xy = np.zeros(len(ind_to_plot))
        ry = np.zeros(len(ind_to_plot))

        if snr == True:
            plt.plot(self.x_snr[ind_to_plot],c='r',label='Xs')
            plt.plot(self.r_snr[ind_to_plot],c='k',label='Rs')
            plt.ylabel('Amps')
        else:
            for i in ind_to_plot:
                xy[i] = np.max(self.x_onoff[:,:,i])
                ry[i] = np.max(self.r_onoff[:,:,i])
            
            plt.plot(xy,c='r',label='Xs')
            plt.plot(ry,c='k',label='Rs')
            plt.ylabel('Max pixel value')
        
        plt.legend()
        plt.xlabel('Tone Index')
        
    def xrvf(self,dets,nf=True):
        if nf == True:
            plt.figure()
        stds = np.zeros(len(dets))
        j = 0
        for i in dets:
            y = np.ravel(self.xmap[:,:,i])
            stds[j] = np.std(y[1:] - y[0])
            j = j + 1
        Z = [x for _,x in sorted(zip(self.f[dets],stds))]
        
        plt.plot(np.sort(self.f[dets]),Z)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Mean standard deviation of Xs')

'''if __name__ == "__main__":
    
    nrows = 21
    ncols = 25
    sf = 488.281/4
    path = '/Users/mmccrackan/toltec/data/tests/wyatt/'
    
    obsnum = '10887_telecon2'
    ncobs = obs(obsnum,nrows,ncols,path,sf,order='C',transpose=False)
    f = {}
    for i in range(len(ncobs.ncs)):
        f[ncobs.nws[i]] = ncobs.ncs[i].f
    
    obsnum = 'coadd_telecon2'
    #obsnum = 10891
    ncobs = obs(obsnum,nrows,ncols,path,sf,order='C',transpose=False)
    
    for i in range(len(ncobs.nws)):
        ncobs.ncs[i].f = f[ncobs.nws[i]]
    
    cs = ['r','g','b','k','c','m','peru']
    marker = ['.','.','.','.','.','.','.','x','x','x','v','v']
    
    #cs = ['k','g']
    gi = True
    
    ncobs.obs_limit('amps', 45, '>',refresh=True)
    #ncobs.obs_limit('amps', 40, '<',refresh=False)
    #ncobs.obs_limit('y', 1.01, '<',refresh=False)

    
    gdets = 0
    ndets = 0
    for i in range(len(ncobs.ncs)):
        gdets = gdets + len(ncobs.ncs[i].indices)
        print(i,len(ncobs.ncs[i].indices))
        ndets = ndets + ncobs.ncs[i].ndets
        
    print('%i/%i' % (gdets,ndets))
    
    a1_1 = np.array(range(0,7))
    a1_4 = np.array(range(7,11))
    a2_0 = np.array(range(11,13))
    
    nws = np.zeros(len(ncobs.nws))

    for i in range(len(nws)):
        nws[i] = int(ncobs.nws[i])
    
    plt.figure()
    for i in range(len(a1_1)):
        try:
            j = np.where(nws == a1_1[i])
            k = j[0][0]
            print(k)
            ncobs.ncs[k].nc_scatter('x','y', c=cs[i],nf=False,plot_design=False,gi=gi)#,marker=marker[i])
        except:
            pass
    plt.title(obsnum + ' 1.1 mm')
    
    plt.figure()
    for i in range(len(a1_4)):
        try:
            j = np.where(nws == a1_4[i])
            k = j[0][0]
            print(k)
            ncobs.ncs[k].nc_scatter('x','y', c=cs[i],nf=False,plot_design=False,gi=gi)#,marker=marker[i])
        except:
            pass
    plt.title(obsnum + ' 1.4 mm')
    
    
    plt.figure()
    for i in range(len(a2_0)):
        try:
            j = np.where(nws == a2_0[i])
            k = j[0][0]
            print(k)
            ncobs.ncs[k].nc_scatter('x','y', c=cs[i],nf=False,plot_design=False,gi=gi)#,marker=marker[i])
        except:
            pass
    plt.title(obsnum + '2.0 mm')
    
    
    # nw=4
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    
    # ax.scatter(ncobs.ncs[nw].p['x'],ncobs.ncs[nw].p['y'],ncobs.ncs[nw].f)
    # #ax.scatter(ncobs.ncs[nw].)
    
    # nw = 0
    # ncobs.ncs[nw].limit('amps',40.0,'>',refresh=True)
    # ncobs.ncs[nw].scale_designed(gi=True)
    # ncobs.ncs[nw].map_update(0,'x','x','y',c='k',gi=True)
    
    
    ncobs.obs_limit('y',0,'!=')
    ncobs.obs_limit('y',np.max(ncobs.ncs[nw].p['y']),'!=',refresh=False)
    ncobs.obs_limit('x',0,'!=',refresh=False)
    ncobs.obs_limit('x',np.max(ncobs.ncs[nw].p['x']),'!=',refresh=False)
    ncobs.obs_limit('fwhmx',7.0,'<',refresh=False)
    
    if obsnum == 10568:
        xtmp = -np.array(ncobs.ncs[nw].p['x'])
        ytmp = np.array(ncobs.ncs[nw].p['y'])
    
        ncobs.ncs[nw].p['x'] = np.cos(np.pi/2)*xtmp + np.sin(np.pi/2.)*ytmp
        ncobs.ncs[nw].p['y'] = -np.sin(np.pi/2)*xtmp + np.cos(np.pi/2.)*ytmp
    
    ncobs.ncs[nw].limit('y',0,'!=')
    ncobs.ncs[nw].limit('y',np.max(ncobs.ncs[nw].p['y']),'!=',refresh=False)
    ncobs.ncs[nw].limit('x',0,'!=',refresh=False)
    ncobs.ncs[nw].limit('x',np.max(ncobs.ncs[nw].p['x']),'!=',refresh=False)
    ncobs.ncs[nw].limit('fwhmx',7.0,'<',refresh=False)
    ncobs.ncs[nw].limit('fwhmx',1.0,'>',refresh=False)
    ncobs.ncs[nw].limit('fwhmy',1.0,'>',refresh=False)
    
    ncobs.ncs[nw].map_update(0,map_type='r',sx='x',sy='y',gi=True,bi=False,s=10,c='m',marker='s',plot_nearest=True)
    #ncobs.ncs[nw].xplot((ncobs.ncs[nw].indices))
    #ncobs.ncs[nw].xrvt(range(ncobs.ncs[nw].ndets),rng=5000000,start=0,ratio=False,vs_Rs=True,lim=100,chunksize=1000)
    #ncobs.ncs[nw].ampvf(gi=False,bi=False,nf=True,plot_f=False,snr=False)
    #ncobs.ncs[nw].psd_pixel_plot(ncobs.ncs[nw].bad_indices)
    #ncobs.ncs[nw].xrvf((ncobs.ncs[nw].indices))
    
    
    from mpl_toolkits import mplot3d
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ax.scatter(ncobs.ncs[nw].p['x'],ncobs.ncs[nw].p['y'],ncobs.ncs[nw].f)
    
    
    #plt.figure()
    
    #for i in range(ncobs.ncs[nw].ndets):
     #   plt.scatter(np.std(np.ravel(ncobs.ncs[nw].xmap[:,:,i] - ncobs.ncs[nw].xmap[0,0,i])),ncobs.ncs[nw].f[i])
    
    
    maxes = np.zeros(ncobs.ncs[nw].ndets)
    maxes_good = np.zeros(len(ncobs.ncs[nw].indices))
    j = 0
    for i in range(ncobs.ncs[nw].ndets):
        maxes[i] = np.max(ncobs.ncs[nw].x_onoff[:,:,i])
        if i in ncobs.ncs[nw].indices:
            maxes_good[j] = np.max(ncobs.ncs[nw].x_onoff[:,:,i])
            j =  j + 1
      
    plt.figure()      
    plt.hist(maxes,100,label='all')
    plt.hist(maxes_good,100,label='good')
    plt.legend()
    plt.xlabel('Peak')
    plt.ylabel('N')
    
    
    
    
    obss = ["coadd", 10887, 10889, 10891, 10893, 10895, 10897]
    
    fig2 = plt.figure(constrained_layout=True,figsize=(10,10))
    gs = gridspec.GridSpec(4, 2,figure=fig2)
    nw = 0
    det = 425
    k=0
    fig2.suptitle('nw: %i det: %i' % (nw, det))     

    for i in range(4):
        for j in range(2):
            if (k) < len(obss):
                f = '/Users/mmccrackan/toltec/data/tests/wyatt/'  + str(obss[k]) + '/' + str(obss[k]) + '_toltec' + str(nw) + '.nc'
                nc = ncdata(f,str(obss[k]),nrows,ncols,str(nw),path,sf,order='C',transpose=False,dx=0,dy=0,df=0)
                ax = plt.subplot(gs[i, j])
                plt.imshow(nc.r_onoff[:,:,det])
                plt.title(obss[k])
                plt.colorbar()
                k = k + 1

     
    
    
    plt.show()
    '''
