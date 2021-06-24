from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

import numpy as np
from datetime import datetime

class toltec_beams():
    """
    This class creates astropy kernels for the convolution of the images with
    the TolTEC beams.  Inherited by sim_fits.
    """
    def __init__(self):
        self.toltec_fwhm = np.array([5.0, 6.3, 9.5]) # arcsec
        self.toltec_stddev = self.toltec_fwhm/(2.0*np.sqrt(2.*np.log(2)))
        
        self.kernels = []
        
        self.kernels.append(Gaussian2DKernel(x_stddev=self.toltec_stddev[0],
                                            y_stddev=self.toltec_stddev[0]))
        self.kernels.append(Gaussian2DKernel(x_stddev=self.toltec_stddev[1],
                                            y_stddev=self.toltec_stddev[1]))
        self.kernels.append(Gaussian2DKernel(x_stddev=self.toltec_stddev[2],
                                             y_stddev=self.toltec_stddev[2]))


class sim_fits(toltec_beams):
    def __init__(self):
        """
        This class creates an hdulist for the TolTEC Simulation pipeline from
        a set of input images and a WCS object, thereby allowing users to run
        their own images through the TolTEC simulator.
        Returns
        -------
        None.
        """
        self.arr_names = ['a1100', 'a1400', 'a2000']
        self.w = [1100.0, 1400.0, 2000.0] # micrometers
        self.pol_names = ['I', 'Q', 'U']
 #       self.required_CDELT = 1./3600 # deg
        self.required_CDELT = 1.*u.arcsec
        self.required_units = 'MJy/sr'
        self.optional_keys = ['RA', 'DEC', 'POSANGLE', 'OBSERVER', 'OBJECT']

    def fits_validator(self, imgs, wcs):
        """
        Checks if input images are in a list and if there are 9 images.
        Also checks if pixel scale is 1 arcsecond.
        """
        
        valid = True
        print('Validating image list\n')
        if isinstance(imgs, list):
            if len(imgs) == 9:
                print('Found 9 images...proceeding...\n')
                valid= True
            else:
                print('Incorrect number of images in list (9 required)\n')
                valid = False
        else:
            print('Images should be given in a list\n')
            valid = False
        

  #     if ((abs(wcs.wcs.cdelt[0]) - self.required_CDELT) > 1e-6) or ((abs(wcs.wcs.cdelt[1]) - self.required_CDELT) > 1e-6):
        if (abs(wcs.pixel_to_world(0,0).separation(wcs.pixel_to_world(0,1)) 
                -self.required_CDELT) > 1e-3*u.arcsec):
            print('Pixel scale is incorrect. Please use 1 arcsec pixels.\n')
            valid = False

        return valid
    
    def generate_fits(self, imgs, wcs, convolve_img=True, **kwargs):
        '''
        This function will generate a PrimaryHDU with a header containing the
        input WCS information and other general meta data.
        The HDU list contains 9 ImageHDU extensions for each array and
        polarization.  It can be accessed from the method self.hdul.
        Users can add comments or additional fields to the headers though
        they will not be used in the simulation software.
        BANDPASSES SHOULD BE APPLIED PRIOR TO GENERATING THE FITS FILE.
        
        INPUT IMAGE ARRAYS SHOULD BE 10 ARCSECONDS LARGER ON EACH SIDE THAN
        THE FINAL INTENDED MAP TO BE OUTPUT FROM CITLALI DUE TO BEAM 
        CONVOLUTION.
        Parameters
        ----------
        imgs : list of 2D arrays
            A list containing images for each TolTEC array and each
            polarization.  They should be ordered as follows:
                [1100_I, 1100_Q, 1100_U, 1400_I, 1400_Q, 1400_U, 2000_I,
                 2000_Q, 2000_U].  If you do not wish to include a map, set 
                it to None and that layer will be created but the data will 
                be left empty
                Units must be in MJy/sr
        wcs : Astropy WCS object
            Standard Astropy WCS object for coordinates.
            Pixel scale must be 1 arcsecond.
            Coordinates must be specified in the ICRS system.
        convolve_img: Set to true to create second hdulist 
                    (self.convolved_hdul) for images convolved with TolTEC
                    beams.  Unconvolved hdu list is stored under self.raw_hdul
        **kwargs : Add an optional header keyword and value.
        Returns
        -------
        None.
        '''
        
        print('Creating unconvolved Images (self.raw_hdul)')
        self.raw_hdul = self.setup_hdul(imgs, wcs, **kwargs)
        
        # Check if convolved image is requested
        if convolve_img == True:
            print('Creating Convolved Images (self.convolved_hdul)')
            convolved_imgs = self.convolve_with_beam(imgs)
            self.convolved_hdul = self.setup_hdul(convolved_imgs, wcs,
                                                 **kwargs)
        

    def setup_hdul(self, imgs, wcs, **kwargs):

        check = self.fits_validator(imgs, wcs)

        # Only generate the hudlist if it passes validation
        if check is True:
            
            header = wcs.to_header()

            self.hdul = fits.HDUList()

            self.hdul.append(fits.PrimaryHDU(header=header))

            self.hdul[0].header.append(('TIMESYS', 'UTC',
                                        'All dates are in UTC time'))
            self.hdul[0].header.append(('SIGUNIT', self.required_units,
                                        'Map units'))
            self.hdul[0].header.append(('OBSERVER', 'N/A', 'Observer name'))
            self.hdul[0].header.append(('OBJECT', 'N/A', 'Target name'))
            self.hdul[0].header.append(('DATE', str(datetime.utcnow()),
                                        'Creation date of this file'))
            self.hdul[0].header.append(('RA', 'N/A',
                                        'Actual Right Ascension of pointing'))
            self.hdul[0].header.append(('DEC', 'N/A',
                                        'Actual Declination of pointing'))
            self.hdul[0].header.append(('POSANGLE', 'N/A',
                                        'Position Angle of pointing'))

            for key, value in kwargs.items():
                if key in self.optional_keys:
                    self.hdul[0].header[key] = value
                else:
                    self.hdul[0].header.append((key, value))

            layer = 1
            wi = 0
            for arr in self.arr_names:
                pi = 0
                for pol in self.pol_names:
                    name = arr + '_' + pol
                    self.hdul.append(fits.ImageHDU(header=header, name=name))
                    self.hdul[layer].header.append(('WAVELNTH', self.w[wi],
                                                    '[micrometer] wavelength'))
                    self.hdul[layer].header.append(('STOKES',
                                                    self.pol_names[pi],
                                                    'Stokes Parameter'))
                    self.hdul[layer].header.append(('SIGUNIT',
                                                    self.required_units,
                                                    'Map units'))

                    if isinstance(imgs[layer-1], np.ndarray):
                        if len(imgs[layer-1].shape) == 2:
                            print('Adding image for ' + name)
                            self.hdul[layer].data = imgs[layer-1]
                    elif imgs[layer-1] == None:
                        print('No image for ' + name + '...skipping...')
                        self.hdul[layer].data = imgs[layer-1]

                    layer += 1
                    pi += 1
                wi += 1
        return self.hdul
    
    def convolve_with_beam(self, imgs):
        #T = toltec_beams()
        convolved_imgs = []
        w = 0
        j = 0
        for i in range(len(imgs)):
            if imgs[i] is not None:
                convolved_imgs.append(convolve(imgs[i], toltec_beams().kernels[j]))
            if w == 2 :
                j = j + 1
                w = 0
            else:
                w = w +1
        return convolved_imgs

if __name__ == "__main__":

    nlayers = 9

    CDELT = 1./3600  # deg
    NAXIS1 = 1024  # randomly chosen
    NAXIS2 = 1024

    img = np.random.normal(0, 0.1, [NAXIS1, NAXIS2, nlayers])
    imgs = []
    for i in range(nlayers):
        imgs.append(img[:, :, i])
        
    imgs[0] = np.zeros([NAXIS1, NAXIS2])

    wcs_input_dict = {
        'CTYPE1': 'RA---TAN',
        'CUNIT1': 'deg',
        'CDELT1': -CDELT,
        'CRPIX1': 1,
        'CRVAL1': 337.5202808,
        'NAXIS1': 1024,
        'CTYPE2': 'DEC--TAN',
        'CUNIT2': 'deg',
        'CDELT2': CDELT,
        'CRPIX2': 1,
        'CRVAL2': -20.833333059999998,
        'NAXIS2': 1024
    }
    wcs_dict = WCS(wcs_input_dict)
    #wcs_dict.wcs.cd = np.array([[-CDELT, 0],[0, CDELT]])
    header = wcs_dict.to_header(relax=True)

    sf = sim_fits()
    sf.generate_fits(imgs=imgs, wcs=wcs_dict, convolve_img=True)
    
    # sf.raw_hdul.writeto('/Users/mmccrackan/toltec/temp/simu_input_example',
                    # output_verify='exception', overwrite=True, checksum=False)

    # sf.convolved_hdul.writeto('/Users/mmccrackan/toltec/temp/simu_input_example',
                    # output_verify='exception', overwrite=True, checksum=False)
