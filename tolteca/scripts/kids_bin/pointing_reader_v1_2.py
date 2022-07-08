#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sun Jan  2 12:31:23 2022

@author: mmccrackan
'''

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D

import glob
import argparse
import sys
import json

if __name__ == '__main__':
    
    # param names in pointing ecsv table
    params = ['amp','x_t','y_t','a_fwhm','b_fwhm','angle']
    params_units = ['MJy/Sr','arcsec','arcsec','arcsec','arcsec','rad']
    
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-p', 
                        help='path to input directory')
    parser.add_argument('--save_to_file', '-s', help='save output to file', 
                        action='store_true')
    parser.add_argument('--output_path', '-o', help='path to output directory')
    parser.add_argument('--images', '-i', nargs='+',
                        help='images to plot', default=['signal_I'])
    parser.add_argument('--obsnum', '-n', help='find by obsnum in input dir', 
                        default='none')
    parser.add_argument('--center', '-c', help='center (crpix, fit, ' 
                        'peak of S/N map)', 
                        default='crpix')
    
    args = parser.parse_args()
    print(args)
    # get path to input directory
    input_path = str(args.input_path)
        
    # get pointing ecsv file
    try:
        if args.obsnum == 'none':
            ecsv_file = glob.glob(input_path+'/ppt*pointing*.ecsv')[0]
        else:
            ecsv_file = glob.glob(input_path+'/ppt*pointing*'+args.obsnum.zfill(6)+'*.ecsv')[0]
        
        table = Table.read(ecsv_file)
    except:
        print('no pointing ecsv file found')
        sys.exit()
    
    # get list of pointing FITS files
    if args.obsnum == 'none':
        fits_files = np.sort(glob.glob(input_path+'/toltec*pointing*.fits'))
    else:
        fits_files = np.sort(glob.glob(input_path+'/toltec*pointing*'+args.obsnum.zfill(6)+'.fits'))

    # check if there are any FITS files
    if fits_files.size == 0:
        print('no pointing FITS files found')
        sys.exit()
        
    # dictionary for pointing parameters, errors, and units
    ppt_dict = {
        'amp': {},
        'x_t': {},
        'y_t': {},
        'a_fwhm': {},
        'b_fwhm': {},
        'angle': {}}
        
    # loop through input FITS files
    for i in range(len(fits_files)):
        # get array name from FITS file
        if 'a1100' in fits_files[i]:
            array = 'a1100'
        elif 'a1400' in fits_files[i]:
            array = 'a1400'
        elif 'a2000' in fits_files[i]:
            array = 'a2000'
        else:
            print('cannot find TolTEC array name from FITS files')
            sys.exit()
        
        # populate parameter dictionary
        for j in range(len(params)):
            key = params[j]
            unit = params_units[j]
            ppt_dict[key] = {
                'value': str(table[key][i]),
                'error': str(table[key + '_err'][i]),
                'units': unit
                }
         
        print('\n')
        print(array)
        print(ppt_dict)
        print('\n')
        
        # choose number of rows for image
        nrows = len(args.images)
        
        # get current FITS file
        img = fits.open(fits_files[i])
        # make WCS object
        wcs = WCS(img[1].header).sub(2)
        
        pix_scale_arcsec = abs(wcs.wcs.cdelt[0])
        crpix1, crpix2 = wcs.wcs.crpix
        zoom_size_arcsec = 30.0
        zoom_size_pix = np.floor(zoom_size_arcsec/pix_scale_arcsec)
        
        # image plotting
        cutout = Cutout2D(img[1].data[0,0,:,:], (crpix1, crpix2), 
                          (zoom_size_pix, zoom_size_pix), wcs=wcs)
        
        # create figure with wcs projection
        fig, ax = plt.subplots(nrows=nrows, ncols=1, 
                               figsize=(5*nrows, 5*nrows), 
                               subplot_kw={'projection': cutout.wcs})
        
        # get obsnum for title
        if args.obsnum == 'none':
            obsnum = str(img[0].header['OBSNUM']).zfill(6)
        else:
            obsnum = args.obsnum.zfill(6)
            
        try:
            source_name = img[0].header['OBJECT']
        except:
            source_name = 'N/A'
            
        for ci in range(len(args.images)):
            if len(args.images) == 1:
                axi = ax
            else:
                axi = ax[ci]
                
            # image plotting
            if args.center == 'crpix':
                cutout = Cutout2D(img[img.index_of(args.images[ci])].data[0,0,:,:], 
                                   (crpix1, crpix2), (zoom_size_pix, zoom_size_pix),
                                   wcs=wcs)
            elif args.center == 'fit':
                # get pixel coordinates of fitted positions
                cx,cy = wcs.all_world2pix(float(ppt_dict["x_t"]["value"]), 
                                         float(ppt_dict["y_t"]["value"]),0)
                cutout = Cutout2D(img[img.index_of(args.images[ci])].data[0,0,:,:], 
                                  (cx, cy),(zoom_size_pix, zoom_size_pix), wcs=wcs)
            elif args.center == 'peak':
                # get position of peak
                y,x = np.where(img[img.index_of('sig2noise_I')].data[0,0,:,:] == np.max(img[img.index_of('sig2noise_I')].data[0,0,:,:]))
                cutout = Cutout2D(img[img.index_of(args.images[ci])].data[0,0,:,:], (x, y), 
                                  (zoom_size_pix, zoom_size_pix), wcs=wcs)
            
            # get pixel coordinates of fitted positions
            cx,cy = cutout.wcs.all_world2pix(float(ppt_dict["x_t"]["value"]), 
                                     float(ppt_dict["y_t"]["value"]),0)
            
            # get pixel coordinates of fitted positions
            x,y = wcs.all_world2pix(float(ppt_dict["x_t"]["value"]), 
                                     float(ppt_dict["y_t"]["value"]),0)
            
            #im = axi.imshow(img[img.index_of(args.images[ci])].data)
            im = axi.imshow(cutout.data)
            
            '''p1 = axi.get_position()
            ax_in = fig.add_axes([0, 0, .3*p1.width, .3*p1.height], projection=wcs)
            p2 = ax_in.get_position()
            ax_in.set_position([p1.x1-p2.width, p1.y1-p2.height, p2.width, 
                                p2.height])
            '''
            
            ax_in = inset_axes(axi,
                    width="30%",
                    height="30%")
            
            ax_in.spines['bottom'].set_color('red')
            ax_in.spines['top'].set_color('red') 
            ax_in.spines['right'].set_color('red')
            ax_in.spines['left'].set_color('red')  
            
            im2 = ax_in.imshow(img[img.index_of(args.images[ci])].data[0,0,:,:],
                               origin='lower')
            
            im2 = ax_in.imshow(img[img.index_of('weight_I')].data[0,0,:,:],
                               origin='lower',alpha=0.15)
            
            ax_in.axes.get_xaxis().set_visible(False)
            ax_in.axes.get_yaxis().set_visible(False)

            axi.grid(color='black', ls='--', alpha=0.25)
            axi.set_title('%s -- %s -- %s -- %s' % (source_name, obsnum, 
                                                    array,args.images[ci]),
                          fontsize=10)
            axi.axvline(cx,c='w',linestyle='--')
            axi.axhline(cy,c='w',linestyle='--')
            
            ax_in.axvline(x,c='w',linestyle='--', linewidth=1, alpha=0.25)
            ax_in.axhline(y,c='w',linestyle='--', linewidth=1, alpha=0.25)
            
            lon = axi.coords[0]
            lat = axi.coords[1]
            
            lon.set_axislabel('Azimuth (arcsec)')
            lat.set_axislabel('Elevation (arcsec)')
                        
            # add parameters to image
            textstr = ''
            for j in range(len(params)):
                key = params[j]
                units = params_units[j]
                if key=='amp':
                    textstr = textstr + (key + '=%3.2f +/- %2.1f %s' % (table[key][i], 
                                                          table[key+'_err'][i], 
                                                          units))
                else:
                    textstr = textstr + (key + '=%3.1f +/- %3.1f %s' % (table[key][i], 
                                                          table[key+'_err'][i], 
                                                          units))
                if j != len(params)-1:
                    textstr = textstr + '\n'
            
            # place a text box in upper left in axes coords
            props = dict(boxstyle='round', facecolor='gray', alpha=0.75)
            axi.text(0.25, 0.25, textstr, transform=axi.transAxes, fontsize=8,
                verticalalignment='top', bbox=props, c='yellow')
            
            # setup colorbar
            divider = make_axes_locatable(axi)
            cax = divider.append_axes('right', size='5%', pad=0.1, 
                                      axes_class=maxes.Axes)
            
            # add colorbar
            fig.colorbar(im, cax=cax, label=img[img.index_of(args.images[ci])].header['UNIT'], 
                         orientation='vertical')
        
        # check if saving is requested
        if args.save_to_file:
            if args.output_path is not None:
                fig.savefig(args.output_path + '/toltec_' + array + '_pointing_' + obsnum + '_image.png', 
                            bbox_inches='tight')
        
                with open(args.output_path + '/toltec_' + array + '_pointing_' + obsnum + '_params.txt', 'w') as convert_file:
                    convert_file.write(json.dumps(ppt_dict))
            else:
                print('no save file output path specified')
                sys.exit()

