#!/usr/bin/env python
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from astropy import units as u
from astropy import constants
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord, SkyOffsetFrame, EarthLocation
from astropy.stats import median_absolute_deviation
from astropy.stats import biweight

#from scipy import interpolate
#from scipy import optimize
#from scipy.optimize import least_squares
#from scipy.optimize import minimize
#from scipy.interpolate import UnivariateSpline
#import scipy.integrate as integrate

_cache = {}

def cached_fits_open(f):
    global _cache
    try:
        x = _cache[f]
    except KeyError:
        x = _cache[f] = fits.open(f)
    import copy
    return copy.deepcopy(x)

class IFUM_UNIT:
    def __init__(self, label):
        if label=='LSB':
            self.Nx, self.Ny = 18, 20
            self.width_y = 17.3
        elif label=='STD':
            self.Nx, self.Ny = 23, 24
            self.width_y = 10.0
        elif label=='HR':
            self.Nx, self.Ny = 27, 32
            self.width_y =  5.0
        elif label=='M2FS':
            self.Nx, self.Ny = 16, 16
            self.width_y =  5.0
        else:
            self.Nx, self.Ny = 0, 0
            self.width_y =  0.
        
        self.label = label
        self.Ntotal = self.Nx*self.Ny
            


def func_parabola(x, a, b, c):
    return a*(x-b)**2+c

def readString_symbol(fileName, column1, symbol): #column number starts from 0
    x0 = []
    for line in open(fileName, 'r'):
        #skip over empty lines or lines starting with spaces or spaces+#
        tempString=line.strip()
        if (tempString[0] == '#' or tempString[0]==''):
            continue

        line = line.split(symbol)
        if line[column1]=='-':
            line[column1]=float('nan')

        x = line[column1].strip()
        x0.append(x)
    return np.array(x0)

def readFloat_space(fileName, column1): #column number starts from 0
    x0 = []
    for line in open(fileName, 'r'):
        #skip over empty lines or lines starting with spaces or spaces+#
        tempString=line.strip()
        if (tempString[0] == '#' or tempString[0]==' '):
            continue

        line = line.split()
        if line[column1]=='-':
            line[column1]=float('nan')

        x = np.float64(line[column1])
        x0.append(x)
    return np.array(x0, dtype='float64')

def mask_img(raw_data,mask_file):
    new_data = np.zeros_like(raw_data)
    for line in open(mask_file,'r'):
        tempString = line.strip()
        if (tempString[0]=='#'):
            continue

        line = [int(item) for item in line.split()]
        if (len(line)==4):
            new_data[line[2]-1:line[3],line[0]-1:line[1]] = raw_data[line[2]-1:line[3],line[0]-1:line[1]]
        else:
            print("Warning: img_mask file has an error!!!")

    return new_data

def pack_4fits_simple(name_file, dir_input, shoe): #,dir_output,flag_img_mask,path_img_mask,config_img_mask):
    data_full = np.array([])
    enoise_full = np.array([])
    flag_egain = False
    for i_temp in range(4):
        name_file_temp = dir_input+'/'+shoe+name_file+'c%d.fits'%(i_temp+1)
        hdul_temp = cached_fits_open(name_file_temp)
        hdr_temp  = hdul_temp[0].header
        data_temp = np.float32(hdul_temp[0].data)

        if i_temp == 0:
            hdr_c1    = hdr_temp
            if ('EGAIN' in hdr_temp):
                egain_c1  = hdr_temp['EGAIN']
                flag_egain = True

        if (flag_egain):
            egain_temp = np.float32(hdr_temp['EGAIN'])
            enoise_temp = np.float32(hdr_temp['ENOISE'])
            enoise_full = np.append(enoise_full, enoise_temp*egain_temp/egain_c1)
            data_temp = data_temp*egain_temp/egain_c1

        if ('DATASEC' in hdr_temp):
            datasec_temp = hdr_temp['DATASEC'].split(',')
            X1 = int( datasec_temp[0].split(':')[0].split('[')[1] )
            X2 = int( datasec_temp[0].split(':')[1] )
            Y1 = int( datasec_temp[1].split(':')[0] )
            Y2 = int( datasec_temp[1].split(':')[1].split(']')[0] )
        elif ('TRIMSEC' in hdr_temp):
            datasec_temp = hdr_temp['TRIMSEC'].split(',')
            X1 = int( datasec_temp[0].split(':')[0].split('[')[1] )
            X2 = int( datasec_temp[0].split(':')[1] )
            Y1 = int( datasec_temp[1].split(':')[0] )
            Y2 = int( datasec_temp[1].split(':')[1].split(']')[0] )
        elif ('CCDSEC' in hdr_temp):
            datasec_temp = hdr_temp['CCDSEC'].split(',')
            X1 = int( datasec_temp[0].split(':')[0].split('[')[1] )
            X2 = int( datasec_temp[0].split(':')[1] )
            Y1 = int( datasec_temp[1].split(':')[0] )
            Y2 = int( datasec_temp[1].split(':')[1].split(']')[0] )
        else:
            X1 = 1
            X2 = hdr_temp['NAXIS1']
            Y1 = 1
            Y2 = hdr_temp['NAXIS2']

        if ('BIASSEC' in hdr_temp):
            biassec_temp = hdr_temp['BIASSEC'].split(',')
            X1_overscan = int( biassec_temp[0].split(':')[0].split('[')[1] )
            X2_overscan = int( biassec_temp[0].split(':')[1] )
            Y1_overscan = int( biassec_temp[1].split(':')[0] )
            Y2_overscan = int( biassec_temp[1].split(':')[1].split(']')[0] )

            bias_temp = data_temp[:,X1_overscan-1:X2_overscan]
            bias_mean_temp = np.mean(bias_temp, axis=1)
            #print(data_temp[0])

            for j_temp in range(len(data_temp)):
                data_temp[j_temp,:] -= bias_mean_temp[j_temp]

        #data_temp = np.int32(data_temp)

        data_temp_sub = data_temp[Y1-1:Y2, X1-1:X2]
        #print(data_temp_sub[0])

        if i_temp==0:   # flip vertically (axis=0)
            data_temp_sub = np.flip(data_temp_sub, axis=0)
            data_half1 = data_temp_sub
        elif i_temp==1: # flip both
            data_temp_sub = np.flip(data_temp_sub)
            data_half1 = np.append(data_half1, data_temp_sub, axis=1)
        elif i_temp==2: # flip horizontally (axis=1)
            data_temp_sub = np.flip(data_temp_sub, axis=1)
            data_half2 = data_temp_sub
        elif i_temp==3: # no flip
            data_half2 = np.append(data_temp_sub, data_half2, axis=1)

        #figure()
        #plt.imshow(data_temp_sub, origin='lower')

    # combine order:
    # 4 3
    # 1 2
    data_full = np.append(data_half2, data_half1, axis=0)
    return data_full, hdr_c1

def pack_4fits(name_file,dir_input,dir_output,flag_img_mask,path_img_mask,config_img_mask):
    data_full = np.array([])
    enoise_full = np.array([])
    flag_egain = False
    for i_temp in range(4):
        name_file_temp = dir_input+'/'+name_file+'c%d.fits'%(i_temp+1)
        hdul_temp = cached_fits_open(name_file_temp)
        hdr_temp  = hdul_temp[0].header
        data_temp = np.float32(hdul_temp[0].data)

        if i_temp == 0:
            hdr_c1    = hdr_temp
            if ('EGAIN' in hdr_temp):
                egain_c1  = hdr_temp['EGAIN']
                flag_egain = True

        if (flag_egain):
            egain_temp = np.float32(hdr_temp['EGAIN'])
            enoise_temp = np.float32(hdr_temp['ENOISE'])
            enoise_full = np.append(enoise_full, enoise_temp*egain_temp/egain_c1)
            data_temp = data_temp*egain_temp/egain_c1

        if ('DATASEC' in hdr_temp):
            datasec_temp = hdr_temp['DATASEC'].split(',')
            X1 = int( datasec_temp[0].split(':')[0].split('[')[1] )
            X2 = int( datasec_temp[0].split(':')[1] )
            Y1 = int( datasec_temp[1].split(':')[0] )
            Y2 = int( datasec_temp[1].split(':')[1].split(']')[0] )
        elif ('TRIMSEC' in hdr_temp):
            datasec_temp = hdr_temp['TRIMSEC'].split(',')
            X1 = int( datasec_temp[0].split(':')[0].split('[')[1] )
            X2 = int( datasec_temp[0].split(':')[1] )
            Y1 = int( datasec_temp[1].split(':')[0] )
            Y2 = int( datasec_temp[1].split(':')[1].split(']')[0] )
        elif ('CCDSEC' in hdr_temp):
            datasec_temp = hdr_temp['CCDSEC'].split(',')
            X1 = int( datasec_temp[0].split(':')[0].split('[')[1] )
            X2 = int( datasec_temp[0].split(':')[1] )
            Y1 = int( datasec_temp[1].split(':')[0] )
            Y2 = int( datasec_temp[1].split(':')[1].split(']')[0] )
        else:
            X1 = 1
            X2 = hdr_temp['NAXIS1']
            Y1 = 1
            Y2 = hdr_temp['NAXIS2']

        if ('BIASSEC' in hdr_temp):
            biassec_temp = hdr_temp['BIASSEC'].split(',')
            X1_overscan = int( biassec_temp[0].split(':')[0].split('[')[1] )
            X2_overscan = int( biassec_temp[0].split(':')[1] )
            Y1_overscan = int( biassec_temp[1].split(':')[0] )
            Y2_overscan = int( biassec_temp[1].split(':')[1].split(']')[0] )

            bias_temp = data_temp[:,X1_overscan-1:X2_overscan]
            bias_mean_temp = np.mean(bias_temp, axis=1)
            #print(data_temp[0])

            for j_temp in range(len(data_temp)):
                data_temp[j_temp,:] -= bias_mean_temp[j_temp]

        #data_temp = np.int32(data_temp)

        data_temp_sub = data_temp[Y1-1:Y2, X1-1:X2]
        #print(data_temp_sub[0])

        if i_temp==0:   # flip vertically (axis=0)
            data_temp_sub = np.flip(data_temp_sub, axis=0)
            data_half1 = data_temp_sub
        elif i_temp==1: # flip both
            data_temp_sub = np.flip(data_temp_sub)
            data_half1 = np.append(data_half1, data_temp_sub, axis=1)
        elif i_temp==2: # flip horizontally (axis=1)
            data_temp_sub = np.flip(data_temp_sub, axis=1)
            data_half2 = data_temp_sub
        elif i_temp==3: # no flip
            data_half2 = np.append(data_temp_sub, data_half2, axis=1)

        #figure()
        #plt.imshow(data_temp_sub, origin='lower')

    # combine order:
    # 4 3
    # 1 2
    data_full = np.append(data_half2, data_half1, axis=0)

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.imshow(data_full, origin='lower')
    #fig.set_tight_layout(True)
    #fig.savefig('./fig/b%s_fits_2D_img_raw_210427.pdf'%(num_file), format='pdf', transparent=True)


    #### mask images
    if (flag_img_mask):
        file_img_mask = path_img_mask+'/img_mask_'+name_file[0]+'_'+config_img_mask
        data_full = mask_img(data_full, file_img_mask)

    #### record packed fits file
    hdu_full = fits.PrimaryHDU(data_full,header=hdr_c1)
    hdul_full = fits.HDUList([hdu_full])
    hdr_full  = hdul_full[0].header
    if ('BIASSEC' in hdr_full):
        del hdr_full['BIASSEC']
        del hdr_full['DATASEC']
        del hdr_full['TRIMSEC']
    if ('NOVERSCN' in hdr_full):
        del hdr_full['NOVERSCN']
    if ('NBIASLNS' in hdr_full):
        del hdr_full['NBIASLNS']
    if (flag_egain):
        hdr_full['ENOISE'] = np.mean(enoise_full)
    hdr_full['NOPAMPS'] = 1
    hdr_full['OPAMP'] = 1
    hdr_full['FILENAME'] = (name_file,'')
    #hdr_full['NOVERSCN'] = (0,'overscan pixels')
    #hdr_full['NBIASLNS'] = (0,'bias lines')
    #hdr_full['BIASSEC'] = ('[%d:%d,%d:%d]'%(naxis1/2+1,naxis1_sub,naxis2/2+1,naxis2_sub),'NOAO: bias section')
    hdr_full['DATASEC'] = ('[1:%d,1:%d]'%(X2*2,Y2*2),'NOAO: data section')
    #hdr_full['TRIMSEC'] = ('[1:%d,1:%d]'%(X2*2,Y2*2),'NOAO: trim section')
    hdr_full['CCDSEC'] = '[1:%d,1:%d]'%(X2*2,Y2*2)
    ## if name_file[1:]=='bias':
    ##     hdr_full['OBJECT'] = 'Bias'
    ##     hdr_full['SLIDE']  = 'HiRes'
    ##     hdr_full['FILTER'] = 'Mgb_Rev2'
    ## if name_file[1:]=='dark':
    ##     hdr_full['OBJECT'] = 'Dark'
    ##     hdr_full['SLIDE']  = 'HiRes'
    ##     hdr_full['FILTER'] = 'Mgb_Rev2'
    ## if (hdr_full['FF-QRTZ']>0):
    ##     hdr_full['EXPTYPE'] = 'Lamp-Quartz'
    ## elif (hdr_full['FF-THNE']>0 and hdr_full['FF-THNE']>0):
    ##     hdr_full['EXPTYPE'] = 'Lamp-ThArNe'
    ## elif (hdr_full['OBJECT']=='Twilight Config 1'):
    ##     hdr_full['EXPTYPE'] = 'Twilight'
    hdr_full['BINNING'] = ('1x1', 'binning') # added by YYS on May 11, 2022

    hdul_full.writeto(dir_output+'/'+name_file+'.fits',overwrite=True)


def write_aperMap(path_MasterSlits, IFU_type, Channel, file_name, file_date, N_xx, N_yy, img_mask_flag,img_mask_path,img_mask_config, add_badfiber_flag, add_badfiber_spat_id):
    N_ap = int(N_xx*N_yy/2)

    hdul = cached_fits_open(path_MasterSlits)
    hdr = hdul[1].header
    data = hdul[1].data

    N_sl  = int(hdr['NSLITS'])
    nspec = int(hdr['NSPEC']) ### binning?
    nspat = int(hdr['NSPAT'])
    map_ap = np.zeros((nspat,nspec), dtype=np.int32)
    print('nspat, nspec=',nspat,nspec)
    print('Note: %d out of %d fibers are found by pypeit_trace_edges.'%(N_sl,N_ap))

    if (add_badfiber_flag):
        print('Note: %d fiber(s) are added manually.'%(len(add_badfiber_spat_id)))
        spat_id_raw = data['spat_id']
        spat_id_new = np.append(spat_id_raw, add_badfiber_spat_id)
        spat_id_new = np.sort(spat_id_new)
        N_new = len(spat_id_new)

        if N_new>N_ap:
            print('!!! Warning: More bad fibers are added. !!!')
        elif N_new<N_ap:
            print('!!! Warning: Less bad fibers are added. !!!')

        for i_ap in range(N_new):
            ap_num = i_ap+1
            spat_id_temp = spat_id_new[i_ap]

            temp_index = np.where(spat_id_raw==spat_id_temp)[0]
            if len(temp_index)==1:
                i_temp = temp_index[0]
                for x_temp in range(nspec):
                    ap_y1 = int(np.round(data[i_temp]['left_init'][x_temp]-1))
                    ap_y2 = int(np.round(data[i_temp]['right_init'][x_temp]))
                    map_ap[ap_y1:ap_y2, x_temp] = np.int32(ap_num)
            else:
                #map_ap[spat_id_temp-2:spat_id_temp+1, int(nspec/2)-2:int(nspec/2)+1] = np.int32(ap_num)
                #print("!!!!!!", spat_id_temp-2, spat_id_temp+1, int(nspec/2)-2, int(nspec/2)+1)

                d1_spat_id = spat_id_temp - spat_id_new[i_ap-1]
                d2_spat_id = spat_id_new[i_ap+1] - spat_id_temp
                print(d1_spat_id, d2_spat_id)
                if d1_spat_id>d2_spat_id:
                    temp_index = np.where(spat_id_raw==spat_id_new[i_ap+1])[0]
                    if len(temp_index)==1:
                        i_temp = temp_index[0]
                        for x_temp in range(nspec):
                            ap_y1 = int(np.round(data[i_temp]['left_init'][x_temp]-1))
                            ap_y2 = int(np.round(data[i_temp]['right_init'][x_temp]))
                            map_ap[ap_y1-d2_spat_id:ap_y2-d2_spat_id, x_temp] = np.int32(ap_num)
                else:
                    temp_index = np.where(spat_id_raw==spat_id_new[i_ap-1])[0]
                    if len(temp_index)==1:
                        i_temp = temp_index[0]
                        for x_temp in range(nspec):
                            ap_y1 = int(np.round(data[i_temp]['left_init'][x_temp]-1))
                            ap_y2 = int(np.round(data[i_temp]['right_init'][x_temp]))
                            map_ap[ap_y1+d1_spat_id:ap_y2+d1_spat_id, x_temp] = np.int32(ap_num)
    else:
        if N_ap>N_sl:
            print('!!! Warning: Missing %d fiber(s). !!!'%(N_ap-N_sl))
        elif N_ap<N_sl:
            print('!!! Warning: Found %d more fiber(s) than expected. !!!'%(N_sl-N_ap))

        for i_ap in range(N_sl):
            ap_num = i_ap+1
            for x_temp in range(nspec):
                ap_y1 = int(np.round(data[i_ap]['left_init'][x_temp]-1))
                ap_y2 = int(np.round(data[i_ap]['right_init'][x_temp]))
                map_ap[ap_y1:ap_y2, x_temp] = np.int32(ap_num)

    #### mask images
    if (img_mask_flag):
        file_img_mask = img_mask_path+'/img_mask_'+Channel+'_'+img_mask_config
        map_ap = mask_img(map_ap, file_img_mask)

    ####
    num_ap = np.zeros(N_ap, dtype=np.int32)
    for i_ap in range(N_ap):
        num_ap[i_ap] = np.sum(map_ap==i_ap+1)
    num_max = np.max(num_ap)

    #plt.imshow(map_ap, origin='lower')

    ### the following need to be modified
    hdu_map = fits.PrimaryHDU(map_ap)
    hdr_map = hdu_map.header
    hdr_map['IFUTYPE'] = (IFU_type, 'type of IFU')
    #hdr_map.set('IFUTYPE', IFU_type, 'type of IFU')
    hdr_map['NIFU1'] = (N_xx, 'number of IFU columns')
    hdr_map['NIFU2'] = (N_yy, 'number of IFU rows')
    hdr_map['NMAX'] = (num_max, 'maximum number of pixels among all apertures')
    hdr_map['BINNING'] = ('1x1', 'binning')
    #hdu_map = fits.PrimaryHDU(map_ap, header=hdr_map)

    hdu_map.writeto(file_name+'_'+file_date+'.fits',overwrite=True)

def write_pypeit_file(dirname, filename, pca='off', smash_range="0.4,0.6"):
    dirname_output = os.path.join(dirname, 'pypeit_file')
    filename_output = filename+'.pypeit'
    filename_fits = filename+'.fits'
    if not os.path.exists(dirname_output):
        os.mkdir(dirname_output)
    file = open(os.path.join(dirname_output, filename_output), 'w')
    file.write("# User-defined execution parameters\n")
    file.write("[rdx]\n")
    file.write("spectrograph = magellan_m2fs_blue\n")
    file.write("[calibrations]\n")
    file.write("    [[slitedges]]\n")
    if pca=='off':
        file.write("        auto_pca = False\n")
    else:
        file.write("        auto_pca = True\n")
    file.write("        smash_range = %s\n"%smash_range)
    file.write("        length_range = 0.9\n")
    file.write("        edge_thresh = 3.0\n")
    file.write("\n")
    file.write("# Setup\n")
    file.write("setup read\n")
    file.write("    Setup A:\n")
    file.write("setup end\n")
    file.write("\n")
    file.write("# Read in the data\n")
    file.write("data read\n")
    file.write(" path %s\n"%dirname)
    file.write("|         filename |                 frametype | binning | \n")
    file.write("| %s |                     trace |     1,1 |\n"%filename_fits)
    file.write("data end\n")
    file.write("\n")
    file.close()

def write_trace_file(data, header, dirname, filename):
    #### write to a fits file
    X2 = len(data[0])/2
    Y2 = len(data)/2

    hdu_full = fits.PrimaryHDU(data,header=header)
    hdul_full = fits.HDUList([hdu_full])
    hdr_full  = hdul_full[0].header
    if ('BIASSEC' in hdr_full):
        del hdr_full['BIASSEC']
        del hdr_full['DATASEC']
        del hdr_full['TRIMSEC']
    if ('NOVERSCN' in hdr_full):
        del hdr_full['NOVERSCN']
    if ('NBIASLNS' in hdr_full):
        del hdr_full['NBIASLNS']
    #if (flag_egain):
    #    hdr_full['ENOISE'] = np.mean(enoise_full)
    hdr_full['NOPAMPS'] = 1
    hdr_full['OPAMP'] = 1
    hdr_full['FILENAME'] = (filename,'')
    hdr_full['DATASEC'] = ('[1:%d,1:%d]'%(X2*2,Y2*2),'NOAO: data section')
    hdr_full['CCDSEC'] = '[1:%d,1:%d]'%(X2*2,Y2*2)
    hdr_full['BINNING'] = ('1x1', 'binning') # added by YYS on May 11, 2022

    #### save trace file
    path_trace = os.path.join(dirname, filename+'.fits')
    hdul_full.writeto(path_trace,overwrite=True)

    #### save backup file
    dir_backup = os.path.join(dirname, "backup_trace")
    if not os.path.exists(dir_backup):
        os.mkdir(dir_backup)
    path_backup = os.path.join(dir_backup, "%s_trace_%s.fits"%(filename[0:5], datetime.today().strftime('%y%m%d_%H%M')))
    hdul_full.writeto(path_backup,overwrite=False)

def cut_apermap(data, header, dirname, filename):
    #### write to a fits file
    #X2 = len(data[0])/2
    #Y2 = len(data)/2

    hdu_full = fits.PrimaryHDU(data,header=header)
    hdul_full = fits.HDUList([hdu_full])
    #hdr_full  = hdul_full[0].header
    #if ('BIASSEC' in hdr_full):
    #    del hdr_full['BIASSEC']
    #    del hdr_full['DATASEC']
    #    del hdr_full['TRIMSEC']
    #if ('NOVERSCN' in hdr_full):
    #    del hdr_full['NOVERSCN']
    #if ('NBIASLNS' in hdr_full):
    #    del hdr_full['NBIASLNS']
    ##if (flag_egain):
    ##    hdr_full['ENOISE'] = np.mean(enoise_full)
    #hdr_full['NOPAMPS'] = 1
    #hdr_full['OPAMP'] = 1
    #hdr_full['FILENAME'] = (filename,'')
    #hdr_full['DATASEC'] = ('[1:%d,1:%d]'%(X2*2,Y2*2),'NOAO: data section')
    #hdr_full['CCDSEC'] = '[1:%d,1:%d]'%(X2*2,Y2*2)
    #hdr_full['BINNING'] = ('1x1', 'binning') # added by YYS on May 11, 2022

    #### save trace file
    today_temp = datetime.today().strftime("%y%m%d")
    today_backup = datetime.today().strftime("%y%m%d_%H%M")
    
    path_trace = os.path.join(dirname, filename+'_%s_3000.fits'%today_temp)
    hdul_full.writeto(path_trace,overwrite=True)

    #### save backup file
    dir_backup = os.path.join(dirname, "backup_aperMap")
    if not os.path.exists(dir_backup):
        os.mkdir(dir_backup)
    path_backup = os.path.join(dir_backup, "%s_%s.fits"%(filename, today_backup))
    hdul_full.writeto(path_backup,overwrite=False)