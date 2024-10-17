import numpy as np
import astropy.units as u

from ccdproc import Combiner
from specutils.spectra import Spectrum1D

#### code refactored from Matt's m2fs_process.py
class columnspec:
    def __init__(self,columns=None,spec=None,mask=None,err=None,
                 pixel=None,continuum=None,rms=None,
                 apertures_initial=None,apertures_profile=None):
        self.columns=columns
        self.spec=spec
        self.mask=mask
        self.err=err
        self.pixel=pixel
        self.continuum=continuum
        self.rms=rms
        self.apertures_initial=apertures_initial
        self.apertures_profile=apertures_profile


def stdmean(a,axis=None,dtype=None,out=None,ddof=0,keepdims=np._NoValue):
    std=np.ma.std(a)
#    print(len(a))
    return std/np.sqrt(len(a))


def column_stack(data,col):
    '''
    stack a column of data, and return a Spectrum1D object
    '''

    column=data.data[:,col]
    column_uncertainty=data.uncertainty[:,col]
    column_mask=data.mask[:,col]

    stack=[]
    sig_stack=[]
    for j in range(0,len(column[0])):
        stack.append(data[:,col[j]])
        sig_stack.append(data.uncertainty._array[:,col[j]])
    sig_stack=np.array(sig_stack)
    c=Combiner(stack)
    c.weights=1./sig_stack**2
    comb=c.average_combine(uncertainty_func=stdmean)
    lamb=np.arange(len(comb.data),dtype='float')
    # unit is pixels, but specutils apparently can't handle that, 
    # so we lie and say Angs.
    spec1d=Spectrum1D(flux=np.array(comb.data)*u.electron,
                      spectral_axis=lamb*u.AA,
                      uncertainty=comb.uncertainty,
                      mask=comb.mask)
    return spec1d


def get_columnspec(data, trace_step, n_lines, verbose=False):
    '''
    get a list of column spectra from a 2D data array
    '''

    n_cols=np.shape(data)[1]
    trace_n=np.int64(n_cols/trace_step)
#    print(n_cols,trace_n)
    trace_cols=np.linspace(0,n_cols,trace_n,dtype='int')

    columnspec_array=[]
    for i in range(0,len(trace_cols)-1):
        if verbose:
            print('working on '+str(i+1)+' of '+str(len(trace_cols))+' trace columns')
        col0=np.arange(n_lines)+trace_cols[i]
        spec1d0=column_stack(data,col0)
        pixel0=(np.arange(len(spec1d0.data),dtype='float'))*u.AA
        # unit is pixels, but specutils apparently can't handle that, 
        # so we lie and say Angs.
        columnspec0=columnspec(columns=col0,spec=spec1d0.data,pixel=pixel0)
        columnspec_array.append(columnspec0)
        
    return columnspec_array