"""Name: py_specrebin_vec.py

Purpose: A python version of x_specrebin, modified such that input and output fluxes are the same.

For spec_interp and rebinspec:
Author: K. M. Hamren
Date: Jan. 2014

For spec_interp_vec and rebinspec_vec:
Author: A. Bhattacharya
Date: Mar. 2022


Examples:

1. Put in just the required arguments, return new rebinned flux array

    f2 = rebinspec(l,f,l2)    OR    f2 = rebinspec_vec(l,f,l2)

2. Put in required arguments + variance, return new rebinned flux array and a rebinned variance array

    f2, var2 = rebinspec(l,f,l2,var = var)    OR    f2, var2 = rebinspec_vec(l,f,l2,var = var)

3. Put in required arguments + ivar, return new rebinned flux array and a rebinned ivar

    f2, ivar2 = rebinspec(l,f,l2,ivar = ivar)    OR    f2, ivar2 = rebinspec_vec(l,f,l2,ivar = ivar)

"""

import numpy as np
import warnings

warnings.simplefilter("ignore", RuntimeWarning)


def spec_interp(wv,fx,nwwv,*args):
    #Required arguments:
    #   - wv: old wavelength array
    #   - fx: flux to be rebinned
    #   - nwwv: new wavelength array
    #
    #Optional argument: variance

    #Optional var argument
    npix = len(wv)
    if len(args) == 0:
        var = np.ones(npix)
        nwvarFlag = False
    else:
        var = args[0]
        nwvarFlag = True

    nwpix = len(nwwv)

    #Calculate wavelength endpoints for each pixel

    wvl = (wv + np.roll(wv,1))/2.
    wvh = (wv + np.roll(wv,-1))/2.
    wvl[0] = wv[0] - (wv[1] - wv[0])/2.
    wvh[npix-1] = wv[npix-1] + (wv[npix-1]-wv[npix-2])/2.

    #Calculate endpoints of the final array
    bwv = np.zeros(nwpix+1)
    bwv[0:nwpix] = (nwwv+np.roll(nwwv,1))/2.
    bwv[0] = nwwv[0] - (nwwv[1] - nwwv[0])/2.
    bwv[nwpix] = nwwv[nwpix-1]+(nwwv[nwpix-1] - nwwv[nwpix - 1])/2.

    #Create tmp arrays for final array

    nwfx = np.zeros(nwpix)
    nwvar = np.zeros(nwpix)
    nwunitfx = np.zeros(nwpix)

    #Loop through the arrays
    for q in range(npix):

        #No overlap
        if (wvh[q] <= bwv[0]) | (wvl[q] >= bwv[nwpix]):
            continue

        #Find pixel that bw is within
        if wvl[q] <= bwv[0]:
            i1 = [0]
        else:
            i1 = np.argwhere((wvl[q] <= np.roll(bwv,-1)) & (wvl[q] > bwv))[0]

        if wvh[q] > bwv[nwpix]:
            i2 = [nwpix-1]
        else:
            i2 = np.argwhere((wvh[q] <= np.roll(bwv,-1)) & (wvh[q] > bwv))[0]

        j1 = i1[0]
        j2 = i2[0]

        #Now Sum up
        for kk in range(j1,j2+1):
            #Rejected pixesl do not get added in
            if var[q] > 0.:
                frac = ( np.min([wvh[q],bwv[kk+1]]) - np.max([wvl[q],bwv[kk]]) ) / (wvh[q]-wvl[q])
                nwfx[kk] = nwfx[kk]+frac*fx[q]
                nwunitfx[kk] = nwunitfx[kk]+frac*1.0

                #Variance
                if nwvarFlag:
                    if (var[q] <= 0.) | (nwvar[kk] == -1):
                       nwvar[kk] = -1
                    else:
                       nwvar[kk] = nwvar[kk]+frac*var[q]

    if nwvarFlag:
        fxOut = nwfx/nwunitfx
        varOut = nwvar*nwunitfx
        
        return fxOut,varOut
    else:
        fxOut = nwfx/nwunitfx
        return fxOut


def rebinspec(*args,**kwargs):
    #Required arguments:
    #   - wv: old wavelength array
    #   - fx: flux to be rebinned
    #   - nwwv: new wavelength array
    #
    #Optional arguments:
    #   - var = var, input and output variance
    #   - ivar = ivar, input and output ivar

    if len(args) != 3:
        print('Proper syntax is: out = rebinspec(wv, fx, nwwv, **kwargs)')
        return np.nan

    else:
        wv, fx, nwwv = args

        var = kwargs.get('var',None)
        ivar = kwargs.get('ivar',None)

        if (var is not None) & (ivar is None):
            nwfx,nwvar = spec_interp(wv,fx,nwwv,var)

            return nwfx, nwvar
        elif (var is None) & (ivar is not None):
            var = 1./ivar
            nwfx,nwvar_1 = spec_interp(wv,fx,nwwv,var)
            nwvar_1[nwvar_1 == 0.0] = -10.0
            nwivar = 1.0/nwvar_1
            nwivar[nwivar < 0.0] = 0.0
            
            return nwfx, nwivar
        else:
            nwfx = spec_interp(wv,fx,nwwv)

            return nwfx


def spec_interp_vec(wv,fx,nwwv,*args):
    # Vectorized version of spec_interp
    # 
    # Required arguments:
    #   - wv: old wavelength array
    #   - fx: flux to be rebinned
    #   - nwwv: new wavelength array
    #
    #Optional argument: variance

    #Optional var argument
    npix = len(wv)
    if len(args) == 0:
        var = np.ones(npix)
        nwvarFlag = False
    else:
        var = args[0]
        nwvarFlag = True

    nwpix = len(nwwv)

    #Calculate wavelength endpoints for each pixel

    wvl = (wv + np.roll(wv,1))/2.
    wvh = (wv + np.roll(wv,-1))/2.
    wvl[0] = wv[0] - (wv[1] - wv[0])/2.
    wvh[npix-1] = wv[npix-1] + (wv[npix-1] - wv[npix-2])/2.

    #Calculate endpoints of the final array
    bwv = np.zeros(nwpix+1)
    bwv[0:nwpix] = (nwwv+np.roll(nwwv,1))/2.
    bwv[0] = nwwv[0] - (nwwv[1] - nwwv[0])/2.
    bwv[nwpix] = nwwv[nwpix-1] + (nwwv[nwpix-1] - nwwv[nwpix-2])/2.

    #Create tmp arrays for final array
    nwfx = np.zeros(nwpix)
    nwvar = np.zeros(nwpix)
    nwunitfx = np.zeros(nwpix)

    #No overlap & variance>0
    chosen_indices = np.argwhere((wvh > bwv[0]) & (wvl < bwv[nwpix]) & (var > 0)).ravel()

    wv_diff = wvh - wvl

    j1 = np.zeros(npix, dtype=int)
    j1[chosen_indices] = np.searchsorted(bwv, wvl[chosen_indices]) - 1
    j1[j1<0] = 0

    j2 = np.zeros(npix, dtype=int)
    j2[chosen_indices] = np.searchsorted(bwv, wvh[chosen_indices])
    j2[wvh>bwv[nwpix]] = nwpix

    wvh[j2==nwpix] = bwv[nwpix]

    if nwvarFlag:

        #Loop through the array (and sum up the flux & variance)
        for q in chosen_indices:

            #Flux
            l_ind, h_ind = j1[q], j2[q]
            bwv_slice = np.zeros(h_ind-l_ind+1)
            bwv_slice[1:-1] = bwv[l_ind+1:h_ind]
            bwv_slice[-1] = wvh[q]
            bwv_slice[0] = wvl[q]
            fracs = np.diff(bwv_slice)/wv_diff[q]
            nwfx[l_ind:h_ind] += fracs*fx[q]
            nwunitfx[l_ind:h_ind] += fracs

            #Variance
            nwvar[l_ind:h_ind] += fracs*var[q]

        fxOut = nwfx/nwunitfx
        varOut = nwvar*nwunitfx
        
        return fxOut, varOut

    else:

        #Loop through the array (and sum up the flux)
        for q in chosen_indices:

            #Flux
            l_ind, h_ind = j1[q], j2[q]
            bwv_slice = np.zeros(h_ind-l_ind+1)
            bwv_slice[1:-1] = bwv[l_ind+1:h_ind]
            bwv_slice[-1] = wvh[q]
            bwv_slice[0] = wvl[q]
            fracs = np.diff(bwv_slice)/wv_diff[q]
            nwfx[l_ind:h_ind] += fracs*fx[q]
            nwunitfx[l_ind:h_ind] += fracs

        fxOut = nwfx/nwunitfx

        return fxOut


def rebinspec_vec(*args,**kwargs):
    # Vectorized version of rebinspec.
    # 
    # Required arguments:
    #   - wv: old wavelength array
    #   - fx: flux to be rebinned
    #   - nwwv: new wavelength array
    #
    #Optional arguments:
    #   - var = var, input and output variance
    #   - ivar = ivar, input and output ivar

    if len(args) != 3:
        print('Proper syntax is: out = rebinspec(wv, fx, nwwv, **kwargs)')
        return np.nan

    else:
        wv, fx, nwwv = args

        var = kwargs.get('var',None)
        ivar = kwargs.get('ivar',None)

        if (var is not None) & (ivar is None):
            nwfx,nwvar = spec_interp_vec(wv,fx,nwwv,var)

            return nwfx, nwvar
        
        elif (var is None) & (ivar is not None):
            var = 1./ivar
            nwfx,nwivar = spec_interp_vec(wv,fx,nwwv,var)
            nonzero_iv = (nwivar != 0.0)
            nwivar[nonzero_iv] = 1.0/nwivar[nonzero_iv]
            
            return nwfx, nwivar

        else:
            nwfx = spec_interp_vec(wv,fx,nwwv)

            return nwfx
