import numpy as np 

def smoother(wave,flux,ivar,sigma=5,n_sigma=5,boxcar_size=50,method='gauss_dist_and_ivar'):
    #final smoothed spectrum array
    smooth_flux = np.zeros_like(wave)

    #possible methods for smoothing
    poss_methods = ['gauss_dist_only','ivar_only','gauss_dist_and_ivar','weighted_average','median']
    #gauss_dist_only = smooth using only gaussian distance from each point
    #ivar_only = smooth only using the ivars as weights
    #gauss_dist_and_ivar = use both gaussian distance and ivar weights
    #weighted_average = simple boxcar that uses a weighted average (with ivar)
    #median = simple boxcar that just calculates median around a point
    if method not in poss_methods:
        #print(f'WARNING: chosen method of {method} is not valid. Setting to "gauss_dist_and_ivar" method.')
        method = 'gauss_dist_and_ivar'
        
    #sigma and n_sigma only matter for methods in ['gauss_dist_only', 'ivar_only', 'gauss_dist_and_ivar']
    #sigma is the size of smoothing length in Angstroms. Bigger is more smoothing, smaller is less
    #n_sigma sets the size of boxcar (shouldn't be smaller than 3)
    #boxcar_size only matter for methods in ['weighted_average', 'median']

    #if method not in poss_methods:
        #raise ValueError(f'ERROR: chosen method of {method} is not valid. Try again.')

    #medians for padding onto the left and right edges
    stat_length = 100 #angstroms from left and right edges to calculate median values
    medians = np.median(flux[(wave < wave[0]+stat_length)]),np.median(flux[(wave > wave[-1]-stat_length)])        
    median_ivars = np.median(ivar[(wave < wave[0]+stat_length)]),np.median(ivar[(wave > wave[-1]-stat_length)])        
        
    for i in range(len(wave)):
        curr_wave = wave[i]
        if method in ['gauss_dist_only','ivar_only','gauss_dist_and_ivar']:
            dist = n_sigma*sigma
        elif method in ['weighted_average','median']:
            dist = boxcar_size*0.5
        region = (wave >= curr_wave-dist) & (wave < curr_wave+dist)

        if curr_wave-dist < wave[0]:
            spacing = np.average(np.diff(wave[region])) #get average wavelength spacing for padding
            npad = max(int(round((wave[0]-(curr_wave-dist))/spacing,0)),1)
            
            dx = np.zeros(np.sum(region)+npad)
            flux_slice = np.zeros(np.sum(region)+npad)
            ivar_slice = np.zeros(np.sum(region)+npad)
            
            dx[:npad] = wave[0]-(np.arange(npad)+1)*spacing
            dx[npad:] = wave[region]
            dx -= curr_wave
            
            flux_slice[:npad] = medians[0]
            flux_slice[npad:] = flux[region]
            ivar_slice[:npad] = median_ivars[0]
            ivar_slice[npad:] = ivar[region]
        elif curr_wave+dist > wave[-1]:
            spacing = np.average(np.diff(wave[region])) #get average wavelength spacing for padding
            npad = max(int(round(((curr_wave+dist)-wave[-1])/spacing,0)),1)

            dx = np.zeros(np.sum(region)+npad)
            flux_slice = np.zeros(np.sum(region)+npad)
            ivar_slice = np.zeros(np.sum(region)+npad)
            
            dx[:np.sum(region)] = wave[region]
            dx[np.sum(region):] = wave[-1]+(np.arange(npad)+1)*spacing
            dx -= curr_wave
            
            flux_slice[:np.sum(region)] = flux[region]
            flux_slice[np.sum(region):] = medians[1]
            ivar_slice[:np.sum(region)] = ivar[region]  
            ivar_slice[np.sum(region):] = median_ivars[1]   
        else:
            dx = wave[region]-curr_wave
            flux_slice = flux[region]
            ivar_slice = ivar[region]
        if method == 'gauss_dist_only':
            weights = np.exp(-0.5*np.power(dx/sigma,2))
        elif method in ['ivar_only','weighted_average']:
            weights = ivar_slice
        elif method == 'gauss_dist_and_ivar':
            weights = ivar_slice*np.exp(-0.5*np.power(dx/sigma,2))
        elif method == 'median':
            weights = np.ones(len(flux_slice)) #equal weights
        weights /= np.sum(weights) #sum of weights should always be 1
        if method == 'median':
            smooth_flux[i] = np.median(flux_slice)
        else:
            smooth_flux[i] = np.sum(flux_slice*weights)
    smooth_flux[np.logical_not(np.isfinite(smooth_flux))] = 0
    return smooth_flux