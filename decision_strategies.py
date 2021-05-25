import numpy as np
import scipy.stats
import pandas

import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.patches as mpatches
import matplotlib
 
import figurefirst as fifi

import fly_plot_lib.plot as fpl

INTERVALS = np.logspace(-1, 5, 50, base=10)
DISTANCES = np.logspace(1, 4, 50, base=10)

def get_loglikelihood_for_pmf(pmf_array, log_df):
    l = []
    for i in range(len(log_df.distance_travelled_residuals_controlled_for_odor_approach.values)):

        ix_d = np.argmin( np.abs(np.log(DISTANCES) - log_df.distance_travelled_residuals_controlled_for_odor_approach.values[i]) )
        ix_i = np.argmin( np.abs(np.log(INTERVALS) - log_df.interval.values[i]) )

        v = pmf_array[ix_d, ix_i]
        if v < 1e-64:
            v = 1e-64
        l.append(float(v))

    ll = np.sum(np.log(np.array(l)))
    
    return ll

def get_AIC(pmf_array, log_df, nparams):
    ll = get_loglikelihood_for_pmf(pmf_array, log_df)
    AIC = 2*nparams - 2*ll
    #print('AIC: ', AIC)

    return AIC

def optimize_loglikelihood(params, function, log_df):
    '''
    function -- should be something like: get_iGPL_pmf
    params -- guess
    '''
    def go(params):
        x, y, pmf_array = function(params)
        ll = get_loglikelihood_for_pmf(pmf_array, log_df)
        print(ll, params)
        return -1*ll

    r = scipy.optimize.minimize(go, params, method='Nelder-Mead')
    #print(r)
    return r

####################################################################################
### iGPL

iGPL_optimal_params = [0.00884494, 0.91084541, 0.08631313, 5.72153615]

def get_iGPL_model(params, interval):
    alpha_a = params[0] #0.13
    beta_a = params[1] #1.5
    alpha_scale = params[2] #0.3
    beta_scale = params[3] #6.5
    #a = alpha_a*np.log10(interval) + beta_a
    a = np.exp(alpha_a*np.log(interval) + beta_a)
    #scale = np.exp(alpha_scale*np.log10(interval) + beta_scale)
    scale = np.exp(alpha_scale*np.log(interval) + beta_scale)
    
    gamma = scipy.stats.gamma(a, 0, scale)
    return gamma

def get_iGPL_pmf(params=None):
    if params is None:
        params = iGPL_optimal_params
    
    x, y = np.meshgrid(INTERVALS, DISTANCES)
    pmf_array = []
    for i, interval in enumerate(INTERVALS):

        gamma = get_iGPL_model(params, interval)

        pmf = gamma.pdf(DISTANCES)
        pmf /= np.max(pmf)

        pmf_array.append(pmf)
        
    pmf_array = np.vstack(pmf_array).T
    pmf_array /= np.nansum(pmf_array)
    
    #for col in range(1, pmf_array.shape[1]-1):
    #    pmf_array[:, col] = pmf_array[:, col] / np.nansum(pmf_array[:, col])
    
    return x,y,pmf_array

class iGPL(object):
    def __init__(self, params=None):
        if params is None:
            params = iGPL_optimal_params
        self.params = params
        self.name = 'iGPL'
    def get_distance_travelled_given_interval(self, interval):
        return get_iGPL_model(self.params, interval).rvs(1)[0]

####################################################################################
### iGPLoffset

#iGPLoffset_optimal_params = [0.04355854,  0.48557622,  0.06391462,  6.1146875 , 40.94915117]
iGPLoffset_optimal_params = [0.02372199, 0.69689361, 0.07855926, 5.9070129]

def get_iGPLoffset_model(params, interval):
    alpha_a = params[0] #0.13
    beta_a = params[1] #1.5
    alpha_loc = 0
    beta_loc = 0
    alpha_scale = params[2] #0.3
    beta_scale = params[3] #6.5
    #a = alpha_a*np.log10(interval) + beta_a
    a = np.exp(alpha_a*np.log(interval) + beta_a)
    
    loc = 0 #alpha_loc*np.log10(interval) + beta_loc
    #scale = np.exp(alpha_scale*np.log10(interval) + beta_scale)
    scale = np.exp(alpha_scale*np.log(interval) + beta_scale)

    offset = 25 #params[4]
    
    gamma = scipy.stats.gamma(a, offset, scale)
    return gamma

def get_iGPLoffset_pmf(params=None):
    if params is None:
        params = iGPLoffset_optimal_params
    
    x, y = np.meshgrid(INTERVALS, DISTANCES)
    pmf_array = []
    for i, interval in enumerate(INTERVALS):

        gamma = get_iGPLoffset_model(params, interval)

        pmf = gamma.pdf(DISTANCES)
        pmf /= np.max(pmf)

        pmf_array.append(pmf)
        
    pmf_array = np.vstack(pmf_array).T
    pmf_array /= np.nansum(pmf_array)
    
    #for col in range(1, pmf_array.shape[1]-1):
    #    pmf_array[:, col] = pmf_array[:, col] / np.nansum(pmf_array[:, col])
    
    return x,y,pmf_array

class iGPLoffset(object):
    def __init__(self, params=None):
        if params is None:
            params = iGPLoffset_optimal_params
        self.params = params
        self.name = 'iGPLoffset'
    def get_distance_travelled_given_interval(self, interval):
        return get_iGPLoffset_model(self.params, interval).rvs(1)[0]


####################################################################################
### GPL

GPL_optimal_params = [2.43053401, 6.24776816]

def get_GPL_model(params, interval):
    alpha_a = 0  #0.13
    beta_a = params[0] #1.5
    alpha_loc = 0
    beta_loc = 0
    alpha_scale = 0 #0.3
    beta_scale = params[1] #6.5
    a = alpha_a*np.log(interval) + beta_a
    loc = alpha_loc*np.log(interval) + beta_loc
    scale = np.exp(alpha_scale*np.log(interval) + beta_scale)
    gamma = scipy.stats.gamma(a, loc, scale)
    return gamma

def get_GPL_pmf(params=None):
    if params is None:
        params = GPL_optimal_params
    
    x, y = np.meshgrid(INTERVALS, DISTANCES)
    pmf_array = []
    for i, interval in enumerate(INTERVALS):

        gamma = get_GPL_model(params, interval)

        pmf = gamma.pdf(DISTANCES)
        pmf /= np.max(pmf)

        pmf_array.append(pmf)
        
    pmf_array = np.vstack(pmf_array).T
    pmf_array /= np.nansum(pmf_array)
    
    #for col in range(1, pmf_array.shape[1]-1):
    #    pmf_array[:, col] = pmf_array[:, col] / np.nansum(pmf_array[:, col])
    
    return x,y,pmf_array

class GPL(object):
    def __init__(self, params=None):
        if params is None:
            params = GPL_optimal_params
        self.params = params
        self.name = 'GPL'
    def get_distance_travelled_given_interval(self, interval):
        return get_GPL_model(self.params, interval).rvs(1)[0]

####################################################################################
### FPL

FPL_optimal_params = [4604.]

def get_FPL_model(params, interval):
    scale = params[0]
    loc = 0
    expon = scipy.stats.expon(loc, scale)
    return expon

def get_FPL_pmf(params=None):
    if params is None:
        params = FPL_optimal_params
    print(params)
    
    x, y = np.meshgrid(INTERVALS, DISTANCES)
    pmf_array = []
    for i, interval in enumerate(INTERVALS):

        expon = get_FPL_model(params, interval)

        pmf = expon.pdf(DISTANCES)
        pmf /= np.max(pmf)

        pmf_array.append(pmf)
        
    pmf_array = np.vstack(pmf_array).T
    pmf_array /= np.nansum(pmf_array)
    
    #for col in range(1, pmf_array.shape[1]-1):
    #    pmf_array[:, col] = pmf_array[:, col] / np.nansum(pmf_array[:, col])
    
    return x,y,pmf_array

class FPL(object):
    def __init__(self, params=None):
        if params is None:
            params = FPL_optimal_params
        self.params = params
        self.name = 'FPL'
    def get_distance_travelled_given_interval(self, interval):
        return get_FPL_model(self.params, interval).rvs(1)[0]

####################################################################################
### TGUD

TGUD_optimal_params = [6.107442769268304,]

def get_TGUD_model(params, interval):
    return np.exp(params[0])

def get_TGUD_pmf(params=None):
    if params is None:
        params = TGUD_optimal_params
    print(params)
    
    x, y = np.meshgrid(INTERVALS, DISTANCES)
    pmf_array = []
    for i, interval in enumerate(INTERVALS):

        d = get_TGUD_model(params, interval)

        pmf = np.zeros_like(DISTANCES)
        ixd = np.argmin( np.abs(DISTANCES-d))
        pmf[ixd] = 1

        pmf_array.append(pmf)
        
    pmf_array = np.vstack(pmf_array).T
    pmf_array /= np.nansum(pmf_array)
    
    #for col in range(1, pmf_array.shape[1]-1):
    #    pmf_array[:, col] = pmf_array[:, col] / np.nansum(pmf_array[:, col])
    
    return x,y,pmf_array

class TGUD(object):
    def __init__(self, params=None):
        if params is None:
            params = TGUD_optimal_params
        self.params = params
        self.name = 'TGUD'
    def get_distance_travelled_given_interval(self, interval):
        return get_TGUD_model(self.params, interval)


####################################################################################
### TGUDnoisy

TGUDnoisy_optimal_params = [6.22201146, 0.91367223]

def get_TGUDnoisy_pmf(params=None):
    if params is None:
        params = TGUDnoisy_optimal_params
    print(params)
    
    x, y = np.meshgrid(INTERVALS, DISTANCES)

    pmf_array = np.zeros([len(DISTANCES), len(INTERVALS)])
    for g in range(2000):
        for i in INTERVALS:
            d_pred = params[0] + np.random.normal(0, params[1])
            d_ix = np.argmin( np.abs(np.log(DISTANCES) - d_pred) )
            i_ix = np.argmin( np.abs(np.log(INTERVALS) - np.log(i)) )

            pmf_array[d_ix, i_ix] += 1

    pmf_array = scipy.ndimage.gaussian_filter(pmf_array, 1)
    
    for col in range(pmf_array.shape[1]):
        pmf_array[:,col] /= np.sum(pmf_array[:,col])
    pmf_array /= np.nansum(pmf_array)
    
    return x,y,pmf_array


####################################################################################
### iTGUD

iTGUD_optimal_params = [5.94787066, 0.10413817]

def get_iTGUD_model(params, interval):
    return np.exp(params[0] + np.log(interval)*params[1])

def get_iTGUD_pmf(params=None):
    if params is None:
        params = iTGUD_optimal_params
    print(params)
    
    x, y = np.meshgrid(INTERVALS, DISTANCES)
    pmf_array = []
    for i, interval in enumerate(INTERVALS):

        d = get_iTGUD_model(params, interval)

        pmf = np.zeros_like(DISTANCES)
        ixd = np.argmin( np.abs(DISTANCES-d))
        pmf[ixd] = 1

        pmf_array.append(pmf)
        
    pmf_array = np.vstack(pmf_array).T
    pmf_array /= np.nansum(pmf_array)
    
    #for col in range(1, pmf_array.shape[1]-1):
    #    pmf_array[:, col] = pmf_array[:, col] / np.nansum(pmf_array[:, col])
    
    return x,y,pmf_array

class iTGUD(object):
    def __init__(self, params=None):
        if params is None:
            params = iTGUD_optimal_params
        self.params = params
        self.name = 'iTGUD'
    def get_distance_travelled_given_interval(self, interval):
        return get_iTGUD_model(self.params, interval)

####################################################################################
### iTGUDnoisy

iTGUDnoisy_optimal_params = [0.08362859, 5.8837739 , 7.68934984, 0.59504161]

def get_iTGUDnoisy_pmf(params=None):

    INTERVALS = np.logspace(-1, 5, 50, base=10)

    #INTERVALS = np.hstack((np.logspace(-10,-1,88,base=10)[0:-1], INTERVALS))
    #INTERVALS = np.hstack((INTERVALS, np.logspace(5,15,80,base=10)[1:]))

    if params is None:
        params = iTGUDnoisy_optimal_params
    print(params)
    
    x, y = np.meshgrid(INTERVALS, DISTANCES)

    slope = params[0]
    intercept = params[1]
    interval_std = params[2]
    distance_std = params[3]

    pmf_array = np.zeros([len(DISTANCES), len(INTERVALS)])
    for g in range(3000):
        for i in INTERVALS:

            i_noisy = (np.log(i) + np.random.normal(0, interval_std))

            d_pred = (i_noisy*slope + intercept) + np.random.normal(0, distance_std)

            if d_pred < np.max(np.log(DISTANCES)):
                d_ix = np.argmin( np.abs(np.log(DISTANCES) - d_pred) )
                i_ix = np.argmin( np.abs(np.log(INTERVALS) - np.log(i)) )

            pmf_array[d_ix, i_ix] += 1

    #pmf_array = pmf_array[:, 88:88+50]
    pmf_array = scipy.ndimage.gaussian_filter(pmf_array, 1)
    
    for col in range(pmf_array.shape[1]):
        pmf_array[:,col] /= np.sum(pmf_array[:,col])
    #pmf_array[0,:] = 0
    #pmf_array[-1,:] = 0
    #pmf_array /= np.nansum(pmf_array)
    
    
    pmf_array /= np.nansum(pmf_array)
    
    return x,y,pmf_array

####################################################################################
### iTGUDnoisyD

iTGUDnoisyD_optimal_params = [0.10666455, 5.75376824, 0.84713312]

def get_iTGUDnoisyD_pmf(params=None):

    INTERVALS = np.logspace(-1, 5, 50, base=10)

    if params is None:
        params = iTGUDnoisyD_optimal_params
    print(params)
    
    x, y = np.meshgrid(INTERVALS, DISTANCES)

    slope = params[0]
    intercept = params[1]
    distance_std = params[2]

    pmf_array = []
    for i, interval in enumerate(INTERVALS):

        normal = scipy.stats.norm(np.log(interval)*slope + intercept, distance_std)

        pmf = normal.pdf(np.log(DISTANCES))
        pmf /= np.max(pmf)

        pmf_array.append(pmf)
    
    pmf_array = np.vstack(pmf_array).T
    pmf_array /= np.nansum(pmf_array)
    
    return x,y,pmf_array


    
    return x,y,pmf_array

####################################################################################
### Run the models

def run_model(interval, pfood, pfoodstep, model):
    '''
    model -- should be an instance of a model class, like iGPL()
    '''
    
    foundfood = False
    travel_times = []
    search_distances = []
    
    N = 0
    while not foundfood:
        try:
            t = interval.rvs()
        except:
            t = interval
        if t < 0:
            t = 1 # protection
        travel_times.append(t)
        
        # gamma
        
        search_distance = model.get_distance_travelled_given_interval(interval)
        search_distances.append(search_distance)
        
        N += 1
        try:
            pfoodval = pfood.rvs()
        except:
            pfoodval = pfood
        if np.random.uniform(0, 1) < pfoodval:
            # there is food to be found
            try:
                if np.random.uniform(0, 1) < scipy.stats.expon.cdf(search_distance, scale=pfoodstep.rvs()):
                    foundfood = True
            except:
                if np.random.uniform(0, 1) < scipy.stats.expon.cdf(search_distance, scale=pfoodstep):
                    foundfood = True
            
    travel_times = np.array(travel_times)
    search_distances = np.array(search_distances)
    search_times = np.exp(0.9952*np.log(search_distances) - 0.2) # comes from fit to the data
    
    search_times_patch = np.sum(search_times)
    search_times_flight = np.sum(travel_times)
    
    return N, search_times_patch, search_times_flight

def iterate_run_model(args, iterations=1000, return_means=True):
    interval, pfood, pfoodstep, model = args
    search_times_patch = []
    search_times_flight = []
    Ns = []
    for i in range(iterations):
        N, stp, stf = run_model(interval, pfood, pfoodstep, model)
        search_times_patch.append(stp)
        search_times_flight.append(stf)
        Ns.append(N)
    if return_means:
        return np.mean(Ns), np.std(Ns), np.mean(search_times_patch), np.std(search_times_patch), np.mean(search_times_flight), np.std(search_times_flight)
    else:
        return Ns, ts

def iterate_through_scenarios(models, 
                              iterations = 1000,
                              pfoodsteps = [10, 100, 1000, 10000, 100000], 
                              pfoods = [0.2, 0.8],
                              name_prefix='simulations'):
    '''
    models -- should be a list of instances of model classes, like iGPL()
    '''
    
    intervals = [1, 10, 100, 1000, 10000, 100000, 1000000]
    

    df = None

    for model in models:
        for pfood in pfoods:
            for pfoodstep in pfoodsteps:
                for interval in intervals:
                    args = [interval, pfood, pfoodstep, model]
                    meanN, stdN, mean_search_times_patch, std_search_times_patch, mean_search_times_flight, std_search_times_flight = iterate_run_model(args, iterations=iterations)
                    
                    print(model.name, pfood, pfoodstep, interval, meanN)

                    new_df = pandas.DataFrame({'pfood': [pfood],
                                               'pfoodstep': [pfoodstep],
                                               'interval': [interval],
                                               'mean_Npatches': [meanN],
                                               'std_Npatches': [stdN],
                                               'mean_search_time_patch': [mean_search_times_patch],
                                               'std_search_time_patch': [std_search_times_patch],
                                               'mean_search_time_flight': [mean_search_times_flight],
                                               'std_search_time_flight': [std_search_times_flight],
                                               'model': [model.name]}, index=[0,])
                    if df is None:
                        df = new_df
                    else:
                        df = pandas.concat([df, new_df], ignore_index=True)

    model_names = [model.name for model in models]
    name = ''
    for model_name in model_names:
        name += model_name + '_'
    name = name[0:-1]
    name = name_prefix + '_' + name + '.hdf'

    df.to_hdf(name, 'simulations')


################################################################################################
### Plot

def plot_itgud(model_pmf_function, log_df, fififig='figures/foraging_fig_4_revision_v2.svg', ticklabels=True):
    axname = str(model_pmf_function).split(' ')[1].split('_')[1]

    layout = fifi.svg_to_axes.FigureLayout(fififig, autogenlayers=True, 
                                           make_mplfigures=True, hide_layers=[])
    ax = layout.axes[(axname, axname)]

    #x, y, pmf = model_pmf_function()
    #im = ax.pcolormesh(x, y, pmf, cmap='bone_r', rasterized=True, zorder=0, vmin=0, vmax=0.001)

    x = np.log(INTERVALS)
    y = np.exp(x*iTGUD_optimal_params[1] + iTGUD_optimal_params[0])
    ax.plot(INTERVALS, y, linewidth=1.5, color='black')

    rect = mpatches.Rectangle((0, 0), 1e5, 1e5, ec="none", facecolor='white', alpha=0.25, zorder=1)
    ax.add_artist(rect)

    ax.scatter( np.exp(log_df.interval), np.exp(log_df.distance_travelled_residuals_controlled_for_odor_approach), 
               c='black', s=3, linewidth=0.25,
               rasterized=True, zorder=10)


    ax.set_xscale('log')
    ax.set_yscale('log')


    ax.set_ylim(1e1, 1e4)
    ax.set_xlim(10**-1, 10**5)
    yticks = [10**1, 10**2, 10**3, 10**4]
    yticklabels = ['$10^1$', '$10^2$', '$10^3$', '$10^4$']

    xticks = [10**-1, 1, 10**1, 10**2, 10**3, 10**4, 10**5]
    xticklabels = ['$10^-1$', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$']


    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'], 
                                     yticks=yticks, 
                                     xticks=xticks,
                                     linewidth=0.5, tick_length=2.5, 
                                     spine_locations={'left': 2.5, 'bottom': 2.5})
    if not ticklabels:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    ax.minorticks_off()

    fifi.mpl_functions.set_fontsize(ax.figure, 6)
    layout.append_figure_to_layer(layout.figures[axname], axname, cleartarget=True)
    layout.write_svg(fififig)


def plot_scatter_and_pmf(model_pmf_function, log_df, fififig='figures/foraging_fig_4_revision_v2.svg', ticklabels=True):
    axname = str(model_pmf_function).split(' ')[1].split('_')[1]
    if axname == 'iTGUD':
        plot_itgud(model_pmf_function, log_df, fififig, ticklabels)
        return

    layout = fifi.svg_to_axes.FigureLayout(fififig, autogenlayers=True, 
                                           make_mplfigures=True, hide_layers=[])
    ax = layout.axes[(axname, axname)]

    x, y, pmf = model_pmf_function()

    im = ax.pcolormesh(x, y, pmf, cmap='bone_r', rasterized=True, zorder=0, vmin=0, vmax=0.001)

    rect = mpatches.Rectangle((0, 0), 1e5, 1e5, ec="none", facecolor='white', alpha=0.25, zorder=1)
    ax.add_artist(rect)

    ax.scatter( np.exp(log_df.interval), np.exp(log_df.distance_travelled_residuals_controlled_for_odor_approach), 
               c='black', s=3, linewidth=0.25,
               rasterized=True, zorder=10)


    ax.set_xscale('log')
    ax.set_yscale('log')


    ax.set_ylim(1e1, 1e4)
    ax.set_xlim(10**-1, 10**5)
    yticks = [10**1, 10**2, 10**3, 10**4]
    yticklabels = ['$10^1$', '$10^2$', '$10^3$', '$10^4$']

    xticks = [10**-1, 1, 10**1, 10**2, 10**3, 10**4, 10**5]
    xticklabels = ['$10^-1$', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$']


    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'], 
                                     yticks=yticks, 
                                     xticks=xticks,
                                     linewidth=0.5, tick_length=2.5, 
                                     spine_locations={'left': 2.5, 'bottom': 2.5})
    if not ticklabels:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    ax.minorticks_off()

    fifi.mpl_functions.set_fontsize(ax.figure, 6)
    layout.append_figure_to_layer(layout.figures[axname], axname, cleartarget=True)
    layout.write_svg(fififig)

def plot_colorbar(fififig=None):
    if fififig is None:
        fififig = 'figures/foraging_fig_4_revision_v2.svg'
    layout = fifi.svg_to_axes.FigureLayout(fififig, autogenlayers=True, 
                                           make_mplfigures=True, hide_layers=[])
    ax = layout.axes[('colorbar', 'colorbar')]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fpl.colorbar(ax, colormap='bone_r', orientation='vertical', ticks=[0, 1])

    rect = mpatches.Rectangle((-0.1, -0.1), 1.2, 1.2, ec="none", facecolor='white', alpha=0.25, zorder=1)
    ax.add_artist(rect)

    layout.append_figure_to_layer(layout.figures['colorbar'], 'colorbar', cleartarget=True)
    layout.write_svg(fififig)


def plot_goodness_of_fit(model_pmf_function, log_df, fififig='supplemental_figures/foraging_fig_s_models_stats.svg', ticklabels=True):
    axname = str(model_pmf_function).split(' ')[1].split('_')[1] + '_LL'

    layout = fifi.svg_to_axes.FigureLayout(fififig, autogenlayers=True, 
                                           make_mplfigures=True, hide_layers=[])
    ax = layout.axes[(axname, axname)]

    x, y, pmf = model_pmf_function()
    pmf_int = ((pmf / np.max(pmf))*1000).astype(int)
    
    pmf_distance_choices = []
    for col in range(pmf_int.shape[1]):
        pmf_distance_choices_for_col = []
        for row in range(pmf_int.shape[0]):
            if pmf_int[row,col] > 0:
                pmf_distance_choices_for_col.extend([DISTANCES[row],]*pmf_int[row,col])
        pmf_distance_choices.append(np.array(pmf_distance_choices_for_col))
        
    true_ll = get_loglikelihood_for_pmf(pmf, log_df)



    loglikelihoods_model = []

    for i in range(1000):

        idx_intervals = np.random.randint(0, log_df.shape[0], log_df.shape[0])
        model_intervals = np.exp(log_df.interval.values)#[idx_intervals])
        model_distances = []
        
        for i in model_intervals:
            ix = np.argmin(np.abs(INTERVALS - i))
            d = np.random.choice(pmf_distance_choices[ix], 1)[0]
            model_distances.append(d)
            
        model_distances = np.array(model_distances)
        
        model_df = pandas.DataFrame({'interval': np.log(model_intervals),
                                     'distance_travelled_residuals_controlled_for_odor_approach': np.log(model_distances)})

        ll = get_loglikelihood_for_pmf(pmf, model_df)
        loglikelihoods_model.append(ll)


    print(true_ll)

    h, b, u = ax.hist(loglikelihoods_model, color='black')
    ax.vlines(true_ll, 0, max(h), color='red')


    xticks = [int(np.min(b)), int(np.max(b))]
    fifi.mpl_functions.adjust_spines(ax, ['top'], 
                                     xticks=xticks, 
                                     #xticks=xticks,
                                     linewidth=0.5, tick_length=1.5, 
                                     spine_locations={'top': 1.5})
    #if not ticklabels:
    #    ax.set_yticklabels([])
    #    ax.set_xticklabels([])

    ax.minorticks_off()


    if 1:
        dy = -3/72.; dx = 0/72. 
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, ax.figure.dpi_scale_trans)
        # apply offset transform to all x ticklabels.
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)


    fifi.mpl_functions.set_fontsize(ax.figure, 5)
    layout.append_figure_to_layer(layout.figures[axname], axname, cleartarget=True)
    layout.write_svg(fififig)


if __name__ == '__main__':

    models = [TGUD(), iTGUD(), GPL(), iGPL()]
    #models = [GPL(), iGPL()]
    #iterate_through_scenarios(models)

    iterate_through_scenarios(models, 
                              iterations = 50000,
                              pfoodsteps = [1, 10, 100, 500, 1000, 5000], #pfoodsteps = [1,10,100,1000,10000], #pfoodsteps = [500, 1000, 1500, 2000], 
                              pfoods = [0.2, 0.8], 
                              name_prefix='realistic_lambdas')