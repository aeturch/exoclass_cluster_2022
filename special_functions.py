from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.interpolate import interp1d
import os
import sys
import dill
from scipy.special import erfc

from .__init__ import DarkAgesError as err
data_dir = os.path.join( os.path.dirname(os.path.realpath( __file__ )), 'data' )

# def boost_fit(rs):
#   # kcut/krh = 10

# def boost_interp(cut):
#   kcut_krh = cut
#   # directory
#   fp = '/Users/alice/Documents/class_folder/more_boost_files/'

#   # read redshift array
#   # with open(fp+'zList.txt','r') as f1:
#   #   rs_list = [float(line.rstrip('\n')) for line in f1]
#   z_arr = np.genfromtxt(fp+'zList.txt')

#   # read boost (with kcut/krh passed as an argument)
#   # with open(fp+'e_boost_'+str(kcut_krh)+'.txt','r') as f2:
#   #   boost = [float(line.rstrip('\n')) for line in f2]
#   boost_arr = np.genfromtxt(fp+'boost_'+str(int(kcut_krh))+'.txt')

#   # get interpolating function
#   # f_boost = interp1d(rs_list,boost)
#   f_boost = interp1d(z_arr,boost_arr)

#   return f_boost

def bfnc(ns):
    ### currently for T_RH = 1 GeV
    if ns >= 0.9665:
        b = 52.368421052632*ns - 49.614078947368
       
    elif ns < 0.9665:
        b = 29.802631578947*ns - 27.804243421053
    return b

def afnc(ns):
    ### currently for T_RH = 1 GeV
    a = 15.131578947368*ns - 13.624671052632
    return a

def get_fitted_boost(z,p):
    if z < p['tail_rs'][0]:
        return p['B0']*(1+(z/p['z_b'])**p['n'])**(p['alpha']/p['n'])
    else:
        for zv,pt in zip(p['tail_rs'][1:],p['tail_params']):
            if z <= zv:
                return np.exp(pt[1])*(z**pt[0])
        if z > p['tail_rs'][-1]:
            tp = p['tail_params'][-1]
            return np.exp(tp[1])*(z**tp[0])
            # for cut = 10 and 20, using the last set of parameters isn't good; boost is just zero
            # if p['tail_rs'][-1] < 2999.7:
            #     return 0
            # else:
            #     tp = p['tail_params'][-1]
            #     return np.exp(tp[1])*(z**tp[0])

def EMDE_boost(cut,ns,redshift):
    ### currently for T_RH = 1 GeV
    params = []
    if cut == 10:
        params = {'B0': 38693980.73920416,
        'z_b': 245.8884493910544,
        'n': 3.932449606038306,
        'alpha': -20,
        'tail_rs': [300., 333.77596176, 371.35464216, 413.16417614, 459.68090085, 511.43478261],
        'tail_params': [(-16.730001257099335, 106.96052667483619),
        (-21.108053589338606, 132.40950191115718),
        (-26.528357117389742, 164.49196215802243),
        (-33.11549416261378, 204.18341476509863),
        (-41.440827694539976, 255.24493635670296)]}
    elif cut == 20:
        params = {'B0': 2331355808.7916265,
        'z_b': 942.2798149809815,
        'n': 4.031677715797148,
        'alpha': -20,
        'tail_rs': [1000.2, 1135.89524165, 1290., 1437.79869539, 1601.23584006, 1783.25117676, 1985.95652174],
        'tail_params': [(-12.872111446852575, 106.5444591856528),
            (-16.338448461764976, 130.92510006636215),
            (-21.19298239840551, 165.65835967914174),
            (-26.689849902363694, 205.63524606541722),
            (-33.38037934720137, 255.0135763922022),
            (-41.835760066556844, 318.3355294158349)]}
    elif cut == 30:
        params = {'B0': 31398622881.3384,
        'z_b': 2225.233300583307,
        'n': 4.131579035517818,
        'alpha': -20,
        'tail_rs': [1700.1, 1904.56210311, 2133.61379014, 2390.21232126, 2677.67061083, 2999.7],
        'tail_params': [(-5.555621908456861, 64.10417162391633),
            (-7.517936173491495, 78.92871210433614),
            (-9.995806185397171, 97.9310652142649),
            (-12.77659910180909, 119.5737042789379),
            (-16.558366438891635, 149.4172283500279)]}
    elif cut == 40:
        params = {'B0': 193971586354.535,
        'z_b': 3515.632553902913,
        'n': 4.881330532104467,
        'alpha': -20,
        'tail_rs': [2000.1, 2168.98691736, 2352.13451712, 2550.7469604, 2766.1300868, 2999.7],
        'tail_params': [(-1.1123717310878836, 34.176288489331846),
            (-1.950881487647835, 40.61271320722354),
            (-2.1434289561023143, 42.11190431923245),
            (-2.92240448344829, 48.19943233919457),
            (-4.401025467842664, 59.909049495537474)]}
    elif cut == 100: # fake decay lol (for 10)
        B0 = 38693980.73920416
        return 1+B0*((1+redshift)**(-3))
    elif cut == 200: # fake decay lol (for 20)
        B0 = 2331355808.7916265
        return 1+B0*((1+redshift)**(-3))
    elif cut == 300: # fake decay lol (for 30)
        B0 = 31398622881.3384
        return 1+B0*((1+redshift)**(-3))
    elif cut == 400: # fake decay lol (for 40)
        B0 = 193971586354.535
        return 1+B0*((1+redshift)**(-3))

    # if ns_switch == 1:
    #     a = afnc(ns)
    #     b = bfnc(ns)
    # else:
    #     a = 1
    #     b = 1

    a = afnc(ns)
    b = bfnc(ns)
    # return 1+b*get_fitted_boost(redshift/a,params)
    return 1 + (1/(1+redshift)**3)*b*get_fitted_boost(redshift/a,params)


# def EMDE_boost(cut,redshift):
#   # get interp1d object
#   f_boost = boost_interp(cut)
#   # return boost at given redshift
#   return f_boost(redshift)

def boost_factor_halos(redshift,zh,fh):
    # ret = 1 + fh*erf(redshift/(1+zh))/redshift**3
    ret = 1 + fh*erfc(redshift/(1+zh))/redshift**3
    return ret

def secondaries_from_cirelli(logEnergies,mass,primary, **DarkOptions):
    from .common import sample_spectrum
    cirelli_dir = os.path.join(data_dir, 'cirelli')
    dumpername = 'cirelli_spectrum_of_{:s}.obj'.format(primary)

    injection_history = DarkOptions.get("injection_history","annihilation")
    if "decay" in injection_history:
        equivalent_mass = mass/2.
    else:
        equivalent_mass = mass
    if equivalent_mass < 5 or equivalent_mass > 1e5:
        raise err('The spectra of Cirelli are only given in the range [5 GeV, 1e2 TeV] assuming DM annihilation. The equivalent mass for the given injection_history ({:.2g} GeV) is not in that range.'.format(equivalent_mass))

    if not hasattr(logEnergies,'__len__'):
        logEnergies = np.asarray([logEnergies])
    else:
        logEnergies = np.asarray(logEnergies)

    if not os.path.isfile( os.path.join(cirelli_dir, dumpername)):
        sys.path.insert(1,cirelli_dir)
        from spectrum_from_cirelli import get_cirelli_spectra
        masses, log10X, dNdLog10X_el, dNdLog10X_ph, dNdLog10X_oth = get_cirelli_spectra(primary)
        total_dNdLog10X = np.asarray([dNdLog10X_el, dNdLog10X_ph, dNdLog10X_oth])
        from .interpolator import NDlogInterpolator
        interpolator = NDlogInterpolator(masses, np.rollaxis(total_dNdLog10X,1), exponent = 0, scale = 'log-log')
        dump_dict = {'dNdLog10X_interpolator':interpolator, 'log10X':log10X}
        with open(os.path.join(cirelli_dir, dumpername),'wb') as dump_file:
            dill.dump(dump_dict, dump_file)
    else:
        with open(os.path.join(cirelli_dir, dumpername),'rb') as dump_file:
            dump_dict = dill.load(dump_file)
            interpolator = dump_dict.get('dNdLog10X_interpolator')
            log10X = dump_dict.get('log10X')
    del dump_dict
    temp_log10E = log10X + np.log10(equivalent_mass)*np.ones_like(log10X)
    temp_el, temp_ph, temp_oth = interpolator.__call__(equivalent_mass) / (10**temp_log10E * np.log(10))[None,:]
    ret_spectra = np.empty(shape=(3,len(logEnergies)))
    ret_spectra = sample_spectrum(temp_el, temp_ph, temp_oth, temp_log10E, mass, logEnergies, **DarkOptions)
    return ret_spectra

def secondaries_from_simple_decay(E_secondary, E_primary, primary):
    if primary not in ['muon','pi0','piCh']:
        raise err('The "simple" decay spectrum you asked for (species: {:s}) is not (yet) known.'.format(primary))

    if not hasattr(E_secondary,'__len__'):
        E_secondary = np.asarray([E_secondary])
    else:
        E_secondary = np.asarray(E_secondary)

    decay_dir  = os.path.join(data_dir, 'simple_decay_spectra')
    dumpername = 'simple_decay_spectrum_of_{:s}.obj'.format(primary)
    original_data = '{:s}_normed.dat'.format(primary)

    if not os.path.isfile( os.path.join(decay_dir, dumpername)):
        data = np.genfromtxt( os.path.join(decay_dir, original_data), unpack = True, usecols=(0,1,2,3))
        from .interpolator import NDlogInterpolator
        spec_interpolator = NDlogInterpolator(data[0,:], data[1:,:].T, exponent = 1, scale = 'lin-log')
        dump_dict = {'spec_interpolator':spec_interpolator}
        with open(os.path.join(decay_dir, dumpername),'wb') as dump_file:
            dill.dump(dump_dict, dump_file)
    else:
        with open(os.path.join(decay_dir, dumpername),'rb') as dump_file:
            dump_dict = dill.load(dump_file)
            spec_interpolator = dump_dict.get('spec_interpolator')

    x = E_secondary / E_primary
    out = spec_interpolator.__call__(x)
    out /= (np.log(10)*E_secondary)[:,None]
    return out

def luminosity_accreting_bh(Energy,recipe,PBH_mass):
    if not hasattr(Energy,'__len__'):
        Energy = np.asarray([Energy])
    if recipe=='spherical_accretion':
        a = 0.5
        Ts = 0.4*511e3
        Emin = 1
        Emax = Ts
        out = np.zeros_like(Energy)
        Emin_mask = Energy > Emin
        # Emax_mask = Ts > Energy
        out[Emin_mask] = Energy[Emin_mask]**(-a)*np.exp(-Energy[Emin_mask]/Ts)
        out[~Emin_mask] = 0.
        # out[~Emax_mask] = 0.

    elif recipe=='disk_accretion':
        a = -2.5+np.log10(PBH_mass)/3.
        Emin = (10/PBH_mass)**0.5
        # print a, Emin
        Ts = 0.4*511e3
        out = np.zeros_like(Energy)
        Emin_mask = Energy > Emin
        out[Emin_mask] = Energy[Emin_mask]**(-a)*np.exp(-Energy[Emin_mask]/Ts)
        out[~Emin_mask] = 0.
        Emax_mask = Ts > Energy
        out[~Emax_mask] = 0.
    else:
        from .__init__ import DarkAgesError as err
        raise err('I cannot understand the recipe "{0}"'.format(recipe))
    # print out, Emax_mask
    return out/Energy #We will remultiply by Energy later in the code
