'''

module for main calculations 


'''
import numpy as np 
import warnings

import torch

from causalflow import causalflow
from causalflow import support as Support


def flood_loss(mon_rain, flood_risk100, income, population, rent_frac, edu_frac, white_frac, 
        Nsample=10000, support_threshold=0.95, device=None): 
    """ calculate flood losses given covariates: 'mean_monthly_rainfall', 'flood_risk100', 
    'median_household_income', 'population', 'renter_fraction', 'educated_fraction', 'white_fraction'

    """
    X = np.array([mon_rain, 
        flood_risk100, 
        np.log10(income), # change to log10 --- this is what causalflow is trained on 
        np.log10(population), # change to log10 --- this is what causalflow is trained on 
        rent_frac, 
        edu_frac, 
        white_frac])
        
    # declare Scenario A CausalFlow
    Cflow = causalflow.CausalFlowA(device=device)

    # read control flows (currently hardcoded) 
    flows = Cflow._load_flows_optuna('flow.control', '/scratch/gpfs/chhahn/noah/floody/flow/')
    Nflow = len(flows) 

    # read and load control support (currently hardcoded) 
    Supp = Support.Support()
    Supp.load_optuna('support.control', '/scratch/gpfs/chhahn/noah/floody/support/', 
            verbose=False)
    
    in_support = Supp.check_support(X, Nsample=Nsample, threshold=support_threshold)[0] 
    if not in_support: warnings.warn("X is out of support")
    
    # sample the flows at X 
    y_samp = []
    for flow in flows:
        _samp = flow.sample((int(Nsample/Nflow),),
                x=torch.tensor(X, dtype=torch.float32).to(Cflow.device),
                show_progress_bars=False)
        y_samp.append(_samp.detach().cpu().numpy())
    y_samp = np.concatenate(y_samp)
    
    return 10**y_samp


def flood_saving(mon_rain, flood_risk100, income, population, rent_frac, edu_frac, white_frac, 
        Nsample=10000, Nsupport=10000, support_threshold=0.95, device=None): 
    """ calculate flood losses given covariates: 'mean_monthly_rainfall', 'flood_risk100', 
    'median_household_income', 'population', 'renter_fraction', 'educated_fraction', 'white_fraction'

    """
    X = np.array([mon_rain, 
        flood_risk100, 
        np.log10(income), # change to log10 --- this is what causalflow is trained on 
        np.log10(population), # change to log10 --- this is what causalflow is trained on 
        rent_frac, 
        edu_frac, 
        white_frac])
        
    # declare Scenario A CausalFlow
    Cflow = causalflow.CausalFlowA(device=device)

    # read flows (currently hardcoded) 
    Cflow.load_flows_optuna_control('flow.control', '/scratch/gpfs/chhahn/noah/floody/flow/')
    Cflow.load_flows_optuna_treated('flow.treated', '/scratch/gpfs/chhahn/noah/floody/flow/')

    # read and load support (currently hardcoded) 
    Supp = Support.Support()
    Supp.load_optuna('support.control', '/scratch/gpfs/chhahn/noah/floody/support/')
    Cflow.support_control = Supp 

    Supp = Support.Support()
    Supp.load_optuna('support.control', '/scratch/gpfs/chhahn/noah/floody/support/')
    Cflow.support_treated = Supp 

    cate = Cflow.CATE(X, Nsample=Nsample, Nsupport=Nsupport, 
            support_threshold=support_threshold, progress_bar=False, transf=lambda x: 10**x)
    return cate 
