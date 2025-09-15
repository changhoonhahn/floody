#!/bin/python 
import argparse
import functools
import os
import time

import numpy as np
import torch
from astropy.table import Table

from floody import data as D
from floody import floody as F


def zipcodes(city): 
    ''' return zipcode array given city 
    '''
    return D.CityZipcodes(city)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("city", type=str, help="city")
    parser.add_argument("-t", "--tag", help="tag calculation", type=str, default='v0')
    parser.add_argument("-n", "--ncf", help="number of causalflow runs", type=int, default=11)
    parser.add_argument("-s", "--scenario", help='forecast scenario', type=str, default='average')
    args = parser.parse_args()
    
    # get zipcodes corresponding to city 
    zcodes = zipcodes(args.city)

    # get zipshape object 
    city = D.read_zipshape(zcodes)


    # get forecast and community characteristics at 2022, 2030, 2040, 2052
    X_2022 = D.get_X(
            np.array(city['ZIPCODE']).astype(int), # zipcodes
            np.array(city['LATITUDE']).astype(float), # latitude
            np.array(city['LONGITUDE']).astype(float), 2022)
    X_2052 = D.get_X(
            np.array(city['ZIPCODE']).astype(int), 
            np.array(city['LATITUDE']).astype(float),
            np.array(city['LONGITUDE']).astype(float), 2052, 
            scenario=args.scenario)

    # calculate flood losses
    median_losses = np.zeros((2, X_2022.shape[0]))
    sig_median_losses = np.zeros((2, X_2022.shape[0]))

    for i, zcode in enumerate(np.array(city['ZIPCODE']).astype(int)): 
        for j, _X in enumerate([X_2022, X_2052]):
            
            _med_losses = [] 
            for _i in range(args.ncf): 
                _losses = F.flood_loss(_X[i,0], _X[i,1], _X[i,2], _X[i,3], _X[i,4], _X[i,5], _X[i,6],
                                  Nsample=10000, support_threshold=0.99, device=None)
                _med_losses.append(np.median(_losses))
                
            median_losses[j, i] = np.median(_med_losses)
            sig_median_losses[j, i] = np.std(_med_losses)

    # calculate CRS savings
    savings = np.zeros((2, X_2022.shape[0]))
    sig_savings = np.zeros((2, X_2022.shape[0]))
    
    for i, zcode in enumerate(np.array(city['ZIPCODE']).astype(int)):
        for j, _X in enumerate([X_2022, X_2052]):

            _savings = []
            for _i in range(args.ncf):
                _saving = F.flood_saving(_X[i,0], _X[i,1], _X[i,2], _X[i,3], _X[i,4], _X[i,5], _X[i,6],
                                  Nsample=10000, support_threshold=0.99, device=None)
                _savings.append(_saving)

            savings[j, i] = np.median(_savings)
            sig_savings[j, i] = np.std(_savings)

    # compiel output 
    cols = ['precip', 'flood_risk100', 'income', 'population', 'renter_fraction', 'educated_fraction', 'white_fraction'] 
    
    output = Table() 
    for year, _X in zip([2022, 2052], [X_2022, X_2052]): 
        for i, col in enumerate(cols): 
            output['%s.%i' % (col, year)] = _X[:,i]

    for i, year in enumerate([2022, 2052]): 
        output['flood_loss.%i' % year] = median_losses[i,:]
        output['sig_flood_loss.%i' % year] = sig_median_losses[i,:]

        output['crs_saving.%i' % year] = savings[i,:]
        output['sig_crs_saving.%i' % year] = sig_savings[i,:]

    output.write(os.path.join('/scratch/gpfs/chhahn/noah/floody/', "floody.%s.%s.hdf5" % (args.city, args.tag)), 
            format='hdf5')
