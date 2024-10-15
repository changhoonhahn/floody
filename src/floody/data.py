'''

module to interface with different datasets 

'''
import os
import numpy as np 
from astropy.table import Table


class ZipData(object): 
    def __init__(self): 
        ''' generic base database where ZIPCODE is one of the columns. This
        class will have a number of convenient methods for dealing with zipcode
        data.
        '''
        self._data = None 

    def _find_zipcode(self, zcode, col): 
        ''' find value corresponding to zipcode for column col 

        args: 
            zcode: int, specifying the zipcode 

            col: str, specifying the column 
        '''
        is_zcode = (self._data['ZIPCODE'] == zcode) 

        if np.sum(is_zcode) == 0: return None  # there are no zipcodes 

        return np.array(self._data[col][is_zcode]) 

    def columns(self): 
        return self._data.columns() 


class FEMA(ZipData): 
    '''
    '''
    def __init__(self): 
        # read data set 
        self._data = _load_data('floody.fema.compiled.hdf5')

    def prepare_train_test(self, sample, split=0.9, seed=42): 
        ''' prepare training and testing datasets (numpy arrays) for CausalFlow

        args: 
            sample: str, specifying whether 'treated' or 'control' sample. 


        kwargs: 
            split: float between 0-1 specifying the train/test split. 
                (Default: 0.9) 
            seed: random seed for shuffling the data. Order will be preserved
                for same seed.

        return: 
            train_data, test_data: np.array, specifying the training and
                testing data. 

        '''
        # data will be read in this column order 
        columns = ['paid_claim_per_policy', # claim paid per policy 
                   'mean_monthly_rainfall', # mean monthly rainfall 
                   'flood_risk100', #'flood_risk_avg_score' we're using 'flood risk 100 as a proxy for flood risk average score 
                   'median_household_income',
                   'population', 
                   'renter_fraction', 
                   'educated_fraction',
                   'white_fraction'] 

        treated = self._data['CRS_TREAT'] 
        if sample == 'treated':  # treated sample
            keep = treated
        elif sample == 'control':  # control sample 
            keep = ~treated
        else: raise ValueError

        data = np.array([self._data[col] for col in columns]).T[keep]

        # shuffle dataset 
        ishfl = np.arange(data.shape[0])
        np.random.seed(seed) 
        np.random.shuffle(ishfl)

        # train data split 
        Ntrain = int(split * float(data.shape[0]))

        return data[ishfl][:Ntrain], data[ishfl][Ntrain:]


class Forecast(ZipData): 
    def __init__(self): 
        ''' precipitation forecasts from CMIP6 models 
        '''
        self._data = _load_data('floody.cmip6.precip_forecast.hdf5')

        # latitude grid 
        self.lat_grid = np.linspace(-89.25, 89.25, 120)
        self.dlat = 1.5

        # longitude_grid
        self.lon_grid = np.linspace(0, 358.5, 240)
        self.dlon = 1.5

    def forecast_latlon(self, lat, lon, year, month): 
        ''' Given latitude and longitude, interpolate precipitation forecast
        using triangle-space cloud method for given year and month. 

        args:
            lat: float, latitude 
            lon: float, longitude
            year: int, year 
            month: int, month 
        '''
        if lat < self.lat_grid[0] or lat > self.lat_grid[-1]: raise ValueError('latitude out of range') 
        if lon < self.lon_grid[0] or lon > self.lon_grid[-1]: raise ValueError('longitude out of range') 

        i_lat = np.digitize(lat, self.lat_grid) # get latitude bins
        i_lon = np.digitize(lon, self.lon_grid) # get longitude bins

        # specified precipitation 
        precip = self._data['precip.%i.%i' % (year, month)] 

        # preciptation predicted by the models interpolated using TSC
        predict = 1./(self.dlat * self.dlon) * (
                (precip[240*i_lat + i_lon] * (self.lat_grid[i_lat] - lat) * (self.lon_grid[i_lon] - lon)) + 
                (precip[240*i_lat + i_lon - 1] * (self.lat_grid[i_lat] - lat) * (lon - self.lon_grid[i_lon-1])) + 
                (precip[240*(i_lat-1) + i_lon] * (lat - self.lat_grid[i_lat-1]) * (self.lon_grid[i_lon] - lon)) + 
                (precip[240*(i_lat-1) + i_lon - 1] * (lat - self.lat_grid[i_lat-1]) * (lon - self.lon_grid[i_lon-1]))
                ) 
        return predict 


class FloodRisk(ZipData): 
    def __init__(self): 
        ''' Flood risk data compiled from the Floodrisk
        '''
        self._data = _load_data('floody.fsf.floodrisk_forecast.hdf5')


def _load_data(filename): 
    ''' load data 
    '''
    if os.path.isdir('/Users/chahah/data/noah/'):  
        # local machine 
        return Table.read(os.path.join('/Users/chahah/data/noah/', filename))
    elif os.path.isdir('/scratch/gpfs/chhahn/noah/'): 
        # della 
        return Table.read(os.path.join('/scratch/gpfs/chhahn/noah/', filename))
    else: 
        raise ValueError


