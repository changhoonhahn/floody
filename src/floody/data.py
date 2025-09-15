'''

module to interface with different datasets 

'''
import os
import numpy as np 
import geopandas as gpd
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

        if np.sum(is_zcode) == 0: 
            print('zipcode %i has no entry' % zcode) 
            return None  # there are no zipcodes 

        return self._data[col][is_zcode]

    def columns(self): 
        return self._data.columns


class FEMA(ZipData): 
    def __init__(self): 
        '''
        '''
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


class Precip(ZipData): 
    def __init__(self): 
        ''' Precipitation data from PRISM
        '''
        self._data = _load_data('floody.prism.precip.hdf5', dtype='table') 


class Census(ZipData): 
    def __init__(self): 
        ''' ACS Census data from 2016-2020
        '''
        self._data = _load_data('floody.acs_census.hdf5', dtype='table') 


def _load_data(filename, dtype='table'): 
    ''' load data 
    '''
    if os.path.isdir('/Users/chahah/data/noah/'):  
        # local machine 
        if dtype == 'table': 
            return Table.read(os.path.join('/Users/chahah/data/noah/', filename))
        elif dtype == 'gpd': 
            return gpd.read_file(os.path.join('/Users/chahah/data/noah/', filename))
    elif os.path.isdir('/scratch/gpfs/chhahn/noah/'): 
        # della 
        if dtype == 'table': 
            return Table.read(os.path.join('/scratch/gpfs/chhahn/noah/floody/', filename))
        elif dtype == 'gpd': 
            return gpd.read_file(os.path.join('/scratch/gpfs/chhahn/noah/floody/', filename))
    else: 
        raise ValueError


def read_zipshape(zipcodes): 
    ''' select shape files for given zipcodes from following file: 
    https://catalog.data.gov/dataset/tiger-line-shapefile-2022-nation-u-s-2020-census-5-digit-zip-code-tabulation-area-zcta5
    '''
    data = _load_data('tl_2022_us_zcta520/tl_2022_us_zcta520.shp', dtype='gpd')

    # some sensible renaming of the columns 
    data = data.rename(columns={'ZCTA5CE20': 'ZIPCODE',
                                  'INTPTLAT20': 'LATITUDE',
                                  'INTPTLON20': 'LONGITUDE'})
    
    # select specifieid zipshapes  
    is_zips = np.zeros(len(data)).astype(bool)
    for z in zipcodes: 
        is_zip = np.array(data['ZIPCODE']).astype(int) == z
        if np.sum(is_zip) != 1:
            raise ValueError('Zipcode %i not list the list' % z) 
        is_zips[is_zip] = True

    return data[is_zips]


def get_X(zipcodes, latitude, longitude, year, scenario='average'): 
    ''' get compiled covariates X for specified year 
    '''
    # check inputs
    if year not in [2022, 2030, 2040, 2052]: raise ValueError 
    if scenario not in ['average', 'max']: raise ValueError

    assert len(zipcodes) == len(latitude)
    assert len(zipcodes) == len(longitude)

    prism   = Precip()
    fsf     = FloodRisk()
    acs     = Census()
    cmip    = Forecast()

    X_20XX = np.empty((len(zipcodes), 7))
    for i, zcode in enumerate(zipcodes):
        # precipitation 
        if year == 2022: 
            # maximum monthly precipitation in 2022
            X_20XX[i,0] = np.max(prism._find_zipcode(zcode, 'precip.2022'))
        else: 
            if scenario == 'average': 
                # average scenario based on the mean of the CMIP6 forecasts  

                # maximum monthly precipitation in 20XX
                _precip = []
                for month in range(1, 13):
                    _precip.append(cmip.forecast_latlon(latitude[i], longitude[i] % 360, 2022, month))
                # mean of max annual monthly preciptiation of 36 models
                max2022 = np.mean(np.max(np.array(_precip), axis=0)) 

                _precip = []
                for month in range(1, 13):
                    _precip.append(cmip.forecast_latlon(latitude[i], longitude[i] % 360, year, month))

                X_20XX[i,0] = (np.mean(np.max(np.array(_precip), axis=0)) - max2022) + np.max(prism._find_zipcode(zcode, 'precip.2022'))

            elif scenario == 'max': 
                # worst case scenario based on the max of the CMIP6 forecasts in over a 5 year period
                
                _precip = []
                for _y in range(year-4, year+1): 
                    for month in range(1, 13):
                        _precip.append(cmip.forecast_latlon(latitude[i], longitude[i] % 360, _y, month))

                X_20XX[i,0] = np.max(np.array(_precip))

        # flood risk100 
        if year == 2022: # 2022
            X_20XX[i,1] = fsf._find_zipcode(zcode, 'risk100.2022')[0]
        else: 
            # interpolate between 2022-2052
            _risk2052 = fsf._find_zipcode(zcode, 'risk100.2052')[0]
            _risk2022 = fsf._find_zipcode(zcode, 'risk100.2022')[0]
            X_20XX[i,1] = _risk2022 + (_risk2052 - _risk2022)/30. * (year - 2022)

        # median income
        X_20XX[i,2] = acs._find_zipcode(zcode, 'income')[0]

        # population
        X_20XX[i,3] = acs._find_zipcode(zcode, 'population')[0]

        # renter fraction
        X_20XX[i,4] = acs._find_zipcode(zcode, 'renter_fraction')[0]

        # educated fraction
        X_20XX[i,5] = acs._find_zipcode(zcode, 'educated_fraction')[0]

        # white fraction
        X_20XX[i,6] = acs._find_zipcode(zcode, 'white_fraction')[0]

    return X_20XX


def CityZipcodes(city):
    ''' return zipcode array given city 
    '''
    if city == 'houston': 
        zcodes = np.array([77002, 77003, 77004, 77005, 77006, 77007, 77008, 77009, 77010, 77011, 77012, 77013, 77014, 77015, 77016, 77017, 77018, 77019, 77020, 77021, 77022, 77023, 77024, 77025, 77026, 77027, 77028, 77029, 77030, 77031, 77032, 77033, 77034, 77035, 77036, 77037, 77038, 77039, 77040, 77041, 77042, 77043, 77044, 77045, 77046, 77047, 77048, 77049, 77050, 77051, 77053, 77054, 77055, 77056, 77057, 77058, 77059, 77060, 77061, 77062, 77063, 77064, 77065, 77066, 77067, 77068, 77069, 77070, 77071, 77072, 77073, 77074, 77075, 77076, 77077, 77078, 77079, 77080, 77082, 77083, 77084, 77085, 77086, 77087, 77088, 77089, 77090, 77092, 77093, 77094, 77095, 77096, 77098, 77099, 77301, 77302, 77303, 77304, 77306, 77316, 77318, 77327, 77328, 77336, 77338, 77339, 77345, 77346, 77354, 77355, 77356, 77357, 77362, 77365, 77372, 77373, 77375, 77377, 77378, 77379, 77380, 77381, 77382, 77384, 77385, 77386, 77388, 77389, 77396, 77401, 77406, 77407, 77423, 77429, 77430, 77433, 77441, 77447, 77449, 77450, 77459, 77461, 77469, 77471, 77477, 77478, 77479, 77480, 77484, 77486, 77489, 77493, 77494, 77498, 77502, 77503, 77504, 77505, 77506, 77510, 77511, 77515, 77517, 77520, 77521, 77530, 77531, 77532, 77534, 77535, 77536, 77539, 77541, 77545, 77546, 77550, 77551, 77554, 77562, 77563, 77565, 77566, 77568, 77571, 77573, 77575, 77578, 77581, 77583, 77584, 77586, 77587, 77590, 77591, 77598, 77650])
    elif city == 'capecoral': 
        zcodes = np.array([33901, 33903, 33903, 33904, 33905, 33905, 33907, 33908, 33909, 33912, 33913, 33913, 33914, 33916, 33917, 33917, 33919, 33920, 33921, 33922, 33928, 33931, 33936, 33956, 33957, 33971, 33972, 33990, 33990, 33991, 33993, 33993, 34134, 34134, 34135])
    elif city == 'chicago': 
        zcodes = np.array([60601, 60602, 60603, 60604, 60605, 60606, 60607, 60608, 60609, 60610, 60611, 60612, 60613, 60614, 60615, 60616, 60617, 60618, 60619, 60620, 60621, 60622, 60623, 60624, 60625, 60626, 60628, 60629, 60630, 60631, 60632, 60633, 60634, 60636, 60637, 60638, 60639, 60640, 60641, 60642, 60643, 60644, 60645, 60646, 60647, 60649, 60651, 60652, 60653, 60654, 60655, 60656, 60657, 60659, 60660, 60661, 60706, 60707, 60803, 60804, 60805, 60827])
    elif city == 'losangeles': 
        zcodes = np.array([90001, 90002, 90005, 90006, 90003, 90004, 90007, 90008, 90013, 90014, 90017, 90021, 90011, 90012, 90015, 90016, 90018, 90019, 90020, 90022, 90025, 90027, 90031, 90032, 90023, 90024, 90026, 90028, 90029, 90033, 90034, 90037, 90039, 90042, 90043, 90045, 90035, 90036, 90038, 90040, 90041, 90044, 90046, 90048, 90057, 90058, 90047, 90049, 90056, 90059, 90062, 90063, 90064, 90065, 90066, 90061, 90067, 90068, 90071, 90077])
    elif city == 'miami': 
        zcodes = np.array([33012, 33015, 33186, 33157, 33033, 33027, 33178, 33142, 33177, 33032, 33161, 33165, 33196, 33125, 33176, 33018, 33175, 33193, 33126, 33016, 33179, 33147, 33162, 33155, 33169, 33010, 33160, 33172, 33014, 33134, 33055, 33030, 33056, 33139, 33183, 33135, 33141, 33180, 33174, 33013, 33130, 33150, 33133, 33143, 33156, 33173, 33127, 33145, 33054, 33144, 33138, 33185, 33166, 33167, 33189, 33168, 33140, 33184, 33131, 33137, 33181, 33034, 33187, 33136, 33154, 33146, 33149, 33035, 33129, 33182, 33170, 33190, 33132, 33037, 33128, 33031, 33194, 33158, 33122, 33039, 33109])
    elif city == 'neworleans': 
        zcodes = np.array([70118, 70119, 70122, 70115, 70127, 70126, 70131, 70117, 70114,
       70125, 70124, 70128, 70130, 70129, 70116, 70113, 70112])
    elif city == 'newyorkcity': 
        zcodes = np.array([10001, 10002, 10003, 10004, 10005, 10006, 10007, 10009, 10010, 10011, 10012, 10013, 10014, 10016, 10017, 10018, 10019, 10021, 10022, 10023, 10024, 10025, 10026, 10027, 10029, 10030, 10031, 10032, 10033, 10034, 10035, 10036, 10038, 10039, 10040, 10044, 10128, 11201, 11203, 11204, 11205, 11207, 11208, 11209, 11210, 11211, 11212, 11213, 11214, 11215, 11216, 11217, 11218, 11219, 11220, 11221, 11222, 11223, 11224, 11225, 11226, 11228, 11229, 11230, 11231, 11232, 11233, 11234, 11235, 11236, 11237, 11238, 11239, 11004, 11005, 11101, 11102, 11103, 11104, 11105, 11106, 11354, 11355, 11356, 11357, 11358, 11360, 11361, 11362, 11363, 11364, 11365, 11366, 11367, 11368, 11369, 11370, 11372, 11373, 11374, 11375, 11377, 11378, 11379, 11385, 11411, 11413, 11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421, 11422, 11423, 11426, 11427, 11428, 11429, 11432, 11433, 11434, 11435, 11436, 11691, 11692, 11693, 11694, 11697, 10451, 10452, 10453, 10454, 10455, 10456, 10457, 10458, 10459, 10460, 10460, 10461, 10462, 10463, 10464, 10464, 10465, 10466, 10467, 10468, 10468, 10469, 10470, 10471, 10472, 10472, 10473, 10474, 10301, 10302, 10303, 10304, 10305, 10306, 10307, 10308, 10309, 10310, 10312, 10314])
    else: 
        raise NotImplementedError
    return zcodes
