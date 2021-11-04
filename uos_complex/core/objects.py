from re import I
import numpy as np
import networkx as nx
import numba
import pandas as pd
import geopandas as gpd
from io import StringIO
import os, sys

def time_dict_format(**fields_timeformat_args): 
    '''let string format (YY)YYMMDDhhmmss to convert time string to `int`.

    example1)
    
    |   year    |   date    |   time    |
    +-----------+-----------+-----------+
    |   2014    |   0502    |   145212  |
    |   2014    |   0502    |   145212  |
    |   2014    |   0502    |   145212  |
    +-----------+-----------+-----------+
    
    >>> praw = RawData('population')
    >>> praw.add_file('population.csv')
    >>> praw.head()
    |   year    |   date    |   time    |
    +-----------+-----------+-----------+
    |   2014    |   0502    |   145212  |
    |   2014    |   0502    |   145212  |
    |   2014    |   0502    |   145212  |
    +-----------+-----------+-----------+
    >>> praw.set_timeformat(year='YYYY', date = 'MMDD', time = 'hhmmss')
    time format specified.
    >>>
    >>>

    example2)
    
    |           date         |
    +------------------------+
    |   20140502 14:22:23    |
    |   20140502 14:22:23    |
    |   20140502 14:22:23    |
    +------------------------+
    
    >>> praw = RawData('population')
    >>> praw.add_file('population.csv')
    >>> praw.head()
    |           date         |
    +------------------------+
    |   20140502 14:22:23    |
    |   20140502 14:22:23    |
    |   20140502 14:22:23    |
    +------------------------+
    >>> praw.set_timeformat(date = 'YYYYMMDD hh:mm:ss')
    time format specified.
    >>>
    >>>
    '''
    pass


class UCRawfiles:
    '''RAW file tracer and checker and controllor.
    This library will check header and folder of RAW files and its validity.'''
    def __init__(self,file = None):
        self._files = []
        self.ext = os.path.splitext(file)[1]
        if file is not None:
            if os.path.isfile(file):
                self._files.append(file)
    
    def add_file(self,*files):
        for file in files:
            if os.path.isfile(file):
                self._files.append(file)
        
    def add_folder(self, foldername):
        if os.path.isdir(foldername):
            for file in os.listdir(foldername):
                if self.ext == os.path.splitext(file)[1]:
                    self._files.append(file)

    def show(self):
        print(self._files)

    def is_empty(self):
        return not bool(self._files)

class UCRawData():
    '''class for taking processing or converting rawdata.
    So, User have to seperate headers into data and geotime format.
    and set path of RAW files or folders.
    
    Parameters
    -------------
    reader : `function`
        the function which takes filename as argument and return pd.DataFrame
    
    '''

    #properties
    _metadata =["name","_files","__points","__origin","__destination","__data_columns", "_mapping", "__metadata", '_reader']

    def __init__(self, name, sample_file, reader = pd.read_csv):
        self.name = name
        self.__point = {}
        self.__origin = {}
        self.__destination = {}
        self._files = UCRawfiles(sample_file)
        self.__data_columns = []
        self._mapping = {}
        self.__geometry = None
        self._reader = reader
        self._df = reader(sample_file)
        self._pre_processer = None

        
    @property
    def pandas(self):        return self._df
    @pandas.getter
    def pandas(self):        return self._df.copy()

    def __repr__(self):
        return str(self.show_data_columns())

    def exclude(self, *arg):
        '''exclude specified data columns. those data columns will be ignored.'''
        for i in arg:
            if not i in self.pandas.columns:
                raise ValueError(f'{i} is not in Data Columns.')

        if not self.__data_columns:
            for i in self.pandas.columns:
                if not i in arg:
                    self.__data_columns.append()
            

    def include(self, *arg):
        '''include only specified data columns. else columns will be ignored.'''
        pass
    
    def set_pre_processor(self, pre_processor):
        pre_processor(self.pandas)
        self._pre_processer = pre_processor

    @property
    def columns(self):
        return 
    @columns.getter
    def columns(self):
        cols = []
        for i in self._df.columns:
            if i in self._mapping:
                cols.append(self._mapping[i])
            else:
                cols.append(i)
        return cols

    @columns.setter
    def columns(self, value):
        if len(value) != len(self.columns):
            raise ValueError("columns must have the same length with RAW data.")
        
        for i,v in zip(self._df.columns, value):
            self._mapping[i] = v

        

    def show_data_columns(self):
        if self.__data_columns:

            cols = []
            for i in self._df.columns:
                if i in self._mapping:
                    cols.append(self._mapping[i])
                else:
                    cols.append(i)
            pandas = self.pandas.set_axis(cols, axis = 1, inplace=False)
            return pandas[self.__data_columns]
        else:
            return self.pandas
    
    def show_files(self):
        self._files.show()

    def add_files(self, *filenames):
        self._files.add_files(*filenames)

    def add_folder(self, foldername):
        self._files.add_folder(foldername)

    def _check_point(self, geometry, time_dict):
        if isinstance(geometry, list):
            for geo in geometry:
                if not geo in self.columns:
                    raise ValueError(f"'{geo}' is not in dataset.")
        if isinstance(geometry, str):
            if not geometry in self.columns:
                raise ValueError(f"'{geometry}' is not in dataset.")
            
        


    def set_origin(self, geo_list, time_dict):
        """set time-geometry columns for Raw geodata.

        Parameters
        ----------
        geo_list : `list`
            It can be two types for specifying geometry, 1) (geo_code, metadata) geo_code is usually single code for geometry data. 2) ((coordinate_x, coordinate_y), name of coordinate system)
        time_dict : `dict`
            {column_name : timeformat} key value pair dictionary.
        """

    
    def set_destination(self, geometry, time_dict = None):
        if not self.__origin:
            raise ValueError('Please set origin first.')

    def processing_data(self, path):
        pass
    
## metadata => gdf

class Metadata():
    @property
    def coordinate(self):
        return self._coordinate

    def check(self, iterable_target):
        raise NotImplemented

class UCgeodict(Metadata): 
    def __init__(self, geo_dict, geomtry_type = 'point'):
        self._geo_dict = geo_dict
        self._geomtry_type = geomtry_type

    def __repr__(self):
        return self._gdf

    @property
    def gdf(self):
        return self._gdf
    
    @gdf.getter
    def gdf(self):
        return self._gdf

    def check(self, iterable_target):
        errors = []
        total_number = len(iterable_target)
        for target in iterable_target:
            if not target in self._geo_dict:
                errors.append(target)
        if errors:
            print(f'Test failed.\n Detail : \n\tTotal number of targets : {total_number}\n\tThe number of Fail to check : {len(errors)}\nThe Failed targets will be returned.')
            return errors
        else:
            return True



class UCDataset:
    '''The prototype class of csns dataset. It is processed version of RAW data.
    It is designed as an implementation of data collect and selector for various methods.'''
    def __init__(self, datatype):
        self._datatype = datatype

    @property
    def datafields(self):
        pass

    def datatype(self):
        return self._datatype
    datatype = property(datatype)
    
    
    @datatype.getter
    def datatype(self):
        return self._datatype

    def to_hdf5(self, path, dtype):
        raise NotImplemented

    @classmethod
    def from_hdf5(cls, path):
        raise NotImplemented
    
    @classmethod
    def from_populatoin(cls, path):
        raise NotImplemented

    @classmethod
    def from_ODpair(cls, path):
        raise NotImplemented

        
    

class UCGeoData(UCDataset):
    @property
    def geo_dict(self):
        return self._geometry_dict

    @property
    def coordinate(self):
        raise NotImplemented

    def coordinate_conversion(self, coord):
        raise NotImplemented


class UCNetData(UCGeoData):
    
    @property
    def network(self):
        pass

    @property
    def threshold(self):
        pass
    

class UCODPairData(UCGeoData):  # (time, geom1 -> geom2) : data (m10, m20, m30,......) 
    pass

class UCSingleData(UCGeoData):  # (time, geom) : data (vel)
    pass

class UCSeqData(UCGeoData):  # (time1, geom1) : data1 'id', (time2, geom2): data2 'id'
    pass



class gObject:
    pass

