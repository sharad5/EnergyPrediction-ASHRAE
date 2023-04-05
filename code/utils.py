# !/usr/bin/env python

from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import pandas as pd
import numpy as np
import os 

from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime

# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
def optimize_df_mem_usage(data, use_float16=True):
    """Optimizes memory usage by casting unecessarily large data types to 
    smaller data types"""
    
    start_mem = data.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in data.columns:
        if is_datetime(data[col]) or is_categorical_dtype(data[col]):
            continue
        col_type = data[col].dtype
        
        if col_type in [np.dtype('<M8[ns]'), np.dtype('>M8[ns]'), np.dtype('datetime64')]:
            continue
        elif col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)
        else:
            data[col] = data[col].astype('category')

    end_mem = data.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.2f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return data

def weather_data_parser(weather_data):
    """Parses, fills in missing values, and updates weather data"""
    
    time_format = '%Y-%m-%d %H:%M:%S'
    start_date = datetime.datetime.strptime(weather_data['timestamp'].min(), time_format)
    end_date = datetime.datetime.strptime(weather_data['timestamp'].max(), time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    for site_id in range(16):
        site_hours = np.array(weather_data[weather_data['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list, site_hours), columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_data = pd.concat([weather_data, new_rows], sort=True)
        weather_data = weather_data.reset_index(drop=True)           

    weather_data['datetime'] = pd.to_datetime(weather_data['timestamp'])
    weather_data['day'] = weather_data['datetime'].dt.day
    weather_data['week'] = weather_data['datetime'].dt.isocalendar().week
    weather_data['month'] = weather_data['datetime'].dt.month

    weather_data = weather_data.set_index(['site_id', 'day', 'month'])

    air_temperature_filler = pd.DataFrame(weather_data.groupby(['site_id','day','month'])['air_temperature'].median(), columns=['air_temperature'])
    weather_data.update(air_temperature_filler, overwrite=False)

    cloud_coverage_filler = weather_data.groupby(['site_id', 'day', 'month'])['cloud_coverage'].median()
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'), columns=['cloud_coverage'])

    weather_data.update(cloud_coverage_filler, overwrite=False)

    due_temperature_filler = pd.DataFrame(weather_data.groupby(['site_id','day','month'])['dew_temperature'].median(), columns=['dew_temperature'])
    weather_data.update(due_temperature_filler, overwrite=False)

    sea_level_filler = weather_data.groupby(['site_id','day','month'])['sea_level_pressure'].median()
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'), columns=['sea_level_pressure'])

    weather_data.update(sea_level_filler, overwrite=False)

    wind_direction_filler =  pd.DataFrame(weather_data.groupby(['site_id','day','month'])['wind_direction'].median(), columns=['wind_direction'])
    weather_data.update(wind_direction_filler, overwrite=False)

    wind_speed_filler =  pd.DataFrame(weather_data.groupby(['site_id','day','month'])['wind_speed'].median(), columns=['wind_speed'])
    weather_data.update(wind_speed_filler, overwrite=False)

    precip_depth_filler = weather_data.groupby(['site_id','day','month'])['precip_depth_1_hr'].median()
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'), columns=['precip_depth_1_hr'])

    weather_data.update(precip_depth_filler, overwrite=False)

    weather_data = weather_data.reset_index()
    weather_data = weather_data.drop(['datetime','day','week','month'], axis=1)

    return weather_data

def load_data(path: str) -> dict:
    """Loads data and returns data in the form of a tuple. Also 
    performs basic feature engineering 
    
    Arguments
    path: string representing the directory where the data is stored
    
    Returns
    dict: dictionary containing the data
    
    """
    # Load form files 
    building = pd.read_csv(os.path.join(path, 'building_metadata.csv'))
    building = optimize_df_mem_usage(building)    
    
    weather_train = pd.read_csv(os.path.join(path, 'weather_train.csv'))
    weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'], infer_datetime_format = True, utc = True).astype('datetime64[ns]')
    weather_train = optimize_df_mem_usage(weather_train)
    
    weather_test = pd.read_csv(os.path.join(path, 'weather_test.csv'))
    weather_test = optimize_df_mem_usage(weather_test)
    
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    train['timestamp'] = pd.to_datetime(train['timestamp'], infer_datetime_format = True, utc = True).astype('datetime64[ns]')
    train = optimize_df_mem_usage(train)
    
    test = pd.read_csv(os.path.join(path, 'test.csv'))
    test = optimize_df_mem_usage(test)
    
    # Remove noisy outliers from the data
    train = train[~((train['building_id'] <= 104) & (train['meter'] == 0) & (train['timestamp'] <= "2016-05-20"))] #pd.Timestamp(2016, 5, 20)
    train = train[train["building_id"] != 1099]
    
    train = train.merge(building, on='building_id', how='left')
    train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')
    
    # Turn site zero data into the correct form 
    # https://www.kaggle.com/c/ashrae-energy-prediction/discussion/119261
    train[(train['site_id'] == 0) & (train['meter'] == 0)]['meter_reading'] = 0.2931 * train[(train['site_id'] == 0) & (train['meter'] == 0)]['meter_reading']
    
    # Make these features approximately gaussian
    train["log_meter_reading"] = np.log1p(train["meter_reading"])
    train['log_square_feet'] =  np.log1p(train['square_feet'])
    
    train['weekday'] = train['timestamp'].dt.weekday
    train['hour'] = train['timestamp'].dt.hour
    train['day'] = train['timestamp'].dt.day
    train['weekend'] = train["timestamp"].dt.weekday.isin([5,6]).astype(int)
    train['month'] = train['timestamp'].dt.month
    
    #
    
    primary_use_enc = TargetEncoder(cols=["primary_use"]).fit(train["primary_use"], train["log_meter_reading"])
    train["primary_use_enc"] = primary_use_enc.transform(train["primary_use"])
    
    return {
        "weather_test": weather_test,
        "X_train": train.drop(columns=['meter_reading', 'log_meter_reading']), 
        "X_test": test, # .drop(columns=['meter_reading', 'log_meter_reading'])
        "y_train": train['log_meter_reading'],
#         "y_test": test['log_meter_reading'],
    }

def get_train_val_split(train, ratio: float = 0.3, random_state: int = 5) -> tuple:
    """Returns the training and validation set from all the training data 
    
    Arguments
    train: dataframe of training data
    ratio: float in the range [0, 1] representing the proportion of training to testing data
    random_state: integer or the random state 
    
    Returns 
    train, test: tuple of the training and testing data
    """
    
    TEST_SIZE = 0.3
    building_meter_df = train.groupby(["building_id", "meter"])\
                            .agg(avg_meter_reading=("log_meter_reading", np.mean))\
                            .drop("avg_meter_reading", axis=1).reset_index()
    building_meter_train_df, building_meter_test_df = train_test_split(building_meter_df, test_size=TEST_SIZE, random_state=random_state)
    
    train_df = train.merge(building_meter_train_df, on=["building_id","meter"], how="inner")
    val_df = train.merge(building_meter_test_df, on=["building_id","meter"], how="inner")
    
    return train_df, val_df 
    