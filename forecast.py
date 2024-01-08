import tensorflow as tf
import pandas as pd

if __name__ == '__main__':
    file_path = 'https://github.com/seonuk/weather-forecast/blob/master/OBS_ASOS_DD_20240107222617.csv'
    weather = pd.read_csv(file_path)

    print(weather.columns)