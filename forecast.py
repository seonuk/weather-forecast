import tensorflow as tf
import pandas as pd

if __name__ == '__main__':
    file_path = 'https://raw.githubusercontent.com/seonuk/weather-forecast/master/OBS_ASOS_DD_20240107222617.csv'
    df = pd.read_csv(file_path)

    df['일시'] = pd.to_datetime(df['일시']).dt.strftime('%Y%m%d').astype(int)

    independent= df.iloc[:, ~df.columns.isin(['최저기온(°C)', '최고기온(°C)', '일시'])]
    dependent = df.iloc[:, df.columns.isin(['최저기온(°C)', '최고기온(°C)'])]
    print(independent)
    print(dependent)
    # independent = df[['최저기온']]
    # dependent = df[['최저기온']]
    X = tf.keras.layers.Input(shape=[3])
    Y = tf.keras.layers.Dense(2)(X)
    model = tf.keras.models.Model(X,Y)
    model.compile(loss='mse')

    model.fit(independent, dependent, epochs=10, verbose=0)

    print(model.predict['independent'])
    # print(model.predict([[15]]))
    # independent = weather_csv[]