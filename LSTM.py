import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    file_path = 'https://raw.githubusercontent.com/seonuk/weather-forecast/master/weather.csv'
    df = pd.read_csv(file_path)

    df['일시'] = pd.to_datetime(df['일시']).dt.strftime('%Y%m%d').astype(int)
    df.set_index('일시', inplace=True)

    result = df.iloc[-1:, ~df.columns.isin(['최저기온(°C)', '최고기온(°C)'])]
    independent= df.iloc[:, ~df.columns.isin(['최저기온(°C)', '최고기온(°C)'])]
    dependent = df.iloc[:, df.columns.isin(['최저기온(°C)', '최고기온(°C)'])]

    x_train, x_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.2, random_state=0)
    x_train = tf.expand_dims(x_train, axis=0)
    x_train = tf.expand_dims(x_train, axis=0)
    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(2))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=1, shuffle=False)

    model.evaluate(x_test, y_test)
    print(model.predict(result))
