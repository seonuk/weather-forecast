import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    file_path = 'https://raw.githubusercontent.com/seonuk/weather-forecast/master/tomorrow.csv'

    df = pd.read_csv(file_path)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    df['일시'] = pd.to_datetime(df['일시']).dt.strftime('%Y%m%d').astype(int)

    result = df.iloc[-1:, ~df.columns.isin(['내일최고온도', '내일최저온도'])]
    independent= df.iloc[:-1, ~df.columns.isin(['내일최고온도', '내일최저온도'])]
    dependent = df.iloc[:-1, df.columns.isin(['내일최고온도', '내일최저온도'])]

    x_train, x_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.2, random_state=0)
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)

    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    print(x_test)

    print(y_train.shape)
    print(y_test.shape)


    X = tf.keras.layers.Input(shape=[x_train.shape[1]])
    H = tf.keras.layers.Dense(24, activation='relu')(X)
    H = tf.keras.layers.BatchNormalization()(H)
    H = tf.keras.layers.Activation('relu')(H)

    H = tf.keras.layers.Dense(13, activation='relu')(H)
    H = tf.keras.layers.BatchNormalization()(H)
    H = tf.keras.layers.Activation('relu')(H)

    H = H = tf.keras.layers.Dense(8, activation='relu')(H)
    H = tf.keras.layers.BatchNormalization()(H)
    H = tf.keras.layers.Activation('relu')(H)

    Y = tf.keras.layers.Dense(y_train.shape[1])(H)
    model = tf.keras.models.Model(X,Y)

    model.compile(loss='mse', optimizer='adam')
    model.fit(independent, dependent, epochs=1000)

    model.evaluate(x_test, y_test)
    print(result)
    print(model.predict(result))