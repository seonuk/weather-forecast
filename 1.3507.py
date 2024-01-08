import pandas as pd
import tensorflow as tf

file_path = 'https://raw.githubusercontent.com/seonuk/weather-forecast/master/weather.csv'
df = pd.read_csv(file_path)

df['일시'] = pd.to_datetime(df['일시']).dt.strftime('%Y%m%d').astype(int)

independent= df.iloc[:, ~df.columns.isin(['최저기온(°C)', '최고기온(°C)', '일시'])]
dependent = df.iloc[:, df.columns.isin(['최저기온(°C)', '최고기온(°C)'])]
print(independent)
print(dependent)
X = tf.keras.layers.Input(shape=[independent.shape[1]])
H = tf.keras.layers.Dense(24, activation='relu')(X)
H = tf.keras.layers.Dense(13, activation='relu')(X)

Y = tf.keras.layers.Dense(dependent.shape[1])(H)
model = tf.keras.models.Model(X,Y)

model.compile(loss='mse')

model.fit(independent, dependent, epochs=1000)

print(model.predict(independent[0:1]))