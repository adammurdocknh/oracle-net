import numpy
# import keras
from keras.models import Sequential
from keras.layers import Dense
# import keras.models
# import keras.layers

print('done')

dataset = numpy.load('./sheets/lcs_2020_summer_player_stats_oracleselixir_scaled.npy')
# dataset = numpy.loadtxt('./sheets/lcs_2020_summer_player_stats_oracleselixir_scaled.csv',delimiter=',')

X = dataset[:,0:24]
Y = dataset[:,24]

model = Sequential()

model.add(Dense(48,input_dim=24,activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(1,activation='tanh'))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

model.fit(X,Y, epochs=15000,batch_size=1000)

# predictions = model.predict_classes(X)

# for i in range(10):
#     print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))

_, accuracy = model.evaluate(X,Y)
print('Accuracy: %.2f' % (accuracy*100))