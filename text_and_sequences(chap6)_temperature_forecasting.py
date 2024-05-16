#%%
import os 
import numpy as np

data_dir='/home/keniour/Téléchargements/jena_climate'
fname=os.path.join(data_dir,'jena_climate_2009_2016.csv')

f=open(fname)
data=f.read()
f.close()

lines=data.split('\n')
header=lines[0].split(',')
lines=lines[1:]

float_data = np.zeros((len(lines),len(header)-1))

for i,line in enumerate(lines):
    values=[float(x) for x in line.split(',')[1:]]
    float_data[i,:]=values

#!Normalizing the data (les degrés, les pressions, etc ne sont pas sur la meme échelle)
mean=float_data[:200000].mean(axis=0)
float_data-=mean
std=float_data[:200000].std(axis=0)
float_data/=std

#!generator yielding timeseries samples and their targets

def generator(data,lookback,delay,min_index,max_index,shuffle=False,
              batch_size=128,step=6):
    if max_index is None:
        max_index=len(data)-delay-1
    i=min_index+lookback
    
    while 1:
        if shuffle:
            rows=np.random.randint(
                min_index+lookback,max_index,size=batch_size
            )
        else :
            if i +batch_size>=max_index:
                i=min_index + lookback
            rows=np.arange(i,min(i+batch_size,max_index))
            i+=len(rows)
    
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
    
#!Preparing the training, validation and test generators

lookback=1440
step=6
delay=144
batch_size=128

train_gen=generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=0,
                    max_index=200000,
                    shuffle=True,
                    step=step,
                    batch_size=batch_size   
)

val_gen=generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    shuffle=True,
                    step=step,
                    batch_size=batch_size   
)
test_gen=generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=300000,
                    max_index=None,
                    shuffle=True,
                    step=step,
                    batch_size=batch_size   
)

val_steps= (300000-200001-lookback)

test_steps=(len(float_data)-300001-lookback)

#!computing the common-sense baseline MAE

def evaluate_naive_method():
    batch_maes=[]
    for step in range(val_steps):
        samples,targets=next(val_gen)
        preds=samples[:,-1,1]
        mae=np.mean(np.abs(preds-targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))
    
evaluate_naive_method()


# %%
#!Training and evaluationg a GRU-based model
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2, #dropout-regularized GRU-based model
                     recurrent_dropout=0.2,
                     input_shape=(None,float_data.shape[-1])))
model.add(layers.GRU(64,activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(),loss='mae')
history=model.fit_generator(train_gen,
                            steps_per_epoch=500,
                            epochs=20,
                            validation_data=val_gen,
                            validation_steps=val_steps)


#%%Training and evaluatin a bidirectional LSTM

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Bidirectional(
    layers.GRU(32), input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                                steps_per_epoch=500,
                                epochs=40,
                                validation_data=val_gen,
                                validation_steps=val_steps)



