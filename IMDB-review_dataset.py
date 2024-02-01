#%% 3.4 : Classifying movie
#nvidia-smi -l 5
import numpy as np
from keras.datasets import imdb

(train_data, train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)



"""préparer les données pour en faire une liste de tensors de mm taille
encoding the integer sequences into a binary matric
Le but la étant de transformer les listes de chaque review en un tenseur de 10.000
"0" ou "1" si le mot est présent ou non 
"""

def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i][sequence]=1.
    return results

x_train = vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)

#vectorisation des labels
y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')



#%% Building the network
from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))



#Compiling the model with an optimizer and a loss function

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


#Validating the approach
x_val=x_train[:10000]
partial_x_train=x_train[10000:]
y_val=y_train[:10000]
partial_y_train=y_train[10000:]


history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=20,
                  batch_size=512,
                  validation_data=(x_val,y_val))

#%%plotting the training and validation loss
import matplotlib.pyplot as plt

history_dict=history.history
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']
acc=history_dict['accuracy']

epochs=range(1,len(acc)+1)

plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


#%%plotting the training and validation accuracy

plt.clf()
acc_values=history_dict['accuracy']
val_acc_values=history_dict["val_accuracy"]

plt.plot(epochs, acc_values, 'bo',label='Training acc')
plt.plot(epochs,val_acc_values,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show

""" 
We can tell an issue of overfitting, after the 4th epochs the validation loss is 
skyrocketting up and his accuracy is going down 
Our model is overtrained!!
We need to try again with less epochs
"""

#%%


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=4,batch_size=512)
results=model.evaluate(x_test,y_test)



#%%Further experiments
""" 
You used two hidden layers. Try using one or three hidden layers, and see how
doing so affects validation and test accuracy.
 Try using layers with more hidden units or fewer hidden units: 32 units, 64 units,
and so on.
 Try using the mse loss function instead of binary_crossentropy.
 Try using the tanh activation (an activation that was popular in the early days of
neural networks) instead of relu
"""

from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))



#Compiling the model with an optimizer and a loss function

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


#Validating the approach
x_val=x_train[:10000]
partial_x_train=x_train[10000:]
y_val=y_train[:10000]
partial_y_train=y_train[10000:]


history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=20,
                  batch_size=512,
                  validation_data=(x_val,y_val))
results=model.evaluate(x_test,y_test)
print(results)



#lotting the training and validation loss
import matplotlib.pyplot as plt

history_dict=history.history
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']
acc=history_dict['accuracy']

epochs=range(1,len(acc)+1)

plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


#plotting the training and validation accuracy

plt.clf()
acc_values=history_dict['accuracy']
val_acc_values=history_dict["val_accuracy"]

plt.plot(epochs, acc_values, 'bo',label='Training acc')
plt.plot(epochs,val_acc_values,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show



# %%Reverse review
word_index = imdb.get_word_index()
reverse_word_index=dict(
    [(value,key)for (key,value)in word_index.items()])
decoded_review=' '.join(
    [reverse_word_index.get(i-3,'?')for i in train_data[0]]
)
# %%
