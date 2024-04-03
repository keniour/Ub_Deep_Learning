#%%
from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
from keras.applications import VGG16

#!Preprocessing images
from keras.preprocessing.image import ImageDataGenerator

#!Setting up a data augmentation (crée de petits chgt dans les images original pour en créer de nouvelles)
train_datagen= ImageDataGenerator(
    rescale=1./255,
    rotation_range=40, #value in degrees that is a range to randomly rotate pictures
    width_shift_range=0.2, #fraction of total width to randomly translate the pictures
    height_shift_range=0.2, #same with height
    shear_range=0.2, #for randomly apply shearing transformations
    zoom_range=0.2, #for randomly zooming inside the picture
    horizontal_flip=True, #permit to flip horizontaly the pictures
    #fill_mode='nearest' #what strategy use for create new pixels that can appear after a translation or a rotation   
)
#technique qui permet d'aider en remixant les paramètres mais vu 
#qu'il n'y a pas de nouvelles "images", il n'y a pas non plus de nouvelles infos

test_datagen=ImageDataGenerator(rescale=1./255) #on n'augmente pas les images de test!!

train_dir='/home/keniour/Documents/Programmation/Dog_vs_cat/train'
train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

validation_dir='/home/keniour/Documents/Programmation/Dog_vs_cat/validation'
validation_generator=test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

#!Using a pretrained convnet
conv_base=VGG16(
    weights='imagenet',
    include_top=False, #for including or not the desely connected classifier on top of the network
    input_shape=(150,150,3)
)
conv_base.trainable = False #to freeze the parameters of the pretrained convnet

model = models.Sequential()
"""
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
"""
model.add(conv_base) #using the pretrained convnet as convolutional base
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))#! 50% de Dropout
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc']
              )


history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=50)

model.save('cats_vs_dogs_pretrained.h5')


#%%
#! Finetune the last convolution block of the pretrained model and the last one (post layers.flatten())

conv_base.trainable=True
set_trainable=False
for layer in conv_base.layers:
    if layer.name=='block5_conv1':
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else :
        layer.trainable=False

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc']
              )


history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=100,
            validation_data=validation_generator,
            validation_steps=50)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

test_dir='/home/keniour/Documents/Programmation/Dog_vs_cat/test'
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

test_loss, test_acc = model.evaluate(test_generator, steps=50)
print('test acc:', test_acc)


