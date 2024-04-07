#%%Vizualizing intermediate activations
#!Model sans certaines caractéristiques (comme le Dropout)
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models

model=load_model('cats_vs_dogs_V1.h5')
model.summary()

img_path='/home/keniour/Documents/Programmation/Dog_vs_cat/test/cats/cat.1503.jpg'
img=image.load_img(img_path,target_size=(150,150))
img_tensor=image.img_to_array(img)
img_tensor=np.expand_dims(img_tensor,axis=0)
img_tensor/=255.

plt.imshow(img_tensor[0])
plt.show()

layer_outputs=[layer.output for layer in model.layers[:8]] #extracts the outputs of the top eight layers
activation_model=models.Model(inputs=model.input, outputs=layer_outputs) #Creates a model that will return thes outputs, given the model input

activations=activation_model.predict(img_tensor) #Returns a list of five Numpy arrays: one array per layer activation

first_layer_activation=activations[0]

print('Fourth channel of the activation of the first layer of the original model :')
plt.matshow(first_layer_activation[0,:,:,4],cmap='viridis')
print('This channel appears to encode a diagonal edge detector')

print('Seventh channel of the activation of the first layer of the original model :')
plt.matshow(first_layer_activation[0,:,:,7],cmap='viridis')

print('VIZUALIZACION OF EVERY CHANNEL IN EVERY INTERMEDIATE ACTIVATION : ')

layer_names = []
for layer in model.layers[:8]:layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:, :,col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                    row * size : (row + 1) * size] = channel_image
            
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

# %%
