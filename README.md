# bt_gan: "deep fakes" aus MdB-Porträts

Adaptiert von https://www.kaggle.com/nageshsingh/generate-realistic-human-face-using-gan

Bildbasis: https://bilddatenbank.bundestag.de/search/picture-result?query=&filterQuery%5Bereignis%5D%5B0%5D=Portr%C3%A4t%2FPortrait&sortVal=3

<table><tr>
<td> <img src="img/original_mdb_sample25.png" alt="original" style="width: 450px;"/> </td>
<td> <img src="img/generated_399.png" alt="generated" style="width: 350px;"/> </td>
</tr></table>

## Inhalt:
* [Beschreibung](#beschreibung)
* [Bilddaten einlesen](#bilddateneinlesen)
* [Neural networks erstellen](#nnerstellen)
* [GAN trainieren](#gantrainieren)
* [Ergebnisse und Diskussion](#ergebnisseunddiskussion)

## Beschreibung <a class="anchor" id="first-bullet"></a>

Wir werden "deep fake"-Gesichter aus öffentlich verfügbaren Bildern von Bundestagsabgeordneten erstellen.

Dazu wird ein GAN (Generative Adversarial Network) verwendet, ein besonderer Ansatz aus dem Bereich Machine Learning / Deep Learning mit Neural Networks (NN).

Der Clou an einem GAN: Es besteht tatsächlich aus zwei Neural Networks, die gegeneinander arbeiten. Das discriminator-NN übernimmt die Rolle einer Detektivin, während das generator-NN die Rolle einer Fälscherin übernimmt. Das generator-NN generiert zufällige Porträts, die dann mit tatsächlichen Porträts vermischt dem discriminator-NN vorgelegt werden. Das discriminator-NN muss nun entscheiden, welche Bilder Fälschungen und welche echt (also aus dem ursprünglichen Datensatz) sind. Dieser Prozess setzt sich iterativ fort: Die Entscheidungen des discriminator-NN werden dem generator-NN gegeben, welches nun versucht, neue und bessere Fälschungen zu erzeugen, die wiederum dem discriminator-NN mit echten Bildern gemischt gezeigt werden. Das discriminator-NN wird aber auch mit jeder Runde besser darin, Fälschungen von echten Bildern zu unterscheiden. Am Ende dieses oft wiederholten Spiels stehen gefälschte Porträts, die den echten Bildern in bestimmten Aspekten sehr ähnlich geworden sind.

Der untenstehende Code implementiert dies in Python/Tensorflow. Zunächst werden notwendige Bibliotheken importiert.


```python
import numpy as np # linear algebra and arrays

import pandas as pd # data processing

import random # for random values

from scipy.ndimage import rotate # to rotate images a little bit

from matplotlib import pyplot as plt # plots

import tensorflow as tf # for nn

import os 
from tqdm import tqdm # system operations

from PIL import Image as Img # image operations

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop # keras for sequential model building
```

Manche Grafikkarten benötigen die nachstehende Einstellung, um RAM-Overloads zu vermeiden:


```python
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

## Bilder einlesen <a class="anchor" id="bildereinlesen"></a>


Einlesen der Bilder (die Bilder sind nicht Teil des Repositorys, einige Originalbilder sind aber im Notebook zu sehen):


```python
PIC_DIR = f'../bundestagportraits_centercrop/'

IMAGES_COUNT = 1684
WIDTH = 128
HEIGHT= 128

images = []
for pic_file in tqdm(os.listdir(PIC_DIR)[:IMAGES_COUNT]):
    pic = Img.open(PIC_DIR + pic_file).convert("RGB") # "LA"
    pic.thumbnail((WIDTH, HEIGHT), Img.ANTIALIAS)
    images.append(np.uint8(pic))
```

    100%|██████████| 1684/1684 [00:00<00:00, 1759.36it/s]



```python
images = np.array(images) / 255
print(images.shape)

np.random.shuffle(images)
```

    (1684, 128, 128, 3)



```python
# helper function for cropping after rotation
def crop_center(img,newdimx,newdimy):
    n,y,x,c = img.shape
    startx = x//2-((newdimx)//2)
    starty = y//2-((newdimy)//2)    
    return img[:,starty:starty+newdimy,startx:startx+newdimx,:]
```


```python
#Display random 25 images
fig = plt.figure(1, figsize=(10, 10))
for i in range(25):
    ir = np.random.randint(IMAGES_COUNT)
    plt.subplot(5, 5, i+1)
    plt.imshow(images[ir])
    plt.axis('off')
plt.show()
fig.savefig('original_mdb_sample25.png', bbox_inches='tight')
```


    
![png](bt_gan_files/bt_gan_13_0.png)
    


Die ursprünglich über 2000 Bilder waren von unterschiedlichen Abmessungen. Alle Bilder wurden automatisiert mittig (mit leichter Verschiebung nach oben) in gleich große Quadrate zugeschnitten. Anschließend wurden von Hand Bilder aussortiert, die kein oder nur wenig Gesicht zeigen und damit nicht Porträts im eigentlichen Sinn sind.

## Neural Networks erstellen <a class="anchor" id="nnerstellen"></a>


Der nachstehende Code beschreibt das erste Neural Network, das generator-NN bzw. die Fälscherin. Der Input für das generator-NN ist "white noise", d.h. ein Array aus Zufallszahlen. Der Output des generator-NN ist ein Bild mit denselben Abmessungen wie die Original-Bilder. Mit unterschiedlichem white noise-Input lassen sich später immer neue, beliebig viele Fälschungen erzeugen. Bemerkenswert ist, dass das generator-NN, also die Fälscherin, nie die echten Bilder sieht!


```python
LATENT_DIM = 32
CHANNELS = 3

def create_generator():
    gen_input = Input(shape=(LATENT_DIM, ))

    x = Dense(128 * 16 * 16)(gen_input)
    x = LeakyReLU()(x)
    x = Reshape((16, 16, 128))(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(CHANNELS, 7, activation='tanh', padding='same')(x)

    generator = Model(gen_input, x)
    return generator
```

Als nächstes wird das discriminator-NN definiert, die Detektivin. Das discriminator-NN erhält als Input ein Bild mit den korrekten Abmessungen, als Output wird nur eine Zahl ausgegeben: Die Entscheidung, ob es sich um eine Fälschung oder nicht handelt.


```python
def create_discriminator():
    disc_input = Input(shape=(HEIGHT, WIDTH, CHANNELS))

    x = Conv2D(256, 3)(disc_input)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(disc_input, x)

    optimizer = RMSprop(
        lr=.0001,
        clipvalue=1.0,
        decay=1e-8
    )

    discriminator.compile(
        optimizer=optimizer,
        loss='binary_crossentropy'
    )

    return discriminator
```


```python
generator = create_generator()
generator.summary()
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 32)]              0         
    _________________________________________________________________
    dense (Dense)                (None, 32768)             1081344   
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 32768)             0         
    _________________________________________________________________
    reshape (Reshape)            (None, 16, 16, 128)       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 16, 16, 256)       819456    
    _________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)    (None, 16, 16, 256)       0         
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 32, 32, 256)       1048832   
    _________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)    (None, 32, 32, 256)       0         
    _________________________________________________________________
    conv2d_transpose_1 (Conv2DTr (None, 64, 64, 256)       1048832   
    _________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)    (None, 64, 64, 256)       0         
    _________________________________________________________________
    conv2d_transpose_2 (Conv2DTr (None, 128, 128, 256)     1048832   
    _________________________________________________________________
    leaky_re_lu_4 (LeakyReLU)    (None, 128, 128, 256)     0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 128, 128, 512)     3277312   
    _________________________________________________________________
    leaky_re_lu_5 (LeakyReLU)    (None, 128, 128, 512)     0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 128, 128, 512)     6554112   
    _________________________________________________________________
    leaky_re_lu_6 (LeakyReLU)    (None, 128, 128, 512)     0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 128, 128, 3)       75267     
    =================================================================
    Total params: 14,953,987
    Trainable params: 14,953,987
    Non-trainable params: 0
    _________________________________________________________________



```python
discriminator = create_discriminator()
discriminator.trainable = False
discriminator.summary()
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         [(None, 128, 128, 3)]     0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 126, 126, 256)     7168      
    _________________________________________________________________
    leaky_re_lu_7 (LeakyReLU)    (None, 126, 126, 256)     0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 62, 62, 256)       1048832   
    _________________________________________________________________
    leaky_re_lu_8 (LeakyReLU)    (None, 62, 62, 256)       0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 30, 30, 256)       1048832   
    _________________________________________________________________
    leaky_re_lu_9 (LeakyReLU)    (None, 30, 30, 256)       0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 14, 14, 256)       1048832   
    _________________________________________________________________
    leaky_re_lu_10 (LeakyReLU)   (None, 14, 14, 256)       0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 6, 6, 256)         1048832   
    _________________________________________________________________
    leaky_re_lu_11 (LeakyReLU)   (None, 6, 6, 256)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0         
    _________________________________________________________________
    dropout (Dropout)            (None, 9216)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 9217      
    =================================================================
    Total params: 4,211,713
    Trainable params: 0
    Non-trainable params: 4,211,713
    _________________________________________________________________


Für das GAN werden beide NNs zusammengesetzt:


```python
gan_input = Input(shape=(LATENT_DIM, ))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
```


```python
optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=optimizer, loss='binary_crossentropy')
```


```python
gan.summary()
```

    Model: "model_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_3 (InputLayer)         [(None, 32)]              0         
    _________________________________________________________________
    model (Functional)           (None, 128, 128, 3)       14953987  
    _________________________________________________________________
    model_1 (Functional)         (None, 1)                 4211713   
    =================================================================
    Total params: 19,165,700
    Trainable params: 14,953,987
    Non-trainable params: 4,211,713
    _________________________________________________________________


## GAN trainieren <a class="anchor" id="gantrainieren"></a>


Nun folgt das Training des NN (dauert ein Weilchen...). Je nach Power der Grafikkarte(n) können hier größere Batchsizes eingesetzt werden. Ein Batch ist die Anzahl der Bilder, die die Grafikkarte gleichzeitig durch das NN schieben kann, um die Parameter des NN upzudaten (insgesamt knapp 15 Mio. trainierbare Parameter!). Alle 50 Iterationen wird ein Bild gespeichert, dass die aktuellen Fälschungen des generator-NN repräsentiert.


```python
import time
iters = 20000
batch_size = 14

RES_DIR = 'res2'
FILE_PATH = '%s/generated_%d.png'
if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)

CONTROL_SIZE_SQRT = 6
control_vectors = np.random.normal(size=(CONTROL_SIZE_SQRT**2, LATENT_DIM)) / 2

start = 0
d_losses = []
a_losses = []
images_saved = 0
for step in range(iters):
    start_time = time.time()
    latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
    generated = generator.predict(latent_vectors)
    
    real = images[start:start + batch_size]
    
    shouldiflip = bool(random.getrandbits(1))
    if shouldiflip:
        real = np.flip(real, 2)
    
#    shouldirotate = bool(random.getrandbits(1))
#    if shouldirotate:
#        real = crop_center(rotate(real, np.random.randint(low=-10, high=11), axes=(2,1), reshape=True), 128, 128)
    
    combined_images = np.concatenate([generated, real])

    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += .05 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)
    d_losses.append(d_loss)

    latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
    misleading_targets = np.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(latent_vectors, misleading_targets)
    a_losses.append(a_loss)

    start += batch_size
    if start > images.shape[0] - batch_size:
        start = 0
        np.random.shuffle(images)

    if step % 50 == 49:
        gan.save_weights('gan.h5')

        print('%d/%d: d_loss: %.4f,  a_loss: %.4f.  (%.1f min)' % (step + 1, iters, d_loss, a_loss, time.time() - start_time))

        control_image = np.zeros((WIDTH * CONTROL_SIZE_SQRT, HEIGHT * CONTROL_SIZE_SQRT, CHANNELS))
        control_generated = generator.predict(control_vectors)
        
        for i in range(CONTROL_SIZE_SQRT ** 2):
            x_off = i % CONTROL_SIZE_SQRT
            y_off = i // CONTROL_SIZE_SQRT
            control_image[x_off * WIDTH:(x_off + 1) * WIDTH, y_off * HEIGHT:(y_off + 1) * HEIGHT, :] = control_generated[i, :, :, :]
        im = Img.fromarray(np.uint8(control_image * 255))#.save(StringIO(), 'jpeg')
        im.save(FILE_PATH % (RES_DIR, images_saved))
        images_saved += 1
```

    50/20000: d_loss: 0.5775,  a_loss: 0.7562.  (3.4 min)
    100/20000: d_loss: 0.6808,  a_loss: 0.7204.  (3.4 min)
    150/20000: d_loss: 0.6862,  a_loss: 0.7299.  (3.4 min)
    200/20000: d_loss: 0.6874,  a_loss: 0.7913.  (3.4 min)
    250/20000: d_loss: 0.6784,  a_loss: 0.8275.  (3.4 min)
    300/20000: d_loss: 0.6776,  a_loss: 0.8092.  (3.4 min)
    350/20000: d_loss: 0.6520,  a_loss: 0.9703.  (3.4 min)
    400/20000: d_loss: 0.6263,  a_loss: 0.8623.  (3.4 min)
    450/20000: d_loss: 0.6803,  a_loss: 0.8993.  (3.4 min)
    500/20000: d_loss: 0.6526,  a_loss: 1.8234.  (3.4 min)
    550/20000: d_loss: 0.6742,  a_loss: 1.0965.  (3.4 min)
    600/20000: d_loss: 0.5863,  a_loss: 0.9006.  (3.5 min)
    650/20000: d_loss: 0.7219,  a_loss: 0.6727.  (3.4 min)
    700/20000: d_loss: 0.6122,  a_loss: 0.9923.  (3.4 min)
    750/20000: d_loss: 0.5624,  a_loss: 1.3906.  (3.4 min)
    800/20000: d_loss: 0.5245,  a_loss: 1.0786.  (3.4 min)
    850/20000: d_loss: 0.5817,  a_loss: 1.1589.  (3.4 min)
    900/20000: d_loss: 0.7089,  a_loss: 1.7059.  (3.4 min)
    950/20000: d_loss: 0.6915,  a_loss: 1.0253.  (3.4 min)
    1000/20000: d_loss: 0.5643,  a_loss: 1.1693.  (3.4 min)
    1050/20000: d_loss: 0.6732,  a_loss: 1.2642.  (3.4 min)
    1100/20000: d_loss: 0.7269,  a_loss: 1.8139.  (3.4 min)
    1150/20000: d_loss: 0.6781,  a_loss: 1.9394.  (3.4 min)
    1200/20000: d_loss: 0.5339,  a_loss: 1.9803.  (3.5 min)
    1250/20000: d_loss: 0.6464,  a_loss: 1.2760.  (3.4 min)
    1300/20000: d_loss: 0.7593,  a_loss: 1.6047.  (3.4 min)
    1350/20000: d_loss: 0.5697,  a_loss: 1.4761.  (3.4 min)
    1400/20000: d_loss: 0.6296,  a_loss: 0.8524.  (3.4 min)
    1450/20000: d_loss: 0.7214,  a_loss: 1.8392.  (3.4 min)
    1500/20000: d_loss: 0.6351,  a_loss: 0.8410.  (3.4 min)
    1550/20000: d_loss: 0.5499,  a_loss: 1.0624.  (3.4 min)
    1600/20000: d_loss: 0.6162,  a_loss: 1.1070.  (3.4 min)
    1650/20000: d_loss: 0.5806,  a_loss: 0.8542.  (3.4 min)
    1700/20000: d_loss: 0.6765,  a_loss: 1.6307.  (3.4 min)
    1750/20000: d_loss: 0.6141,  a_loss: 0.8763.  (3.4 min)
    1800/20000: d_loss: 0.6486,  a_loss: 1.2349.  (3.5 min)
    1850/20000: d_loss: 0.6827,  a_loss: 1.5316.  (3.4 min)
    1900/20000: d_loss: 0.6340,  a_loss: 1.1071.  (3.4 min)
    1950/20000: d_loss: 0.6315,  a_loss: 1.2834.  (3.4 min)
    2000/20000: d_loss: 0.7733,  a_loss: 1.2457.  (3.4 min)
    2050/20000: d_loss: 0.5188,  a_loss: 1.0245.  (3.4 min)
    2100/20000: d_loss: 0.7680,  a_loss: 1.1895.  (3.4 min)
    2150/20000: d_loss: 0.7165,  a_loss: 1.1813.  (3.4 min)
    2200/20000: d_loss: 0.8110,  a_loss: 1.7523.  (3.4 min)
    2250/20000: d_loss: 0.6490,  a_loss: 0.8627.  (3.4 min)
    2300/20000: d_loss: 0.6188,  a_loss: 1.0151.  (3.4 min)
    2350/20000: d_loss: 0.6906,  a_loss: 1.0934.  (3.4 min)
    2400/20000: d_loss: 0.7198,  a_loss: 1.0836.  (3.5 min)
    2450/20000: d_loss: 0.5274,  a_loss: 1.7921.  (3.5 min)
    2500/20000: d_loss: 0.6462,  a_loss: 0.8222.  (3.4 min)
    2550/20000: d_loss: 0.5881,  a_loss: 1.2983.  (3.4 min)
    2600/20000: d_loss: 0.5906,  a_loss: 1.5941.  (3.4 min)
    2650/20000: d_loss: 0.6390,  a_loss: 1.4312.  (3.4 min)
    2700/20000: d_loss: 0.7762,  a_loss: 0.8817.  (3.4 min)
    2750/20000: d_loss: 0.6664,  a_loss: 0.9268.  (3.4 min)
    2800/20000: d_loss: 0.6639,  a_loss: 1.0654.  (3.4 min)
    2850/20000: d_loss: 0.6675,  a_loss: 0.9839.  (3.4 min)
    2900/20000: d_loss: 0.6345,  a_loss: 1.1992.  (3.4 min)
    2950/20000: d_loss: 0.6158,  a_loss: 0.8897.  (3.4 min)
    3000/20000: d_loss: 0.6840,  a_loss: 1.0305.  (3.5 min)
    3050/20000: d_loss: 0.7069,  a_loss: 1.0882.  (3.4 min)
    3100/20000: d_loss: 0.5733,  a_loss: 1.5545.  (3.4 min)
    3150/20000: d_loss: 0.5842,  a_loss: 1.0542.  (3.5 min)
    3200/20000: d_loss: 0.7374,  a_loss: 0.9853.  (3.4 min)
    3250/20000: d_loss: 0.6797,  a_loss: 1.4367.  (3.4 min)
    3300/20000: d_loss: 0.6920,  a_loss: 0.8156.  (3.4 min)
    3350/20000: d_loss: 0.6880,  a_loss: 0.8595.  (3.4 min)
    3400/20000: d_loss: 0.6047,  a_loss: 0.8379.  (3.4 min)
    3450/20000: d_loss: 0.6366,  a_loss: 1.3359.  (3.4 min)
    3500/20000: d_loss: 0.5682,  a_loss: 0.7548.  (3.5 min)
    3550/20000: d_loss: 0.6310,  a_loss: 1.1110.  (3.4 min)
    3600/20000: d_loss: 0.6901,  a_loss: 0.9491.  (3.5 min)
    3650/20000: d_loss: 0.7184,  a_loss: 1.0910.  (3.4 min)
    3700/20000: d_loss: 0.6092,  a_loss: 1.1697.  (3.4 min)
    3750/20000: d_loss: 0.5961,  a_loss: 1.2299.  (3.4 min)
    3800/20000: d_loss: 0.6326,  a_loss: 1.1589.  (3.4 min)
    3850/20000: d_loss: 0.6207,  a_loss: 1.1281.  (3.4 min)
    3900/20000: d_loss: 0.6366,  a_loss: 1.1117.  (3.4 min)
    3950/20000: d_loss: 0.6746,  a_loss: 1.0164.  (3.4 min)
    4000/20000: d_loss: 0.7010,  a_loss: 0.8922.  (3.4 min)
    4050/20000: d_loss: 0.7091,  a_loss: 1.0433.  (3.4 min)
    4100/20000: d_loss: 0.5329,  a_loss: 1.3798.  (3.4 min)
    4150/20000: d_loss: 0.5509,  a_loss: 1.2866.  (3.4 min)
    4200/20000: d_loss: 0.6521,  a_loss: 1.3460.  (3.5 min)
    4250/20000: d_loss: 0.5399,  a_loss: 1.2580.  (3.4 min)
    4300/20000: d_loss: 0.6360,  a_loss: 0.7794.  (3.4 min)
    4350/20000: d_loss: 0.6030,  a_loss: 0.9316.  (3.4 min)
    4400/20000: d_loss: 0.5649,  a_loss: 1.3802.  (3.4 min)
    4450/20000: d_loss: 0.6922,  a_loss: 1.3954.  (3.4 min)
    4500/20000: d_loss: 0.6266,  a_loss: 1.0383.  (3.4 min)
    4550/20000: d_loss: 0.5654,  a_loss: 1.4675.  (3.4 min)
    4600/20000: d_loss: 0.7521,  a_loss: 1.1440.  (3.4 min)
    4650/20000: d_loss: 0.5559,  a_loss: 0.8688.  (3.4 min)
    4700/20000: d_loss: 0.8255,  a_loss: 0.8430.  (3.4 min)
    4750/20000: d_loss: 0.5596,  a_loss: 1.0092.  (3.4 min)
    4800/20000: d_loss: 0.6571,  a_loss: 0.8961.  (3.5 min)
    4850/20000: d_loss: 0.5788,  a_loss: 0.7576.  (3.4 min)
    4900/20000: d_loss: 0.6572,  a_loss: 0.8023.  (3.4 min)
    4950/20000: d_loss: 0.5578,  a_loss: 1.5219.  (3.4 min)
    5000/20000: d_loss: 0.7263,  a_loss: 1.0765.  (3.4 min)
    5050/20000: d_loss: 0.6042,  a_loss: 1.1759.  (3.4 min)
    5100/20000: d_loss: 0.6004,  a_loss: 1.0660.  (3.4 min)
    5150/20000: d_loss: 0.6487,  a_loss: 1.4730.  (3.4 min)
    5200/20000: d_loss: 0.7166,  a_loss: 1.4274.  (3.4 min)
    5250/20000: d_loss: 0.6229,  a_loss: 1.1668.  (3.4 min)
    5300/20000: d_loss: 0.6032,  a_loss: 1.2304.  (4.0 min)
    5350/20000: d_loss: 0.7166,  a_loss: 1.5519.  (3.4 min)
    5400/20000: d_loss: 0.5623,  a_loss: 1.0982.  (3.5 min)
    5450/20000: d_loss: 0.6086,  a_loss: 1.0825.  (3.4 min)
    5500/20000: d_loss: 0.6653,  a_loss: 0.9435.  (3.4 min)
    5550/20000: d_loss: 0.8074,  a_loss: 1.3258.  (3.4 min)
    5600/20000: d_loss: 0.5533,  a_loss: 1.0652.  (3.4 min)
    5650/20000: d_loss: 0.6535,  a_loss: 0.9941.  (3.4 min)
    5700/20000: d_loss: 0.6624,  a_loss: 1.4193.  (3.4 min)
    5750/20000: d_loss: 0.6891,  a_loss: 1.0240.  (3.4 min)
    5800/20000: d_loss: 0.6565,  a_loss: 0.7153.  (3.4 min)
    5850/20000: d_loss: 0.5870,  a_loss: 1.2775.  (3.4 min)
    5900/20000: d_loss: 0.6257,  a_loss: 1.0491.  (3.4 min)
    5950/20000: d_loss: 0.4595,  a_loss: 0.9478.  (3.4 min)
    6000/20000: d_loss: 0.5540,  a_loss: 1.1395.  (3.5 min)
    6050/20000: d_loss: 0.5727,  a_loss: 1.0128.  (3.4 min)
    6100/20000: d_loss: 0.6143,  a_loss: 0.8718.  (3.4 min)
    6150/20000: d_loss: 0.5965,  a_loss: 0.8571.  (3.4 min)
    6200/20000: d_loss: 0.6954,  a_loss: 1.2518.  (3.4 min)
    6250/20000: d_loss: 0.6529,  a_loss: 1.3872.  (3.4 min)
    6300/20000: d_loss: 0.5905,  a_loss: 1.6973.  (3.4 min)
    6350/20000: d_loss: 0.5712,  a_loss: 1.1069.  (3.5 min)
    6400/20000: d_loss: 0.6407,  a_loss: 1.1223.  (3.4 min)
    6450/20000: d_loss: 0.6311,  a_loss: 1.0880.  (3.4 min)
    6500/20000: d_loss: 0.6918,  a_loss: 1.3153.  (3.4 min)
    6550/20000: d_loss: 0.7110,  a_loss: 1.2758.  (3.4 min)
    6600/20000: d_loss: 0.6867,  a_loss: 1.3541.  (3.5 min)
    6650/20000: d_loss: 0.6089,  a_loss: 1.3769.  (3.4 min)
    6700/20000: d_loss: 0.6358,  a_loss: 1.2750.  (3.4 min)
    6750/20000: d_loss: 0.6056,  a_loss: 1.0177.  (3.4 min)
    6800/20000: d_loss: 0.6071,  a_loss: 1.2235.  (3.4 min)
    6850/20000: d_loss: 0.4409,  a_loss: 1.3025.  (3.4 min)
    6900/20000: d_loss: 0.6986,  a_loss: 1.4765.  (3.4 min)
    6950/20000: d_loss: 0.4630,  a_loss: 1.6623.  (3.4 min)
    7000/20000: d_loss: 0.4977,  a_loss: 1.2581.  (3.4 min)
    7050/20000: d_loss: 0.6254,  a_loss: 1.3824.  (3.4 min)
    7100/20000: d_loss: 0.6509,  a_loss: 1.0312.  (3.4 min)
    7150/20000: d_loss: 0.5411,  a_loss: 1.3020.  (3.4 min)
    7200/20000: d_loss: 0.5871,  a_loss: 0.8511.  (3.5 min)
    7250/20000: d_loss: 0.5708,  a_loss: 1.0434.  (3.4 min)
    7300/20000: d_loss: 0.6214,  a_loss: 1.5796.  (3.4 min)
    7350/20000: d_loss: 0.5806,  a_loss: 1.0405.  (3.4 min)
    7400/20000: d_loss: 0.5708,  a_loss: 1.1485.  (3.4 min)
    7450/20000: d_loss: 0.6471,  a_loss: 1.4330.  (3.4 min)
    7500/20000: d_loss: 0.5694,  a_loss: 1.3349.  (3.4 min)
    7550/20000: d_loss: 0.5925,  a_loss: 1.9813.  (3.4 min)
    7600/20000: d_loss: 0.5894,  a_loss: 1.0977.  (3.5 min)
    7650/20000: d_loss: 0.5377,  a_loss: 1.8044.  (3.4 min)
    7700/20000: d_loss: 0.5427,  a_loss: 1.1154.  (3.4 min)
    7750/20000: d_loss: 0.7622,  a_loss: 1.0704.  (3.4 min)
    7800/20000: d_loss: 0.5082,  a_loss: 1.0180.  (3.6 min)
    7850/20000: d_loss: 0.5485,  a_loss: 1.4268.  (3.4 min)
    7900/20000: d_loss: 0.6350,  a_loss: 1.2905.  (3.4 min)
    7950/20000: d_loss: 0.5930,  a_loss: 1.4648.  (3.4 min)
    8000/20000: d_loss: 0.7011,  a_loss: 1.1583.  (3.4 min)
    8050/20000: d_loss: 0.5263,  a_loss: 1.0560.  (3.4 min)
    8100/20000: d_loss: 0.5866,  a_loss: 1.0027.  (3.4 min)
    8150/20000: d_loss: 0.6135,  a_loss: 1.1564.  (3.4 min)
    8200/20000: d_loss: 0.6464,  a_loss: 1.0563.  (3.4 min)
    8250/20000: d_loss: 0.6054,  a_loss: 0.9156.  (3.4 min)
    8300/20000: d_loss: 0.5879,  a_loss: 1.1049.  (3.4 min)
    8350/20000: d_loss: 0.6130,  a_loss: 1.6076.  (3.4 min)
    8400/20000: d_loss: 0.5215,  a_loss: 1.1429.  (3.5 min)
    8450/20000: d_loss: 0.4336,  a_loss: 1.2639.  (3.4 min)
    8500/20000: d_loss: 0.5305,  a_loss: 1.9200.  (3.4 min)
    8550/20000: d_loss: 0.5400,  a_loss: 1.0220.  (3.4 min)
    8600/20000: d_loss: 0.4751,  a_loss: 1.5211.  (3.4 min)
    8650/20000: d_loss: 0.4903,  a_loss: 1.0800.  (3.4 min)
    8700/20000: d_loss: 0.5281,  a_loss: 1.4760.  (3.4 min)
    8750/20000: d_loss: 0.5958,  a_loss: 1.0631.  (3.4 min)
    8800/20000: d_loss: 0.6182,  a_loss: 0.8744.  (3.4 min)
    8850/20000: d_loss: 0.6236,  a_loss: 1.1609.  (3.5 min)
    8900/20000: d_loss: 0.4945,  a_loss: 1.1490.  (3.4 min)
    8950/20000: d_loss: 0.4169,  a_loss: 1.7028.  (3.4 min)
    9000/20000: d_loss: 0.7059,  a_loss: 1.3561.  (3.5 min)
    9050/20000: d_loss: 0.6081,  a_loss: 1.5556.  (3.4 min)
    9100/20000: d_loss: 0.5345,  a_loss: 1.8665.  (3.4 min)
    9150/20000: d_loss: 0.5736,  a_loss: 1.8432.  (3.4 min)
    9200/20000: d_loss: 0.5564,  a_loss: 1.1737.  (3.4 min)
    9250/20000: d_loss: 0.5886,  a_loss: 1.4858.  (3.4 min)
    9300/20000: d_loss: 0.6680,  a_loss: 1.0513.  (3.4 min)
    9350/20000: d_loss: 0.6471,  a_loss: 0.8321.  (3.4 min)
    9400/20000: d_loss: 0.5749,  a_loss: 0.9994.  (3.4 min)
    9450/20000: d_loss: 0.5511,  a_loss: 1.4871.  (3.4 min)
    9500/20000: d_loss: 0.6010,  a_loss: 1.1517.  (3.4 min)
    9550/20000: d_loss: 0.5416,  a_loss: 1.1495.  (3.4 min)
    9600/20000: d_loss: 0.5422,  a_loss: 1.3314.  (3.6 min)
    9650/20000: d_loss: 0.5220,  a_loss: 1.0181.  (3.5 min)
    9700/20000: d_loss: 0.5269,  a_loss: 1.6742.  (3.4 min)
    9750/20000: d_loss: 0.5373,  a_loss: 1.5438.  (3.4 min)
    9800/20000: d_loss: 0.5837,  a_loss: 1.3059.  (3.4 min)
    9850/20000: d_loss: 0.5752,  a_loss: 1.6989.  (3.4 min)
    9900/20000: d_loss: 0.5387,  a_loss: 1.0283.  (3.4 min)
    9950/20000: d_loss: 0.4147,  a_loss: 1.4750.  (3.4 min)
    10000/20000: d_loss: 0.5501,  a_loss: 0.9271.  (3.4 min)
    10050/20000: d_loss: 0.6652,  a_loss: 1.3458.  (3.4 min)
    10100/20000: d_loss: 0.7128,  a_loss: 1.2568.  (3.4 min)
    10150/20000: d_loss: 0.6560,  a_loss: 0.9057.  (3.4 min)
    10200/20000: d_loss: 0.5686,  a_loss: 1.3011.  (3.5 min)
    10250/20000: d_loss: 0.6214,  a_loss: 0.9016.  (3.4 min)
    10300/20000: d_loss: 0.5942,  a_loss: 1.4577.  (3.4 min)
    10350/20000: d_loss: 0.6707,  a_loss: 1.0532.  (3.4 min)
    10400/20000: d_loss: 0.5056,  a_loss: 1.1956.  (3.4 min)
    10450/20000: d_loss: 0.5209,  a_loss: 1.3231.  (3.4 min)
    10500/20000: d_loss: 0.5432,  a_loss: 1.0807.  (3.4 min)
    10550/20000: d_loss: 0.6016,  a_loss: 0.6415.  (3.4 min)
    10600/20000: d_loss: 0.4171,  a_loss: 1.3848.  (3.4 min)
    10650/20000: d_loss: 0.5847,  a_loss: 1.0827.  (3.4 min)
    10700/20000: d_loss: 0.6171,  a_loss: 1.5815.  (3.4 min)
    10750/20000: d_loss: 0.4439,  a_loss: 1.1620.  (3.4 min)
    10800/20000: d_loss: 0.6189,  a_loss: 1.1926.  (3.5 min)
    10850/20000: d_loss: 0.5055,  a_loss: 0.8861.  (3.4 min)
    10900/20000: d_loss: 0.5805,  a_loss: 1.7949.  (3.4 min)
    10950/20000: d_loss: 0.5393,  a_loss: 1.8112.  (3.4 min)
    11000/20000: d_loss: 0.6347,  a_loss: 1.4462.  (3.4 min)
    11050/20000: d_loss: 0.5989,  a_loss: 0.9708.  (3.4 min)
    11100/20000: d_loss: 0.5560,  a_loss: 1.3309.  (3.4 min)
    11150/20000: d_loss: 0.6447,  a_loss: 1.2238.  (3.4 min)
    11200/20000: d_loss: 0.5658,  a_loss: 1.5643.  (3.4 min)
    11250/20000: d_loss: 0.5724,  a_loss: 0.9604.  (3.4 min)
    11300/20000: d_loss: 0.5177,  a_loss: 1.4531.  (3.4 min)
    11350/20000: d_loss: 0.6387,  a_loss: 0.9550.  (3.4 min)
    11400/20000: d_loss: 0.6675,  a_loss: 1.1177.  (3.5 min)
    11450/20000: d_loss: 0.4761,  a_loss: 1.5325.  (3.4 min)
    11500/20000: d_loss: 0.5313,  a_loss: 1.4710.  (3.4 min)
    11550/20000: d_loss: 0.5637,  a_loss: 0.9730.  (3.4 min)
    11600/20000: d_loss: 0.7443,  a_loss: 1.4353.  (3.4 min)
    11650/20000: d_loss: 0.5408,  a_loss: 0.8314.  (3.4 min)
    11700/20000: d_loss: 0.6052,  a_loss: 1.2684.  (3.4 min)
    11750/20000: d_loss: 0.4877,  a_loss: 1.6268.  (3.4 min)
    11800/20000: d_loss: 0.5516,  a_loss: 1.2546.  (3.4 min)
    11850/20000: d_loss: 0.5728,  a_loss: 1.3651.  (3.4 min)
    11900/20000: d_loss: 0.4758,  a_loss: 1.3081.  (3.4 min)
    11950/20000: d_loss: 0.5103,  a_loss: 1.7045.  (3.4 min)
    12000/20000: d_loss: 0.6972,  a_loss: 1.0786.  (8.3 min)
    12050/20000: d_loss: 0.4640,  a_loss: 1.1435.  (8.1 min)
    12100/20000: d_loss: 0.6727,  a_loss: 1.8008.  (8.0 min)
    12150/20000: d_loss: 0.5869,  a_loss: 1.0188.  (3.3 min)
    12200/20000: d_loss: 0.5413,  a_loss: 2.2525.  (3.4 min)
    12250/20000: d_loss: 0.5589,  a_loss: 0.9877.  (3.4 min)
    12300/20000: d_loss: 0.6775,  a_loss: 0.8884.  (3.4 min)
    12350/20000: d_loss: 0.4898,  a_loss: 1.3468.  (3.4 min)
    12400/20000: d_loss: 0.4454,  a_loss: 1.2751.  (3.4 min)
    12450/20000: d_loss: 0.5968,  a_loss: 1.4505.  (3.4 min)
    12500/20000: d_loss: 0.5154,  a_loss: 1.8362.  (3.4 min)
    12550/20000: d_loss: 0.7227,  a_loss: 1.5566.  (3.4 min)
    12600/20000: d_loss: 0.4789,  a_loss: 1.8411.  (3.5 min)
    12650/20000: d_loss: 0.4270,  a_loss: 1.3746.  (3.4 min)
    12700/20000: d_loss: 0.6641,  a_loss: 0.8390.  (3.4 min)
    12750/20000: d_loss: 0.6369,  a_loss: 0.9709.  (3.4 min)
    12800/20000: d_loss: 0.7275,  a_loss: 1.9223.  (3.4 min)
    12850/20000: d_loss: 0.5276,  a_loss: 1.8458.  (3.4 min)
    12900/20000: d_loss: 0.5522,  a_loss: 1.0629.  (3.4 min)
    12950/20000: d_loss: 0.5825,  a_loss: 1.4058.  (3.5 min)
    13000/20000: d_loss: 0.6049,  a_loss: 1.9491.  (3.4 min)
    13050/20000: d_loss: 0.5378,  a_loss: 1.5421.  (3.4 min)
    13100/20000: d_loss: 0.5225,  a_loss: 2.0960.  (3.4 min)
    13150/20000: d_loss: 0.5284,  a_loss: 1.7349.  (3.4 min)
    13200/20000: d_loss: 0.5807,  a_loss: 1.5942.  (3.5 min)
    13250/20000: d_loss: 0.5092,  a_loss: 1.3214.  (3.4 min)
    13300/20000: d_loss: 0.5674,  a_loss: 1.2865.  (3.4 min)
    13350/20000: d_loss: 0.6640,  a_loss: 1.5892.  (3.4 min)
    13400/20000: d_loss: 0.4807,  a_loss: 1.3814.  (3.4 min)
    13450/20000: d_loss: 0.6299,  a_loss: 1.4211.  (3.4 min)
    13500/20000: d_loss: 0.4782,  a_loss: 1.0651.  (3.4 min)
    13550/20000: d_loss: 0.4553,  a_loss: 1.5884.  (3.4 min)
    13600/20000: d_loss: 0.5715,  a_loss: 1.4555.  (3.4 min)
    13650/20000: d_loss: 0.6141,  a_loss: 1.3145.  (3.4 min)
    13700/20000: d_loss: 0.3271,  a_loss: 1.6174.  (3.4 min)
    13750/20000: d_loss: 0.7085,  a_loss: 1.0476.  (3.4 min)
    13800/20000: d_loss: 0.5664,  a_loss: 1.6004.  (3.5 min)
    13850/20000: d_loss: 0.6064,  a_loss: 1.2503.  (3.4 min)
    13900/20000: d_loss: 0.5231,  a_loss: 1.5327.  (3.4 min)
    13950/20000: d_loss: 0.5155,  a_loss: 1.9755.  (3.4 min)
    14000/20000: d_loss: 0.6360,  a_loss: 1.2163.  (3.4 min)
    14050/20000: d_loss: 0.5201,  a_loss: 1.2201.  (3.4 min)
    14100/20000: d_loss: 0.5608,  a_loss: 1.5749.  (3.4 min)
    14150/20000: d_loss: 0.5462,  a_loss: 1.5274.  (3.4 min)
    14200/20000: d_loss: 0.4429,  a_loss: 1.3678.  (3.4 min)
    14250/20000: d_loss: 0.5168,  a_loss: 1.1911.  (3.4 min)
    14300/20000: d_loss: 0.5928,  a_loss: 1.5794.  (3.4 min)
    14350/20000: d_loss: 0.5859,  a_loss: 1.7718.  (3.4 min)
    14400/20000: d_loss: 0.4781,  a_loss: 1.6771.  (3.5 min)
    14450/20000: d_loss: 0.4984,  a_loss: 1.4362.  (3.4 min)
    14500/20000: d_loss: 0.3911,  a_loss: 1.3354.  (3.4 min)
    14550/20000: d_loss: 0.7326,  a_loss: 1.6066.  (3.4 min)
    14600/20000: d_loss: 0.4913,  a_loss: 1.6769.  (3.4 min)
    14650/20000: d_loss: 0.4361,  a_loss: 1.6899.  (3.4 min)
    14700/20000: d_loss: 0.4193,  a_loss: 1.9385.  (3.4 min)
    14750/20000: d_loss: 0.4464,  a_loss: 1.3285.  (3.4 min)
    14800/20000: d_loss: 0.4033,  a_loss: 1.5478.  (3.4 min)
    14850/20000: d_loss: 0.4248,  a_loss: 1.1004.  (3.4 min)
    14900/20000: d_loss: 0.4195,  a_loss: 1.7527.  (3.4 min)
    14950/20000: d_loss: 0.4046,  a_loss: 1.6788.  (3.4 min)
    15000/20000: d_loss: 0.6527,  a_loss: 1.2899.  (3.5 min)
    15050/20000: d_loss: 0.5788,  a_loss: 1.9601.  (3.4 min)
    15100/20000: d_loss: 0.5754,  a_loss: 1.7245.  (3.4 min)
    15150/20000: d_loss: 0.4538,  a_loss: 1.8548.  (3.4 min)
    15200/20000: d_loss: 0.5488,  a_loss: 1.8042.  (3.4 min)
    15250/20000: d_loss: 0.6874,  a_loss: 1.5975.  (3.4 min)
    15300/20000: d_loss: 0.5023,  a_loss: 1.0884.  (3.4 min)
    15350/20000: d_loss: 0.5608,  a_loss: 1.4224.  (3.4 min)
    15400/20000: d_loss: 0.5244,  a_loss: 1.6367.  (3.4 min)
    15450/20000: d_loss: 0.5429,  a_loss: 2.5387.  (3.4 min)
    15500/20000: d_loss: 0.4496,  a_loss: 1.6583.  (3.4 min)
    15550/20000: d_loss: 0.4394,  a_loss: 1.3978.  (3.4 min)
    15600/20000: d_loss: 0.5149,  a_loss: 1.7500.  (3.5 min)
    15650/20000: d_loss: 0.3997,  a_loss: 1.2634.  (3.4 min)
    15700/20000: d_loss: 0.6737,  a_loss: 1.8896.  (3.4 min)
    15750/20000: d_loss: 0.4789,  a_loss: 1.5023.  (3.4 min)
    15800/20000: d_loss: 0.6596,  a_loss: 1.3070.  (3.4 min)
    15850/20000: d_loss: 0.3763,  a_loss: 1.5724.  (3.4 min)
    15900/20000: d_loss: 0.4692,  a_loss: 1.7785.  (3.4 min)
    15950/20000: d_loss: 0.3764,  a_loss: 1.2850.  (3.4 min)
    16000/20000: d_loss: 0.5357,  a_loss: 1.7252.  (3.4 min)
    16050/20000: d_loss: 0.6208,  a_loss: 1.9308.  (3.4 min)
    16100/20000: d_loss: 0.6995,  a_loss: 1.1323.  (3.4 min)
    16150/20000: d_loss: 0.4373,  a_loss: 1.8149.  (3.4 min)
    16200/20000: d_loss: 0.4657,  a_loss: 1.4910.  (3.5 min)
    16250/20000: d_loss: 0.5723,  a_loss: 1.3776.  (3.5 min)
    16300/20000: d_loss: 0.4792,  a_loss: 2.3112.  (3.4 min)
    16350/20000: d_loss: 0.5428,  a_loss: 2.3695.  (3.4 min)
    16400/20000: d_loss: 0.7625,  a_loss: 1.2830.  (3.4 min)
    16450/20000: d_loss: 0.4286,  a_loss: 1.5619.  (3.4 min)
    16500/20000: d_loss: 0.3966,  a_loss: 1.6350.  (3.4 min)
    16550/20000: d_loss: 0.4584,  a_loss: 1.5683.  (3.4 min)
    16600/20000: d_loss: 0.4695,  a_loss: 2.2056.  (3.4 min)
    16650/20000: d_loss: 0.4083,  a_loss: 1.9677.  (3.4 min)
    16700/20000: d_loss: 0.4937,  a_loss: 1.5969.  (3.4 min)
    16750/20000: d_loss: 0.3737,  a_loss: 1.6241.  (3.4 min)
    16800/20000: d_loss: 0.6207,  a_loss: 2.7236.  (3.5 min)
    16850/20000: d_loss: 0.4464,  a_loss: 2.2665.  (3.4 min)
    16900/20000: d_loss: 0.3892,  a_loss: 1.8946.  (3.4 min)
    16950/20000: d_loss: 0.3451,  a_loss: 1.7608.  (3.4 min)
    17000/20000: d_loss: 0.4911,  a_loss: 2.3003.  (3.4 min)
    17050/20000: d_loss: 0.2725,  a_loss: 1.8846.  (3.4 min)
    17100/20000: d_loss: 0.5484,  a_loss: 1.8849.  (3.4 min)
    17150/20000: d_loss: 0.5614,  a_loss: 1.6036.  (3.4 min)
    17200/20000: d_loss: 0.3098,  a_loss: 1.7474.  (3.4 min)
    17250/20000: d_loss: 0.3246,  a_loss: 1.0975.  (3.4 min)
    17300/20000: d_loss: 0.4673,  a_loss: 1.9682.  (3.4 min)
    17350/20000: d_loss: 0.5037,  a_loss: 1.3283.  (3.4 min)
    17400/20000: d_loss: 0.5785,  a_loss: 2.0536.  (3.5 min)
    17450/20000: d_loss: 0.4822,  a_loss: 1.6754.  (3.4 min)
    17500/20000: d_loss: 0.4547,  a_loss: 1.4914.  (3.4 min)
    17550/20000: d_loss: 0.3707,  a_loss: 2.1146.  (3.4 min)
    17600/20000: d_loss: 0.5908,  a_loss: 2.2857.  (3.4 min)
    17650/20000: d_loss: 0.3846,  a_loss: 1.8431.  (3.4 min)
    17700/20000: d_loss: 0.3350,  a_loss: 1.6628.  (3.4 min)
    17750/20000: d_loss: 0.5092,  a_loss: 2.2127.  (3.4 min)
    17800/20000: d_loss: 0.3813,  a_loss: 1.9801.  (3.4 min)
    17850/20000: d_loss: 0.2109,  a_loss: 1.7470.  (3.4 min)
    17900/20000: d_loss: 0.4032,  a_loss: 1.5017.  (3.4 min)
    17950/20000: d_loss: 0.4478,  a_loss: 0.8812.  (3.4 min)
    18000/20000: d_loss: 0.4021,  a_loss: 2.2494.  (3.5 min)
    18050/20000: d_loss: 0.4505,  a_loss: 1.3666.  (3.4 min)
    18100/20000: d_loss: 0.3616,  a_loss: 1.7925.  (3.4 min)
    18150/20000: d_loss: 0.5565,  a_loss: 1.5865.  (3.4 min)
    18200/20000: d_loss: 0.4453,  a_loss: 2.1137.  (3.4 min)
    18250/20000: d_loss: 0.4758,  a_loss: 2.6298.  (3.4 min)
    18300/20000: d_loss: 0.4889,  a_loss: 2.1763.  (3.4 min)
    18350/20000: d_loss: 0.3584,  a_loss: 2.6202.  (3.4 min)
    18400/20000: d_loss: 0.6040,  a_loss: 1.1459.  (3.4 min)
    18450/20000: d_loss: 0.5251,  a_loss: 2.1345.  (3.4 min)
    18500/20000: d_loss: 0.4811,  a_loss: 1.8816.  (3.4 min)
    18550/20000: d_loss: 0.3751,  a_loss: 2.5209.  (3.4 min)
    18600/20000: d_loss: 0.2397,  a_loss: 1.9114.  (3.5 min)
    18650/20000: d_loss: 0.4609,  a_loss: 2.4251.  (3.4 min)
    18700/20000: d_loss: 0.4780,  a_loss: 1.4109.  (3.4 min)
    18750/20000: d_loss: 0.3617,  a_loss: 2.6670.  (3.4 min)
    18800/20000: d_loss: 0.4281,  a_loss: 2.2514.  (3.4 min)
    18850/20000: d_loss: 0.3570,  a_loss: 1.9246.  (3.4 min)
    18900/20000: d_loss: 0.4345,  a_loss: 1.6975.  (3.4 min)
    18950/20000: d_loss: 0.4865,  a_loss: 1.4114.  (3.4 min)
    19000/20000: d_loss: 0.5544,  a_loss: 1.9243.  (3.4 min)
    19050/20000: d_loss: 0.3343,  a_loss: 2.0982.  (3.4 min)
    19100/20000: d_loss: 0.5604,  a_loss: 1.1325.  (3.4 min)
    19150/20000: d_loss: 0.4253,  a_loss: 1.5198.  (3.4 min)
    19200/20000: d_loss: 0.5674,  a_loss: 1.3493.  (3.5 min)
    19250/20000: d_loss: 0.4985,  a_loss: 1.3959.  (3.4 min)
    19300/20000: d_loss: 0.7266,  a_loss: 4.4396.  (3.4 min)
    19350/20000: d_loss: 0.3673,  a_loss: 1.9860.  (3.4 min)
    19400/20000: d_loss: 0.3131,  a_loss: 1.8036.  (3.4 min)
    19450/20000: d_loss: 0.4588,  a_loss: 1.5795.  (3.4 min)
    19500/20000: d_loss: 0.6336,  a_loss: 1.3626.  (3.4 min)
    19550/20000: d_loss: 0.3585,  a_loss: 1.4615.  (3.5 min)
    19600/20000: d_loss: 0.5399,  a_loss: 2.2818.  (3.4 min)
    19650/20000: d_loss: 0.3376,  a_loss: 2.0284.  (3.4 min)
    19700/20000: d_loss: 0.4573,  a_loss: 1.4667.  (3.4 min)
    19750/20000: d_loss: 0.6182,  a_loss: 1.2988.  (3.4 min)
    19800/20000: d_loss: 0.4906,  a_loss: 1.4840.  (3.5 min)
    19850/20000: d_loss: 0.3601,  a_loss: 1.8660.  (3.4 min)
    19900/20000: d_loss: 0.3246,  a_loss: 2.0607.  (3.4 min)
    19950/20000: d_loss: 0.4495,  a_loss: 1.4924.  (3.4 min)
    20000/20000: d_loss: 0.3448,  a_loss: 2.3194.  (3.4 min)


Wir können die Entwicklung der Treffsicherheit der Detektivin (des discriminator-NNs) darstellen (Plot links). Es wird deutlich, dass die Treffsicherheit abnimmt mit fortschreitenden Iterationschritten, aber es gibt auch eine sehr große Varianz in der Trefferrate.

Auf der anderen Seite wird der Fortschritt der Fälscherin bzw. des generator-NN (Plot rechts) gemessen an der Fähigkeit, das discriminator-NN von der Echtheit der Fälschungen zu überzeugen. Auch hier zeigt sich eine große Varianz, aber ein Fortschritt in die richtige Richtung (bessere Fäschungen).


```python
plt.figure(1, figsize=(12, 8))
plt.subplot(121)
plt.plot(d_losses, color='red')
plt.xlabel('epochs')
plt.ylabel('discriminant losses')
plt.subplot(122)
plt.plot(a_losses)
plt.xlabel('epochs')
plt.ylabel('adversary losses')
plt.show()
```


    
![png](bt_gan_files/bt_gan_28_0.png)
    


## Ergebnisse und Diskussion <a class="anchor" id="ergebnisseunddiskussion"></a>


Zum Ende eine Sichtung der Ergebnisse. Die ersten Versuche der Fälscherin/des generator-NN haben wenig Ähnlichkeit mit den originalen Bildern, zeigen aber grundsätzliche Eigenschaften eines Porträts: Schulterpartie, Hintergrund, Farbe der Haut, grundsätzliche Position von Haaren, Mund und Augen.

<table><tr>
<td> <img src="img/generated_0.png" alt="50 Iterationen" style="width: 250px;"/> </td>
<td> <img src="img/generated_1.png" alt="100 Iterationen" style="width: 250px;"/> </td>
<td> <img src="img/generated_4.png" alt="250 Iterationen" style="width: 250px;"/> </td>
<td> <img src="img/generated_9.png" alt="500 Iterationen" style="width: 250px;"/> </td>
</tr></table>

Nach jeweils 5.000, 10.000, 15.000 Iterationen und zum Ende bei etwa 20.000 Iterationen stimmt die grundsätzliche Struktur der generierten Bilder ziemlich gut. Nach 10.000 Iterationen verbessern sich die Bilder kaum noch:

<table><tr>
<td> <img src="img/generated_99.png" alt="5.000 Iterationen" style="width: 250px;"/> </td>
<td> <img src="img/generated_199.png" alt="10.000 Iterationen" style="width: 250px;"/> </td>
<td> <img src="img/generated_299.png" alt="15.000 Iterationen" style="width: 250px;"/> </td>
<td> <img src="img/generated_399.png" alt="20.000 Iterationen" style="width: 250px;"/> </td>
</tr></table>

Es wird aber auch deutlich, dass die Bilder weit entfernt sind von echt-wirkenden "Deep Fakes": Zum Teil stimmt die Anzahl von Augen/Mündern nicht, schwarz-weiß-Bilder und Farbfotos werden nicht unterschieden vom GAN, Brillen stellen ein Problem dar usw.

Mögliche Ansätze, um diese Probleme zu lösen, sind: alle Bilder nur schwarz-weiß berücksichtigen, größeren Datensatz verwenden oder den Datensatz künstlich erweitern (z.B. durch Spiegeln der Aufnahmen).

Ein weiterer wichtiger Aspekt, der schon jetzt deutlich wird: die "besten" generierten Bilder repräsentieren diese, die im ursprünglichen Datensatz am häufigsten vorhanden sind. Mit anderen Worten, *weiße*, meist männliche Menschen mit Kurzhaarschnitten. Brillen, Langhaarfrisuren und nicht-*weiße* Hautfarben stellen das NN vor Schwierigkeiten auf Grund der Zusammensetzung des Datensatzes.


```python
# dieser Code erzeugt ein gif aus den iterativen Ergebnissen -- wird aber sehr groß! >100MB
import imageio
import shutil
import os
import glob
images_to_gif = []
RES_DIR = 'res2/'
files = list(filter(os.path.isfile, glob.glob(RES_DIR + "*")))
# files = os.listdir(RES_DIR)
files.sort(key=lambda x: os.path.getmtime(x))
for filename in files:
    images_to_gif.append(imageio.imread(filename))
imageio.mimsave('training.gif', images_to_gif)
# shutil.rmtree(RES_DIR)
```


```python
# run "jupyter nbconvert --to markdown bt_gan.ipynb" in terminal
# this makes md file of notebook
# rename to README.md and add-commit-push
```
