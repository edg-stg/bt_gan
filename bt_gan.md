# bt_gan: "deep fakes" aus MdB-Porträts

Adaptiert von https://www.kaggle.com/nageshsingh/generate-realistic-human-face-using-gan

Bildbasis: https://bilddatenbank.bundestag.de/search/picture-result?query=&filterQuery%5Bereignis%5D%5B0%5D=Portr%C3%A4t%2FPortrait&sortVal=3

<table><tr>
<td> <img src="img/original_mdb_sample25.png" alt="original" style="width: 350px;"/> </td>
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

    100%|██████████| 1684/1684 [00:01<00:00, 1297.01it/s]



```python
images = np.array(images) / 255
print(images.shape)

np.random.shuffle(images)
```

    (1684, 128, 128, 3)



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


![png](bt_gan_files/bt_gan_12_0.png)


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
batch_size = 8

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

        print('%d/%d: d_loss: %.4f,  a_loss: %.4f.  (%.1f sec)' % (step + 1, iters, d_loss, a_loss, time.time() - start_time))

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

    50/20000: d_loss: 0.6287,  a_loss: 1.2977.  (1.2 sec)
    100/20000: d_loss: 0.6604,  a_loss: 0.7833.  (1.2 sec)
    150/20000: d_loss: 0.6987,  a_loss: 0.7448.  (1.2 sec)
    200/20000: d_loss: 0.6707,  a_loss: 0.7626.  (1.2 sec)
    250/20000: d_loss: 0.6989,  a_loss: 0.8521.  (1.2 sec)
    300/20000: d_loss: 0.6814,  a_loss: 0.8524.  (1.3 sec)
    350/20000: d_loss: 0.6496,  a_loss: 0.9053.  (1.2 sec)
    400/20000: d_loss: 0.6532,  a_loss: 0.9889.  (1.2 sec)
    450/20000: d_loss: 0.6818,  a_loss: 0.7508.  (1.2 sec)
    500/20000: d_loss: 0.6373,  a_loss: 1.0637.  (1.2 sec)
    550/20000: d_loss: 0.6782,  a_loss: 1.2143.  (1.2 sec)
    600/20000: d_loss: 0.6722,  a_loss: 0.8336.  (1.2 sec)
    650/20000: d_loss: 0.5994,  a_loss: 1.8455.  (1.2 sec)
    700/20000: d_loss: 0.6436,  a_loss: 0.9319.  (1.2 sec)
    750/20000: d_loss: 0.6568,  a_loss: 0.9417.  (1.2 sec)
    800/20000: d_loss: 0.5270,  a_loss: 1.5665.  (1.3 sec)
    850/20000: d_loss: 0.5354,  a_loss: 0.8269.  (1.3 sec)
    900/20000: d_loss: 0.4796,  a_loss: 1.6908.  (1.2 sec)
    950/20000: d_loss: 0.5889,  a_loss: 1.0138.  (1.2 sec)
    1000/20000: d_loss: 0.4900,  a_loss: 1.4149.  (1.2 sec)
    1050/20000: d_loss: 0.5827,  a_loss: 1.1486.  (1.3 sec)
    1100/20000: d_loss: 0.7489,  a_loss: 0.9996.  (1.2 sec)
    1150/20000: d_loss: 0.6256,  a_loss: 1.4069.  (1.2 sec)
    1200/20000: d_loss: 0.6601,  a_loss: 1.3182.  (1.2 sec)
    1250/20000: d_loss: 0.5463,  a_loss: 1.0090.  (1.2 sec)
    1300/20000: d_loss: 0.7457,  a_loss: 0.8660.  (1.2 sec)
    1350/20000: d_loss: 0.4346,  a_loss: 1.1132.  (1.2 sec)
    1400/20000: d_loss: 0.5583,  a_loss: 0.9749.  (1.2 sec)
    1450/20000: d_loss: 0.6307,  a_loss: 1.5054.  (1.2 sec)
    1500/20000: d_loss: 0.5142,  a_loss: 1.6439.  (1.2 sec)
    1550/20000: d_loss: 0.5928,  a_loss: 0.8013.  (1.2 sec)
    1600/20000: d_loss: 0.8461,  a_loss: 1.2268.  (1.2 sec)
    1650/20000: d_loss: 0.5100,  a_loss: 1.0805.  (1.2 sec)
    1700/20000: d_loss: 0.5713,  a_loss: 1.3586.  (1.2 sec)
    1750/20000: d_loss: 0.6261,  a_loss: 1.5731.  (1.2 sec)
    1800/20000: d_loss: 0.5758,  a_loss: 1.3044.  (1.2 sec)
    1850/20000: d_loss: 0.5296,  a_loss: 1.7260.  (1.3 sec)
    1900/20000: d_loss: 0.6350,  a_loss: 0.7884.  (1.2 sec)
    1950/20000: d_loss: 0.5583,  a_loss: 1.2924.  (1.2 sec)
    2000/20000: d_loss: 0.4548,  a_loss: 1.0862.  (1.2 sec)
    2050/20000: d_loss: 0.6129,  a_loss: 1.6679.  (1.2 sec)
    2100/20000: d_loss: 0.5717,  a_loss: 1.1209.  (1.4 sec)
    2150/20000: d_loss: 0.5378,  a_loss: 1.2406.  (1.2 sec)
    2200/20000: d_loss: 0.5154,  a_loss: 1.0600.  (1.2 sec)
    2250/20000: d_loss: 0.7385,  a_loss: 1.1649.  (1.2 sec)
    2300/20000: d_loss: 0.5965,  a_loss: 1.4309.  (1.3 sec)
    2350/20000: d_loss: 0.4785,  a_loss: 0.8350.  (1.2 sec)
    2400/20000: d_loss: 0.5949,  a_loss: 1.1900.  (1.2 sec)
    2450/20000: d_loss: 0.6854,  a_loss: 1.2946.  (1.2 sec)
    2500/20000: d_loss: 0.5482,  a_loss: 1.7755.  (1.2 sec)
    2550/20000: d_loss: 0.5889,  a_loss: 1.1000.  (1.2 sec)
    2600/20000: d_loss: 0.6801,  a_loss: 1.4864.  (1.2 sec)
    2650/20000: d_loss: 0.4706,  a_loss: 1.1469.  (1.2 sec)
    2700/20000: d_loss: 0.6356,  a_loss: 1.2925.  (1.2 sec)
    2750/20000: d_loss: 0.6582,  a_loss: 1.1597.  (1.2 sec)
    2800/20000: d_loss: 0.4907,  a_loss: 0.9981.  (1.2 sec)
    2850/20000: d_loss: 0.5291,  a_loss: 0.9691.  (1.2 sec)
    2900/20000: d_loss: 0.6074,  a_loss: 1.4349.  (1.2 sec)
    2950/20000: d_loss: 0.6791,  a_loss: 1.4812.  (1.2 sec)
    3000/20000: d_loss: 0.5103,  a_loss: 1.6014.  (1.2 sec)
    3050/20000: d_loss: 0.5565,  a_loss: 1.3531.  (1.2 sec)
    3100/20000: d_loss: 0.4701,  a_loss: 1.3119.  (1.2 sec)
    3150/20000: d_loss: 0.6719,  a_loss: 1.4191.  (1.4 sec)
    3200/20000: d_loss: 0.4920,  a_loss: 1.0538.  (1.2 sec)
    3250/20000: d_loss: 0.6712,  a_loss: 1.0715.  (1.2 sec)
    3300/20000: d_loss: 0.4986,  a_loss: 1.1814.  (1.2 sec)
    3350/20000: d_loss: 0.7234,  a_loss: 2.4749.  (1.2 sec)
    3400/20000: d_loss: 0.7416,  a_loss: 0.9895.  (1.2 sec)
    3450/20000: d_loss: 0.4685,  a_loss: 1.2963.  (1.2 sec)
    3500/20000: d_loss: 0.5587,  a_loss: 0.7424.  (1.2 sec)
    3550/20000: d_loss: 0.7439,  a_loss: 1.4102.  (1.2 sec)
    3600/20000: d_loss: 0.6054,  a_loss: 1.9089.  (1.2 sec)
    3650/20000: d_loss: 0.6213,  a_loss: 0.8041.  (1.2 sec)
    3700/20000: d_loss: 0.7022,  a_loss: 1.4546.  (1.2 sec)
    3750/20000: d_loss: 0.7582,  a_loss: 1.1137.  (1.2 sec)
    3800/20000: d_loss: 0.6952,  a_loss: 1.2461.  (1.2 sec)
    3850/20000: d_loss: 0.6422,  a_loss: 1.2941.  (1.2 sec)
    3900/20000: d_loss: 0.4799,  a_loss: 0.6992.  (1.2 sec)
    3950/20000: d_loss: 0.5411,  a_loss: 1.0889.  (1.2 sec)
    4000/20000: d_loss: 0.6201,  a_loss: 1.4487.  (1.2 sec)
    4050/20000: d_loss: 0.5692,  a_loss: 1.1334.  (1.2 sec)
    4100/20000: d_loss: 0.9069,  a_loss: 1.3471.  (1.2 sec)
    4150/20000: d_loss: 0.5393,  a_loss: 1.4425.  (1.2 sec)
    4200/20000: d_loss: 0.7078,  a_loss: 0.9127.  (1.3 sec)
    4250/20000: d_loss: 0.6229,  a_loss: 1.2441.  (1.2 sec)
    4300/20000: d_loss: 0.5692,  a_loss: 1.2266.  (1.2 sec)
    4350/20000: d_loss: 0.5376,  a_loss: 0.8703.  (1.2 sec)
    4400/20000: d_loss: 0.5638,  a_loss: 1.0013.  (1.2 sec)
    4450/20000: d_loss: 0.6274,  a_loss: 1.1578.  (1.2 sec)
    4500/20000: d_loss: 0.6496,  a_loss: 0.9861.  (1.2 sec)
    4550/20000: d_loss: 0.4727,  a_loss: 1.1598.  (1.2 sec)
    4600/20000: d_loss: 0.6001,  a_loss: 1.3065.  (1.2 sec)
    4650/20000: d_loss: 0.5590,  a_loss: 0.9141.  (1.2 sec)
    4700/20000: d_loss: 0.6575,  a_loss: 0.9347.  (1.2 sec)
    4750/20000: d_loss: 0.6618,  a_loss: 1.3349.  (1.2 sec)
    4800/20000: d_loss: 0.5408,  a_loss: 1.4666.  (1.2 sec)
    4850/20000: d_loss: 0.5470,  a_loss: 1.0137.  (1.2 sec)
    4900/20000: d_loss: 0.9041,  a_loss: 1.7210.  (1.3 sec)
    4950/20000: d_loss: 0.7122,  a_loss: 1.0145.  (1.2 sec)
    5000/20000: d_loss: 0.6434,  a_loss: 1.0754.  (1.2 sec)
    5050/20000: d_loss: 0.4499,  a_loss: 0.9924.  (1.2 sec)
    5100/20000: d_loss: 0.5288,  a_loss: 1.0681.  (1.2 sec)
    5150/20000: d_loss: 0.6112,  a_loss: 1.1438.  (1.2 sec)
    5200/20000: d_loss: 0.5761,  a_loss: 1.2106.  (1.2 sec)
    5250/20000: d_loss: 0.5889,  a_loss: 1.4503.  (1.3 sec)
    5300/20000: d_loss: 0.5530,  a_loss: 1.0838.  (1.2 sec)
    5350/20000: d_loss: 0.7119,  a_loss: 1.1005.  (1.2 sec)
    5400/20000: d_loss: 0.4923,  a_loss: 1.2095.  (1.2 sec)
    5450/20000: d_loss: 0.4937,  a_loss: 1.2693.  (1.2 sec)
    5500/20000: d_loss: 0.6193,  a_loss: 1.5046.  (1.2 sec)
    5550/20000: d_loss: 0.6170,  a_loss: 1.4371.  (1.2 sec)
    5600/20000: d_loss: 0.4238,  a_loss: 0.9708.  (1.2 sec)
    5650/20000: d_loss: 0.6866,  a_loss: 1.1022.  (1.2 sec)
    5700/20000: d_loss: 0.4869,  a_loss: 1.0816.  (1.2 sec)
    5750/20000: d_loss: 0.7371,  a_loss: 1.3889.  (1.2 sec)
    5800/20000: d_loss: 0.5256,  a_loss: 1.3655.  (1.2 sec)
    5850/20000: d_loss: 0.5149,  a_loss: 1.3689.  (1.2 sec)
    5900/20000: d_loss: 0.6098,  a_loss: 1.0051.  (1.2 sec)
    5950/20000: d_loss: 0.6179,  a_loss: 1.4412.  (1.2 sec)
    6000/20000: d_loss: 0.5675,  a_loss: 1.0967.  (1.2 sec)
    6050/20000: d_loss: 0.5532,  a_loss: 1.2430.  (1.2 sec)
    6100/20000: d_loss: 0.5362,  a_loss: 1.3045.  (1.2 sec)
    6150/20000: d_loss: 0.6332,  a_loss: 1.2531.  (1.2 sec)
    6200/20000: d_loss: 0.7150,  a_loss: 1.4664.  (1.2 sec)
    6250/20000: d_loss: 0.7995,  a_loss: 1.4649.  (1.2 sec)
    6300/20000: d_loss: 0.6166,  a_loss: 1.3036.  (1.3 sec)
    6350/20000: d_loss: 0.7923,  a_loss: 1.6460.  (1.2 sec)
    6400/20000: d_loss: 0.6614,  a_loss: 0.8927.  (1.2 sec)
    6450/20000: d_loss: 0.7247,  a_loss: 0.8985.  (1.2 sec)
    6500/20000: d_loss: 0.6687,  a_loss: 0.6691.  (1.2 sec)
    6550/20000: d_loss: 0.5736,  a_loss: 1.2243.  (1.2 sec)
    6600/20000: d_loss: 0.8232,  a_loss: 1.7165.  (1.2 sec)
    6650/20000: d_loss: 0.5216,  a_loss: 1.5383.  (1.2 sec)
    6700/20000: d_loss: 0.4255,  a_loss: 1.4367.  (1.2 sec)
    6750/20000: d_loss: 0.6481,  a_loss: 1.4156.  (1.2 sec)
    6800/20000: d_loss: 0.6743,  a_loss: 1.4776.  (1.2 sec)
    6850/20000: d_loss: 0.5711,  a_loss: 0.8392.  (1.2 sec)
    6900/20000: d_loss: 0.8681,  a_loss: 1.5699.  (1.2 sec)
    6950/20000: d_loss: 0.5864,  a_loss: 1.4245.  (1.2 sec)
    7000/20000: d_loss: 0.4628,  a_loss: 1.8255.  (1.2 sec)
    7050/20000: d_loss: 0.4458,  a_loss: 1.1195.  (1.2 sec)
    7100/20000: d_loss: 0.6672,  a_loss: 1.2508.  (1.2 sec)
    7150/20000: d_loss: 0.6233,  a_loss: 1.6547.  (1.2 sec)
    7200/20000: d_loss: 0.8403,  a_loss: 1.2947.  (1.2 sec)
    7250/20000: d_loss: 0.4757,  a_loss: 1.1593.  (1.2 sec)
    7300/20000: d_loss: 0.5595,  a_loss: 1.0865.  (1.2 sec)
    7350/20000: d_loss: 0.6944,  a_loss: 1.3175.  (1.3 sec)
    7400/20000: d_loss: 0.5784,  a_loss: 1.6822.  (1.2 sec)
    7450/20000: d_loss: 0.6825,  a_loss: 1.2450.  (1.2 sec)
    7500/20000: d_loss: 0.9964,  a_loss: 0.8392.  (1.2 sec)
    7550/20000: d_loss: 0.5780,  a_loss: 1.3926.  (1.2 sec)
    7600/20000: d_loss: 0.6413,  a_loss: 0.9915.  (1.2 sec)
    7650/20000: d_loss: 0.4650,  a_loss: 0.7640.  (1.2 sec)
    7700/20000: d_loss: 0.5546,  a_loss: 0.8803.  (1.2 sec)
    7750/20000: d_loss: 0.6050,  a_loss: 1.4970.  (1.2 sec)
    7800/20000: d_loss: 0.5705,  a_loss: 1.0648.  (1.2 sec)
    7850/20000: d_loss: 0.4617,  a_loss: 1.1824.  (1.2 sec)
    7900/20000: d_loss: 0.7639,  a_loss: 1.5874.  (1.3 sec)
    7950/20000: d_loss: 0.5396,  a_loss: 1.1066.  (1.2 sec)
    8000/20000: d_loss: 0.9064,  a_loss: 1.6833.  (1.2 sec)
    8050/20000: d_loss: 0.5777,  a_loss: 1.9350.  (1.2 sec)
    8100/20000: d_loss: 0.6763,  a_loss: 1.1116.  (1.2 sec)
    8150/20000: d_loss: 0.7041,  a_loss: 1.7825.  (1.2 sec)
    8200/20000: d_loss: 0.3815,  a_loss: 1.3900.  (1.3 sec)
    8250/20000: d_loss: 0.4203,  a_loss: 1.2414.  (1.2 sec)
    8300/20000: d_loss: 0.5686,  a_loss: 1.0368.  (1.2 sec)
    8350/20000: d_loss: 0.5336,  a_loss: 2.6320.  (1.2 sec)
    8400/20000: d_loss: 0.6706,  a_loss: 2.0689.  (1.3 sec)
    8450/20000: d_loss: 0.4677,  a_loss: 1.4437.  (1.2 sec)
    8500/20000: d_loss: 0.5184,  a_loss: 1.6202.  (1.2 sec)
    8550/20000: d_loss: 0.5919,  a_loss: 1.1938.  (1.2 sec)
    8600/20000: d_loss: 0.6391,  a_loss: 1.3203.  (1.2 sec)
    8650/20000: d_loss: 0.5026,  a_loss: 1.2251.  (1.2 sec)
    8700/20000: d_loss: 0.6757,  a_loss: 0.9599.  (1.2 sec)
    8750/20000: d_loss: 0.6334,  a_loss: 1.2744.  (1.2 sec)
    8800/20000: d_loss: 0.5686,  a_loss: 1.4290.  (1.2 sec)
    8850/20000: d_loss: 0.4826,  a_loss: 1.4788.  (1.2 sec)
    8900/20000: d_loss: 0.4824,  a_loss: 1.2711.  (1.2 sec)
    8950/20000: d_loss: 0.6127,  a_loss: 1.2879.  (1.2 sec)
    9000/20000: d_loss: 0.8018,  a_loss: 0.7436.  (1.2 sec)
    9050/20000: d_loss: 0.5070,  a_loss: 1.5593.  (1.2 sec)
    9100/20000: d_loss: 0.4605,  a_loss: 2.7006.  (1.2 sec)
    9150/20000: d_loss: 0.4874,  a_loss: 1.7328.  (1.2 sec)
    9200/20000: d_loss: 0.6160,  a_loss: 1.2873.  (1.2 sec)
    9250/20000: d_loss: 0.6768,  a_loss: 2.2836.  (1.2 sec)
    9300/20000: d_loss: 0.6328,  a_loss: 1.0180.  (1.3 sec)
    9350/20000: d_loss: 0.4871,  a_loss: 2.7134.  (1.2 sec)
    9400/20000: d_loss: 0.5624,  a_loss: 1.4773.  (1.2 sec)
    9450/20000: d_loss: 0.7297,  a_loss: 1.4779.  (1.4 sec)
    9500/20000: d_loss: 0.4782,  a_loss: 1.0438.  (1.2 sec)
    9550/20000: d_loss: 0.4002,  a_loss: 1.8306.  (1.2 sec)
    9600/20000: d_loss: 0.3865,  a_loss: 1.6500.  (1.3 sec)
    9650/20000: d_loss: 0.5102,  a_loss: 1.0329.  (1.3 sec)
    9700/20000: d_loss: 0.3373,  a_loss: 2.0104.  (1.2 sec)
    9750/20000: d_loss: 0.5942,  a_loss: 1.6440.  (1.2 sec)
    9800/20000: d_loss: 0.5709,  a_loss: 0.9394.  (1.2 sec)
    9850/20000: d_loss: 0.5614,  a_loss: 1.1046.  (1.2 sec)
    9900/20000: d_loss: 0.6466,  a_loss: 0.9627.  (1.2 sec)
    9950/20000: d_loss: 0.7359,  a_loss: 1.6719.  (1.3 sec)
    10000/20000: d_loss: 0.5412,  a_loss: 1.5995.  (1.2 sec)
    10050/20000: d_loss: 0.4145,  a_loss: 1.5412.  (1.2 sec)
    10100/20000: d_loss: 0.5611,  a_loss: 1.5298.  (1.2 sec)
    10150/20000: d_loss: 0.5021,  a_loss: 0.9667.  (1.2 sec)
    10200/20000: d_loss: 0.4025,  a_loss: 1.3572.  (1.2 sec)
    10250/20000: d_loss: 0.5602,  a_loss: 1.1702.  (1.2 sec)
    10300/20000: d_loss: 0.4277,  a_loss: 1.3542.  (1.2 sec)
    10350/20000: d_loss: 0.4992,  a_loss: 1.2374.  (1.2 sec)
    10400/20000: d_loss: 0.5320,  a_loss: 1.8252.  (1.2 sec)
    10450/20000: d_loss: 0.5214,  a_loss: 1.6782.  (1.2 sec)
    10500/20000: d_loss: 0.5447,  a_loss: 1.0129.  (1.3 sec)
    10550/20000: d_loss: 0.5248,  a_loss: 1.5872.  (1.2 sec)
    10600/20000: d_loss: 0.5069,  a_loss: 2.7275.  (1.3 sec)
    10650/20000: d_loss: 0.3582,  a_loss: 1.9633.  (1.2 sec)
    10700/20000: d_loss: 0.5886,  a_loss: 1.8838.  (1.2 sec)
    10750/20000: d_loss: 0.4079,  a_loss: 1.2656.  (1.2 sec)
    10800/20000: d_loss: 0.4114,  a_loss: 1.8875.  (1.2 sec)
    10850/20000: d_loss: 0.4399,  a_loss: 1.7400.  (1.2 sec)
    10900/20000: d_loss: 0.4663,  a_loss: 1.5545.  (1.2 sec)
    10950/20000: d_loss: 0.3783,  a_loss: 1.7732.  (1.2 sec)
    11000/20000: d_loss: 0.4122,  a_loss: 1.4444.  (1.3 sec)
    11050/20000: d_loss: 0.5190,  a_loss: 2.4003.  (1.2 sec)
    11100/20000: d_loss: 0.4633,  a_loss: 1.5554.  (1.2 sec)
    11150/20000: d_loss: 0.3851,  a_loss: 1.5340.  (1.3 sec)
    11200/20000: d_loss: 0.8338,  a_loss: 2.4672.  (1.2 sec)
    11250/20000: d_loss: 0.5067,  a_loss: 2.2984.  (1.2 sec)
    11300/20000: d_loss: 0.3029,  a_loss: 1.8663.  (1.2 sec)
    11350/20000: d_loss: 0.4950,  a_loss: 1.1779.  (1.2 sec)
    11400/20000: d_loss: 0.4697,  a_loss: 1.4663.  (1.2 sec)
    11450/20000: d_loss: 0.4371,  a_loss: 2.7534.  (1.2 sec)
    11500/20000: d_loss: 0.3452,  a_loss: 1.6145.  (1.2 sec)
    11550/20000: d_loss: 0.5239,  a_loss: 1.4360.  (1.3 sec)
    11600/20000: d_loss: 0.5727,  a_loss: 1.2143.  (1.2 sec)
    11650/20000: d_loss: 0.5650,  a_loss: 1.5989.  (1.2 sec)
    11700/20000: d_loss: 0.5013,  a_loss: 2.1552.  (1.2 sec)
    11750/20000: d_loss: 0.5205,  a_loss: 1.3716.  (1.2 sec)
    11800/20000: d_loss: 0.5270,  a_loss: 1.8184.  (1.3 sec)
    11850/20000: d_loss: 0.5716,  a_loss: 1.8264.  (1.2 sec)
    11900/20000: d_loss: 0.4187,  a_loss: 1.2685.  (1.2 sec)
    11950/20000: d_loss: 0.4636,  a_loss: 0.9642.  (1.2 sec)
    12000/20000: d_loss: 0.3572,  a_loss: 1.2474.  (1.2 sec)
    12050/20000: d_loss: 0.4229,  a_loss: 1.2722.  (1.2 sec)
    12100/20000: d_loss: 0.4248,  a_loss: 2.4078.  (1.2 sec)
    12150/20000: d_loss: 0.5996,  a_loss: 1.5037.  (1.2 sec)
    12200/20000: d_loss: 0.4589,  a_loss: 2.3634.  (1.3 sec)
    12250/20000: d_loss: 0.5177,  a_loss: 0.9501.  (1.3 sec)
    12300/20000: d_loss: 0.7536,  a_loss: 0.7184.  (1.2 sec)
    12350/20000: d_loss: 0.5984,  a_loss: 1.5237.  (1.2 sec)
    12400/20000: d_loss: 0.4155,  a_loss: 1.9863.  (1.2 sec)
    12450/20000: d_loss: 0.4764,  a_loss: 1.2643.  (1.2 sec)
    12500/20000: d_loss: 0.4712,  a_loss: 1.9195.  (1.2 sec)
    12550/20000: d_loss: 0.4310,  a_loss: 0.9470.  (1.3 sec)
    12600/20000: d_loss: 0.4297,  a_loss: 1.8257.  (1.3 sec)
    12650/20000: d_loss: 0.4939,  a_loss: 1.8632.  (1.2 sec)
    12700/20000: d_loss: 0.6569,  a_loss: 1.1090.  (1.2 sec)
    12750/20000: d_loss: 0.5424,  a_loss: 2.1429.  (1.2 sec)
    12800/20000: d_loss: 0.5708,  a_loss: 1.9859.  (1.2 sec)
    12850/20000: d_loss: 0.2529,  a_loss: 1.7331.  (1.2 sec)
    12900/20000: d_loss: 0.5960,  a_loss: 2.1073.  (1.2 sec)
    12950/20000: d_loss: 0.4527,  a_loss: 1.4875.  (1.2 sec)
    13000/20000: d_loss: 0.5987,  a_loss: 1.1537.  (1.2 sec)
    13050/20000: d_loss: 0.2295,  a_loss: 2.9723.  (1.2 sec)
    13100/20000: d_loss: 0.4018,  a_loss: 2.1339.  (1.2 sec)
    13150/20000: d_loss: 0.4912,  a_loss: 2.0195.  (1.3 sec)
    13200/20000: d_loss: 0.4546,  a_loss: 1.6337.  (1.2 sec)
    13250/20000: d_loss: 0.5020,  a_loss: 1.7029.  (1.2 sec)
    13300/20000: d_loss: 0.4297,  a_loss: 2.5946.  (1.2 sec)
    13350/20000: d_loss: 0.6182,  a_loss: 2.2821.  (1.2 sec)
    13400/20000: d_loss: 0.5501,  a_loss: 1.7094.  (1.2 sec)
    13450/20000: d_loss: 0.4859,  a_loss: 2.3445.  (1.2 sec)
    13500/20000: d_loss: 0.3264,  a_loss: 1.9319.  (1.2 sec)
    13550/20000: d_loss: 0.4441,  a_loss: 1.5146.  (1.2 sec)
    13600/20000: d_loss: 0.4582,  a_loss: 0.9649.  (1.3 sec)
    13650/20000: d_loss: 0.4613,  a_loss: 0.9185.  (1.3 sec)
    13700/20000: d_loss: 0.5311,  a_loss: 2.2768.  (1.2 sec)
    13750/20000: d_loss: 0.5604,  a_loss: 1.3415.  (1.3 sec)
    13800/20000: d_loss: 0.5868,  a_loss: 2.1776.  (1.2 sec)
    13850/20000: d_loss: 0.4322,  a_loss: 2.5758.  (1.2 sec)
    13900/20000: d_loss: 0.4012,  a_loss: 1.9788.  (1.2 sec)
    13950/20000: d_loss: 0.4593,  a_loss: 2.6030.  (1.2 sec)
    14000/20000: d_loss: 0.3492,  a_loss: 2.0546.  (1.2 sec)
    14050/20000: d_loss: 0.4355,  a_loss: 3.2340.  (1.2 sec)
    14100/20000: d_loss: 0.8453,  a_loss: 2.3388.  (1.2 sec)
    14150/20000: d_loss: 0.6979,  a_loss: 3.5366.  (1.2 sec)
    14200/20000: d_loss: 0.4048,  a_loss: 1.6975.  (1.2 sec)
    14250/20000: d_loss: 0.4160,  a_loss: 1.5006.  (1.2 sec)
    14300/20000: d_loss: 0.2969,  a_loss: 1.6976.  (1.2 sec)
    14350/20000: d_loss: 0.3223,  a_loss: 1.8087.  (1.2 sec)
    14400/20000: d_loss: 0.3334,  a_loss: 2.3374.  (1.2 sec)
    14450/20000: d_loss: 0.7832,  a_loss: 2.2323.  (1.2 sec)
    14500/20000: d_loss: 0.1855,  a_loss: 1.8227.  (1.3 sec)
    14550/20000: d_loss: 0.6364,  a_loss: 0.8945.  (1.2 sec)
    14600/20000: d_loss: 0.4111,  a_loss: 1.7039.  (1.2 sec)
    14650/20000: d_loss: 0.6796,  a_loss: 2.4875.  (1.2 sec)
    14700/20000: d_loss: 0.3671,  a_loss: 2.1516.  (1.3 sec)
    14750/20000: d_loss: 0.2426,  a_loss: 1.6288.  (1.2 sec)
    14800/20000: d_loss: 0.5459,  a_loss: 2.1025.  (1.2 sec)
    14850/20000: d_loss: 0.3719,  a_loss: 2.2471.  (1.2 sec)
    14900/20000: d_loss: 0.5124,  a_loss: 2.6817.  (1.2 sec)
    14950/20000: d_loss: 0.7121,  a_loss: 1.4420.  (1.3 sec)
    15000/20000: d_loss: 0.4583,  a_loss: 2.5084.  (1.2 sec)
    15050/20000: d_loss: 0.2839,  a_loss: 1.6832.  (1.2 sec)
    15100/20000: d_loss: 0.3751,  a_loss: 1.4758.  (1.2 sec)
    15150/20000: d_loss: 0.2411,  a_loss: 2.2686.  (1.2 sec)
    15200/20000: d_loss: 0.4766,  a_loss: 2.0219.  (1.2 sec)
    15250/20000: d_loss: 0.3566,  a_loss: 2.9308.  (1.2 sec)
    15300/20000: d_loss: 0.3036,  a_loss: 1.7999.  (1.2 sec)
    15350/20000: d_loss: 0.5371,  a_loss: 1.9768.  (1.2 sec)
    15400/20000: d_loss: 0.4623,  a_loss: 1.7363.  (1.2 sec)
    15450/20000: d_loss: 0.4735,  a_loss: 2.3519.  (1.2 sec)
    15500/20000: d_loss: 0.4012,  a_loss: 1.3967.  (1.2 sec)
    15550/20000: d_loss: 0.2180,  a_loss: 2.5948.  (1.2 sec)
    15600/20000: d_loss: 0.3993,  a_loss: 1.6863.  (1.2 sec)
    15650/20000: d_loss: 0.2079,  a_loss: 3.8036.  (1.2 sec)
    15700/20000: d_loss: 0.4332,  a_loss: 3.2329.  (1.2 sec)
    15750/20000: d_loss: 0.4101,  a_loss: 2.3470.  (1.3 sec)
    15800/20000: d_loss: 0.3342,  a_loss: 2.3342.  (1.2 sec)
    15850/20000: d_loss: 0.8230,  a_loss: 1.4502.  (1.2 sec)
    15900/20000: d_loss: 0.2965,  a_loss: 1.6779.  (1.2 sec)
    15950/20000: d_loss: 0.3420,  a_loss: 2.7044.  (1.2 sec)
    16000/20000: d_loss: 0.2422,  a_loss: 2.0792.  (1.3 sec)
    16050/20000: d_loss: 0.6233,  a_loss: 1.9121.  (1.2 sec)
    16100/20000: d_loss: 0.3175,  a_loss: 1.5261.  (1.2 sec)
    16150/20000: d_loss: 0.2771,  a_loss: 2.0496.  (1.2 sec)
    16200/20000: d_loss: 0.3937,  a_loss: 2.9074.  (1.2 sec)
    16250/20000: d_loss: 0.2206,  a_loss: 2.4817.  (1.3 sec)
    16300/20000: d_loss: 0.2996,  a_loss: 2.7736.  (1.2 sec)
    16350/20000: d_loss: 0.4124,  a_loss: 2.2350.  (1.2 sec)
    16400/20000: d_loss: 0.3414,  a_loss: 3.6739.  (1.2 sec)
    16450/20000: d_loss: 0.5984,  a_loss: 1.8361.  (1.2 sec)
    16500/20000: d_loss: 0.2385,  a_loss: 1.8763.  (1.2 sec)
    16550/20000: d_loss: 0.2783,  a_loss: 3.1888.  (1.2 sec)
    16600/20000: d_loss: 0.1708,  a_loss: 3.2342.  (1.2 sec)
    16650/20000: d_loss: 0.4095,  a_loss: 3.3246.  (1.2 sec)
    16700/20000: d_loss: 0.2542,  a_loss: 2.3731.  (1.2 sec)
    16750/20000: d_loss: 0.2029,  a_loss: 2.8437.  (1.2 sec)
    16800/20000: d_loss: 0.2326,  a_loss: 1.9934.  (1.4 sec)
    16850/20000: d_loss: 0.3635,  a_loss: 3.3362.  (1.2 sec)
    16900/20000: d_loss: 0.6277,  a_loss: 3.0204.  (1.2 sec)
    16950/20000: d_loss: 0.4994,  a_loss: 3.0782.  (1.2 sec)
    17000/20000: d_loss: 0.2692,  a_loss: 2.4810.  (1.2 sec)
    17050/20000: d_loss: 0.3344,  a_loss: 1.3564.  (1.3 sec)
    17100/20000: d_loss: 0.3624,  a_loss: 3.8014.  (1.2 sec)
    17150/20000: d_loss: 0.2968,  a_loss: 1.2399.  (1.2 sec)
    17200/20000: d_loss: 0.5631,  a_loss: 1.8414.  (1.2 sec)
    17250/20000: d_loss: 0.5183,  a_loss: 4.2546.  (1.2 sec)
    17300/20000: d_loss: 0.2296,  a_loss: 2.5374.  (1.3 sec)
    17350/20000: d_loss: 0.4768,  a_loss: 3.2803.  (1.2 sec)
    17400/20000: d_loss: 0.2802,  a_loss: 3.0303.  (1.2 sec)
    17450/20000: d_loss: 0.2555,  a_loss: 2.4287.  (1.2 sec)
    17500/20000: d_loss: 0.5402,  a_loss: 3.5774.  (1.2 sec)
    17550/20000: d_loss: 0.2333,  a_loss: 2.8287.  (1.2 sec)
    17600/20000: d_loss: 0.4267,  a_loss: 1.6947.  (1.3 sec)
    17650/20000: d_loss: 0.1848,  a_loss: 2.3651.  (1.2 sec)
    17700/20000: d_loss: 0.4431,  a_loss: 2.2925.  (1.2 sec)
    17750/20000: d_loss: 0.2098,  a_loss: 1.8839.  (1.2 sec)
    17800/20000: d_loss: 0.4572,  a_loss: 1.6587.  (1.2 sec)
    17850/20000: d_loss: 0.3611,  a_loss: 3.3621.  (1.3 sec)
    17900/20000: d_loss: 0.3897,  a_loss: 3.8500.  (1.2 sec)
    17950/20000: d_loss: 0.3814,  a_loss: 2.4131.  (1.2 sec)
    18000/20000: d_loss: 0.2494,  a_loss: 2.2243.  (1.2 sec)
    18050/20000: d_loss: 0.3305,  a_loss: 1.9838.  (1.2 sec)
    18100/20000: d_loss: 0.2605,  a_loss: 1.9827.  (1.2 sec)
    18150/20000: d_loss: 0.2090,  a_loss: 2.8602.  (1.2 sec)
    18200/20000: d_loss: 0.2282,  a_loss: 2.0197.  (1.2 sec)
    18250/20000: d_loss: 0.3789,  a_loss: 1.7844.  (1.2 sec)
    18300/20000: d_loss: 0.6525,  a_loss: 2.1435.  (1.2 sec)
    18350/20000: d_loss: 0.7019,  a_loss: 5.5726.  (1.2 sec)
    18400/20000: d_loss: 0.3336,  a_loss: 3.6454.  (1.2 sec)
    18450/20000: d_loss: 0.3237,  a_loss: 2.4596.  (1.2 sec)
    18500/20000: d_loss: 0.4002,  a_loss: 2.6676.  (1.2 sec)
    18550/20000: d_loss: 0.6006,  a_loss: 1.7994.  (1.2 sec)
    18600/20000: d_loss: 0.3789,  a_loss: 2.1232.  (1.2 sec)
    18650/20000: d_loss: 0.4609,  a_loss: 1.0611.  (1.2 sec)
    18700/20000: d_loss: 0.4707,  a_loss: 2.8704.  (1.2 sec)
    18750/20000: d_loss: 0.5981,  a_loss: 1.1099.  (1.2 sec)
    18800/20000: d_loss: 0.3050,  a_loss: 3.7593.  (1.2 sec)
    18850/20000: d_loss: 0.1026,  a_loss: 1.9586.  (1.3 sec)
    18900/20000: d_loss: 0.3333,  a_loss: 2.3542.  (1.3 sec)
    18950/20000: d_loss: 0.3911,  a_loss: 2.1405.  (1.2 sec)
    19000/20000: d_loss: 0.3687,  a_loss: 2.2626.  (1.3 sec)
    19050/20000: d_loss: 0.1838,  a_loss: 2.4488.  (1.2 sec)
    19100/20000: d_loss: 0.4359,  a_loss: 1.6662.  (1.2 sec)
    19150/20000: d_loss: 0.1793,  a_loss: 3.4552.  (1.2 sec)
    19200/20000: d_loss: 0.3713,  a_loss: 2.8097.  (1.2 sec)
    19250/20000: d_loss: 0.1733,  a_loss: 3.0816.  (1.2 sec)
    19300/20000: d_loss: 0.3727,  a_loss: 3.8677.  (1.2 sec)
    19350/20000: d_loss: 0.3117,  a_loss: 3.5184.  (1.2 sec)
    19400/20000: d_loss: 0.2529,  a_loss: 1.8962.  (1.2 sec)
    19450/20000: d_loss: 0.4423,  a_loss: 2.3841.  (1.2 sec)
    19500/20000: d_loss: 0.3350,  a_loss: 4.5622.  (1.2 sec)
    19550/20000: d_loss: 0.5112,  a_loss: 3.3635.  (1.2 sec)
    19600/20000: d_loss: 0.2512,  a_loss: 3.7205.  (1.2 sec)
    19650/20000: d_loss: 0.2837,  a_loss: 3.0916.  (1.2 sec)
    19700/20000: d_loss: 0.1819,  a_loss: 2.3800.  (1.2 sec)
    19750/20000: d_loss: 0.2532,  a_loss: 3.1655.  (1.2 sec)
    19800/20000: d_loss: 0.3984,  a_loss: 2.1843.  (1.2 sec)
    19850/20000: d_loss: 0.2055,  a_loss: 3.4721.  (1.2 sec)
    19900/20000: d_loss: 0.5445,  a_loss: 3.1390.  (1.2 sec)
    19950/20000: d_loss: 0.3109,  a_loss: 2.2375.  (1.3 sec)
    20000/20000: d_loss: 0.3055,  a_loss: 3.0131.  (1.2 sec)


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


![png](bt_gan_files/bt_gan_27_0.png)


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
