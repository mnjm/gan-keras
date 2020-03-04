from __future__ import print_function, division
from cgan_utils import pre_train_discriminator, train_gan
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LeakyReLU, Flatten, Conv2DTranspose
from tensorflow.keras.layers import Input, Embedding, Concatenate, Reshape, BatchNormalization, ReLU
from matplotlib import pyplot as plt
# from keras.datasets.fashion_mnist import load_data
from keras.datasets.mnist import load_data

np.random.seed(25)

def get_mnist_data():
    (train_X, train_Y), (_, _) = load_data()
    train_X = np.expand_dims(train_X, axis=-1)
    # scale to -1 to +1
    train_X = train_X.astype(np.float)
    train_X = (train_X - 127.5) / 127.5
    return train_X, train_Y

def plot_mnist_data(real_X,rows,cols):
    x_shape = real_X.shape[1:-1]
    for i in range(rows * cols):
        plt.subplot(rows,cols,1+i)
        plt.axis('off')
        sample = real_X[i]
        sample = sample.reshape(x_shape)
        sample = ((sample + 1) / 2) * 255.0
        sample = sample.astype(np.uint8)
        plt.imshow(sample, cmap='gray')
    plt.show()

def define_discriminator(inp_shape, n_classes):
    # Label embedding
    label_inp = Input(shape=(1,), name='label_input')
    l = Embedding(n_classes, 50) (label_inp)
    l = Dense(784) (l)
    l = Reshape(inp_shape) (l)
    # Image input
    img_inp = Input(shape=inp_shape, name='image_input')
    l = Concatenate() ([img_inp, l])
    l = BatchNormalization(axis=-1) (l)
    l = Conv2D(64, (3,3), strides=(2,2), padding='same') (l)
    l = BatchNormalization(axis=-1) (l)
    l = LeakyReLU(alpha=0.2) (l)
    l = Dropout(0.4) (l)
    l = Conv2D(64, (3,3), strides=(2,2), padding='same') (l)
    l = BatchNormalization(axis=-1) (l)
    l = LeakyReLU(alpha=0.2) (l)
    l = Dropout(0.4) (l)
    l = Flatten() (l)
    l = Dense(1, activation='sigmoid', name='disc_out') (l)
    model = Model([img_inp, label_inp], l, name="Discriminator")
    optimzer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimzer, metrics=['accuracy'])
    return model

def define_generator(latent_dim, n_classes):
    # Note: Model is not compiled with optimizer
    # Label embedding
    label_inp = Input(shape=(1,), name='label_input')
    l = Embedding(n_classes, 50) (label_inp)
    l = Dense(7*7*1) (l)
    l = Reshape((7,7,1)) (l)
    l = BatchNormalization(axis=-1) (l)
    # Latent input
    latent_inp = Input(shape=(latent_dim,), name='latent_input')
    gen = Dense(128*7*7) (latent_inp)
    gen = BatchNormalization(axis=-1) (gen)
    # gen = LeakyReLU(alpha=0.2) (gen)
    gen = ReLU() (gen)
    gen = Reshape((7,7,128)) (gen)
    l = Concatenate() ([gen, l])
    # upscale to 14x14
    l = Conv2DTranspose(128,(4,4),strides=(2,2), padding='same') (l)
    l = BatchNormalization(axis=-1) (l)
    # l = LeakyReLU(alpha=0.2) (l)
    l = ReLU() (l)
    # upscale to 28x28
    l = Conv2DTranspose(128,(4,4),strides=(2,2), padding='same') (l)
    l = BatchNormalization(axis=-1) (l)
    # l = LeakyReLU(alpha=0.2) (l)
    l = ReLU() (l)
    l = Conv2DTranspose(1,(7,7),activation='tanh', padding='same') (l)
    model = Model([latent_inp, label_inp], l, name="Generator")
    return model

def define_gan(gen_model, dis_model):
    # Make weights in the discriminator not trainable
    dis_model.trainable = False
    # Note: traninable property doesn't impact after the model is compiled. in define_discriminator
    # the model is compiled w/o reseting the trainable property(default True), so when 
    # train_on_batch is called with discrimator's weights will be updated. in GAN model(below), 
    # discrimator model's trainable property is reset before compiling the model so the 
    # discrimator's weights will not be updated.
    gen_latent_inp, gen_label_inp = gen_model.input
    gen_img_out = gen_model.output
    # connect gen image out and label input to discriminator
    gan_out = dis_model([gen_img_out, gen_label_inp])
    # create model with latent noise, label as input with d(g(z)) as output
    model = Model([gen_latent_inp, gen_label_inp], gan_out, name="GAN")
    # Adam with lr 0.0002 and momentum = 0.5
    optimzer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimzer)
    return model

if __name__ == '__main__':

    # get MNIST data
    real_data = get_mnist_data()
    n_classes = 10
    n_batchs = 256
    x_shape = real_data[0].shape[1:]
    latent_dim = 100
    n_epochs = 1000

    # plot_mnist_data(real_data[0], 10,10)

    dis_model = define_discriminator(x_shape, n_classes)
    dis_model.summary()
    print()
    # print('Pre-training Discriminator Model with real and noise samples')
    # history = pre_train_discriminator(dis_model, real_data, n_iter=100, n_batch=n_batchs, 
    #         inp_shape=x_shape, n_classes=n_classes)
    # plt.plot(history)
    # plt.xlabel('n_iter')
    # plt.ylabel('acc')
    # plt.title('Pretraining discriminator acc curve')
    # plt.legend(['real', 'fake/noise'])
    # plt.show()

    gen_model = define_generator(latent_dim, n_classes)
    gen_model.summary()

    gan_model = define_gan(gen_model, dis_model)
    gan_model.summary()

    train_gan(gen_model, dis_model, gan_model, real_data, latent_dim, n_classes=n_classes,
            n_epochs=n_epochs, n_batch=n_batchs, debug=True, log_file_name='mnist_digit_cgan')
