from __future__ import print_function
from gan_utils import pre_train_discriminator, train_gan
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LeakyReLU, Flatten, Conv2DTranspose, Reshape
from tensorflow.keras.datasets.mnist import load_data
from matplotlib import pyplot as plt

np.random.seed(25)

def get_mnist_data():
    (train_X, _), (_, _) = load_data()
    train_X = np.expand_dims(train_X, axis=-1)
    # scale to -1 to +1
    train_X = train_X.astype(np.float)
    train_X = (train_X - 127.5) / 127.5
    return train_X

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

def define_discriminator(inp_shape):
    model = Sequential(name = "Discriminator")
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=inp_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    optimzer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimzer, metrics=['accuracy'])
    return model

def define_generator(latent_dim):
    # Note: Model is not compiled with optimizer
    model = Sequential(name = 'Generator')
    # for initial 7x7 image
    n_nodes = 7 * 7 * 128
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7,7,128)))
    # upsample to 14x14
    model.add(Conv2DTranspose(128,(4,4),strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 28x28
    model.add(Conv2DTranspose(128,(4,4),strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # convert to single image
    model.add(Conv2DTranspose(1,(7,7),activation='tanh', padding='same'))
    return model

def define_gan(gen_model, dis_model):
    # Make weights in the discriminator not trainable
    dis_model.trainable = False
    # Note: traninable property doesn't impact after the model is compiled. in define_discriminator
    # the model is compiled w/o reseting the trainable property(default True), so when 
    # train_on_batch is called with discrimator's weights will be updated. in GAN model(below), 
    # discrimator model's trainable property is reset before compiling the model so the 
    # discrimator's weights will not be updated.
    model = Sequential(name = 'GAN')
    # Adding generator
    model.add(gen_model)
    # Adding discriminator
    model.add(dis_model)
    # Adam with lr 0.0002 and momentum = 0.5
    optimzer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimzer)
    return model

if __name__ == '__main__':

    # get MNIST data
    real_X = get_mnist_data()
    n_batchs = 256
    x_shape = real_X.shape[1:]
    latent_dim = 100
    n_epochs = 1000

    # plot_mnist_data(real_X, 10,10)

    dis_model = define_discriminator(x_shape)
    dis_model.summary()
    # print()
    print('Pre-training Discriminator Model with real and noise samples')
    history = pre_train_discriminator(dis_model, real_X, n_iter=100, n_batch=n_batchs, 
            inp_shape=x_shape)
    # plt.plot(history)
    # plt.xlabel('n_iter')
    # plt.ylabel('acc')
    # plt.title('Pretraining discriminator acc curve')
    # plt.legend(['real', 'fake/noise'])
    # plt.show()

    gen_model = define_generator(latent_dim)
    gen_model.summary()
    print()

    gan_model = define_gan(gen_model, dis_model)
    gan_model.summary()
    print()

    train_gan(gen_model, dis_model, gan_model, real_X, latent_dim, n_epochs=n_epochs,
            n_batch=n_batchs, debug=True, log_file_name='mnist_digit_gan')
