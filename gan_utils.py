from __future__ import print_function, division
import keras
import cv2
import numpy as np

def gen_real_samples(real_X, n_samples):
    # gen random idxs
    idxs = np.random.randint(0, real_X.shape[0], n_samples)
    mini_X = real_X[idxs]
    # gen real labels
    mini_Y = np.ones((n_samples,1))
    return mini_X, mini_Y

def gen_noise_samples(inp_shape, n_samples):
    mini_X = np.random.rand(reduce((lambda x,y: x*y), inp_shape) * n_samples)
    shap = [n_samples] + list(inp_shape)
    mini_X = mini_X.reshape(tuple(shap))
    # scale to -1 to +1
    # gen fake labels
    mini_Y = np.zeros((n_samples,1))
    return mini_X, mini_Y

def pre_train_discriminator(dis_model, real_X, n_iter, n_batch, inp_shape):
    half_batch = int(n_batch / 2)
    history = np.zeros((n_iter, 2))
    for i in range(1,n_iter+1):
        mini_X_real, mini_Y_real = gen_real_samples(real_X, half_batch)
        mini_X_fake, mini_Y_fake = gen_noise_samples(inp_shape, half_batch)
        # train on real samples
        _, real_acc = dis_model.train_on_batch(mini_X_real, mini_Y_real)
        # train on fake/noise samples
        _, fake_acc = dis_model.train_on_batch(mini_X_fake, mini_Y_fake)
        history[i-1] = np.array((real_acc, fake_acc)) * 100
        if i%10 == 0:
            real_acc, fake_acc = real_acc*100, fake_acc*100
            print('Pre-training Discriminator: n: %d RealAcc: %.2f FakeAcc: %.2f Combined: %.2f'%
                (i, real_acc, fake_acc, np.mean((real_acc, fake_acc))))
    return history

def gen_latent_points(latent_dim, n_samples):
    # generate from gaussian
    mini_latent_inp = np.random.randn(latent_dim * n_samples)
    mini_latent_inp = mini_latent_inp.reshape(n_samples, latent_dim)
    return mini_latent_inp

def gen_fake_samples(gen_model, latent_dim, n_samples):
    # generate latent vectors
    mini_latent_inp = gen_latent_points(latent_dim, n_samples) 
    # predict g(mini_latent_inp)
    mini_fake_x = gen_model.predict(mini_latent_inp)
    # fake label
    mini_fake_y = np.zeros((n_samples, 1))
    return mini_fake_x, mini_fake_y

def train_gan(gen_model, dis_model, gan_model, real_X, latent_dim, n_epochs, n_batch, debug=False):
    batch_per_epoch = int(real_X.shape[0]/n_batch)
    half_batch = int(n_batch/2)

    if debug:
        rows,cols = 10,10
        show_x_latents = gen_latent_points(latent_dim, rows*cols)
        img_size = rows*real_X.shape[1], cols*real_X.shape[2]
        video = cv2.VideoWriter('gan_training.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),24,
                (800,900), False)
        debug_img = np.zeros(img_size, dtype=np.uint8)
        disp_img = np.zeros((900,800), dtype=np.uint8)
        cv2.namedWindow('Training Debug', cv2.WINDOW_NORMAL)

    for epoch in range(n_epochs):
        for batch in range(batch_per_epoch):
            # sampling random real samples
            mini_X_real, mini_Y_real = gen_real_samples(real_X, half_batch)
            # sampling fake
            mini_X_fake, mini_Y_fake = gen_fake_samples(gen_model, latent_dim, half_batch)
            mini_X = np.vstack((mini_X_real, mini_X_fake))
            mini_Y = np.vstack((mini_Y_real, mini_Y_fake))
            # train discriminator
            dis_loss, _ = dis_model.train_on_batch(mini_X, mini_Y)
            # sampling latent points
            x_latent = gen_latent_points(latent_dim, n_batch)
            # create inverted labels for real(so 1)
            y_latent = np.ones((n_batch,1))
            gen_loss = gan_model.train_on_batch(x_latent, y_latent)
            disp = "E:%d/%d B:%d/%d D:%.3f G:%.3f"%(epoch+1, n_epochs, batch,
                batch_per_epoch, dis_loss, gen_loss)
            print(disp)
            if debug:
                disp_img[:100,:] *= 0
                cv2.putText(disp_img,disp, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                fake_samples = gen_model.predict(show_x_latents)
                fake_samples = np.uint8(fake_samples*255)
                fake_samples = fake_samples.reshape(fake_samples.shape[:-1])
                for i in range(rows*cols):
                    row_i, col_i = i // rows, i % cols
                    row_i, col_i = row_i*real_X.shape[1], col_i*real_X.shape[2]
                    row_j, col_j = row_i+real_X.shape[1], col_i+real_X.shape[2]
                    debug_img[row_i:row_j, col_i:col_j] = fake_samples[i]
                disp_img[100:,:] = cv2.resize(debug_img, (800,800))
                cv2.imshow('Training Debug', disp_img)
                cv2.waitKey(1)
                video.write(disp_img)
    if debug:
        cv2.destroyAllWindows()
