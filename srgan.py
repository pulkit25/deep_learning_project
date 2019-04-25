"""
Super-resolution of CelebA using Generative Adversarial Networks.

The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0

Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to 'datasets/'
4. Run the sript using command 'python srgan.py'
"""

from __future__ import print_function, division
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense
from keras.layers import BatchNormalization, Activation, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.activations import relu
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
from data_loader import DataLoader
import numpy as np
import os
import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type = str, default = './data', help = 'data directory')
parser.add_argument('--outputdir', type = str, default = './outputs', help = 'output directory')
parser.add_argument('--upscale', type = int, default = 4, help = 'upscaling factor')
parser.add_argument('--inputdim', type = int, default = 256, help = 'input image dimensions(square image)')
parser.add_argument('--nresblocks', type = int, default = 16, help = 'number of residual blocks')
parser.add_argument('--lr', type = float, default = 0.0002, help = 'learning rate')
parser.add_argument('--epochs', type = int, default = 1000, help = 'number of epochs')
parser.add_argument('--batchsize', type = int, default = 1, help = 'batch size')
parser.add_argument('--pretrain', type = bool, default = False, help = 'Pretrain with SRResnet')
args = parser.parse_args()

class SRGAN():
    def __init__(self):
        # Input shape
        
        self.channels = 3
        self.lr_height = args.inputdim                # Low resolution height
        self.lr_width = args.inputdim                  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height * args.upscale   # High resolution height
        self.hr_width = self.lr_width * args.upscale     # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # Number of residual blocks in the generator
        self.n_residual_blocks = args.nresblocks

        optimizer = Adam(args.lr)
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss = 'mse', optimizer = optimizer, metrics = ['accuracy'])

        # Configure data loader
        self.dataset_name = args.datadir #'./datasets/img_align_celeba'
        self.data_loader = DataLoader(dataset_name = self.dataset_name, img_res = (self.hr_height, self.hr_width))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'mse', optimizer = optimizer, metrics = ['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        
        
        #SRResnet pretraining
        if(args.pretrain):
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size = 0)
            self.generator.compile(loss = 'mse', optimizer = optimizer)
            self.generator.fit(imgs_lr,imgs_hr, batch_size = 32, epochs=10**5)
            self.generator.save('SRResnet.hdf5')
        else:
            self.generator.load_model('SRResnet.hdf5')
        
        
        # High res. and low res. images
        img_hr = Input(shape = self.hr_shape)
        img_lr = Input(shape = self.lr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)

        # For the combined model we will only train the generator, freeze discriminator
        self.discriminator.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        # self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined = Model([img_lr, img_hr], [validity, fake_features, fake_hr])

        # self.combined.compile(loss = ['binary_crossentropy', 'mse'], loss_weights = [1e-3, 1], optimizer = optimizer)
        self.combined.compile(loss = ['binary_crossentropy', 'mse', 'mse'], loss_weights = [1e-1, 0.006, 1], optimizer = optimizer)
        print(self.combined.metrics_names)


    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights = "imagenet")
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape = self.hr_shape)

        # Extract image features
        img_features = vgg(img)

        return Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size = 3, strides = 1, padding = 'same')(layer_input)
            # d = PReLU(shared_axes = [1, 2])(d)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum = 0.8)(d)
            d = Conv2D(filters, kernel_size = 3, strides = 1, padding = 'same')(d)
            d = BatchNormalization(momentum = 0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size = 2)(layer_input)
            u = Conv2D(256, kernel_size = 3, strides = 1, padding = 'same')(u)
            # u = PReLU(shared_axes = [1, 2])(u)
            u = Activation('relu')(u)
            return u

        # Low resolution image input
        img_lr = Input(shape = self.lr_shape)

        # Pre-residual block
        c1 = Conv2D(64, kernel_size = 9, strides = 1, padding = 'same')(img_lr)
        # c1 = PReLU(shared_axes = [1, 2])(c1)
        c1 = Activation('relu')(c1)

        # Propogate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c2 = Conv2D(64, kernel_size = 3, strides = 1, padding = 'same')(r)
        c2 = BatchNormalization(momentum = 0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size = 9, strides = 1, padding = 'same', activation = 'tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides = 1, bn = True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size = 3, strides = strides, padding = 'same')(layer_input)
            d = LeakyReLU(alpha = 0.2)(d)
            if bn:
                d = BatchNormalization(momentum = 0.8)(d)
            return d

        # Input img
        d0 = Input(shape = self.hr_shape)

        d1 = d_block(d0, self.df, bn = False)
        d2 = d_block(d1, self.df, strides = 2)
        d3 = d_block(d2, self.df * 2)
        d4 = d_block(d3, self.df * 2, strides = 2)
        d5 = d_block(d4, self.df * 4)
        d6 = d_block(d5, self.df * 4, strides = 2)
        d7 = d_block(d6, self.df * 8)
        d8 = d_block(d7, self.df * 8, strides = 2)

        d9 = Dense(self.df * 16)(d8)
        d10 = LeakyReLU(alpha = 0.2)(d9)
        validity = Dense(1, activation = 'sigmoid')(d10)

        return Model(d0, validity)

    def train(self, epochs, batch_size = 1, sample_interval = 50):

        start_time = datetime.datetime.now()

        for epoch in range(epochs):

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features, imgs_hr])
            # g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])
            print(g_loss)

            if epoch % 100 == 0:
                with open('loss.txt', 'a') as f:
                    f.writelines(['%.3f %.3f %.3f %.3f'%(g_loss[0], g_loss[1], g_loss[2], g_loss[3])])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("%d time: %s" % (epoch, elapsed_time))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)


    def sample_images(self, epoch):
        os.makedirs(args.outputdir, exist_ok = True)
        r = 2

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size = 2, is_testing = True)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images
        for i in range(r):
            fig = plt.figure()
            PSNR = self.psnr(imgs_hr[i], fake_hr[i])
            plt.title('Gen - PSNR = ' + str(PSNR))
            plt.imshow(fake_hr[i])
            fig.savefig(args.outputdir + '/%d_gen%d.png' % (epoch, i))
            plt.close()
		
		# Save high resolution originals
            fig = plt.figure()
            plt.title('High res')
            plt.imshow(imgs_hr[i])
            fig.savefig(args.outputdir + '/%d_high_res%d.png' % (epoch, i))
            plt.close()
		# Save low resolution images for comparison
            fig = plt.figure()
            plt.title('Low res')
            plt.imshow(imgs_lr[i])
            fig.savefig(args.outputdir + '/%d_lowres%d.png' % (epoch, i))
            plt.close()
            
    def psnr(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                             " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                       str(y_pred.shape))
    
        return -10. * np.log10(np.mean(np.square(y_pred - y_true)))

if __name__ == '__main__':
    gan = SRGAN()
    gan.train(epochs = args.epochs, batch_size = args.batchsize, sample_interval = 50)
