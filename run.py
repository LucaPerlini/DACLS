import argparse
import numpy as np
import random
import torch

from dcgan_modelV2 import DCMusicSpectroGAN
from cgan_model import CMusicSpectroGAN
from trainV2 import train_dcgan, train_cgan
from utils import *

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gan', "--gan", required=True, type=str, choices=["dcgan", "cgan"],
                        help="Specify GAN model architecture.")
    parser.add_argument('-d', "--dataset", required=True, type=str,
                        help="Path to spectrogram images dataset.")
    parser.add_argument('-log', action="store_true", help="Use log spectrograms, save to Results_log/")
    parser.add_argument('-mel', action="store_true", help="Use mel spectrograms, save to Results_mel/")
    return parser


def create_numbered_folder(base_path):
    import os
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    i = 1
    while True:
        new_path = os.path.join(base_path, str(i))
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
        i += 1


def main():
    args = argparser().parse_args()

    # Determina la cartella base dei risultati
    if args.log:
        base_results_folder = "Results_log"
    elif args.mel:
        base_results_folder = "Results_mel"
    else:
        base_results_folder = "Results"

    # Crea cartella numerata all'interno di quella base
    result_path = create_numbered_folder(base_results_folder)

    if args.gan == "dcgan":
        device = "cuda"

        # Hyperparameters
        batch_size = 32
        image_size = (128, 128)
        nc = 1    # number of channels
        nz = 100  # Size of the latent vector
        ngf = 64  # Number of generator filters
        ndf = 64  # Number of discriminator filters
        num_epochs = 120
        #lr = 0.0002
        beta1 = 0.6730

        lambda_l1 = 6.19
        lambda_mse = 9.68
        lambda_fm = 9.58
        lr_g = 0.000196 
        lr_d = 0.000051
        dropout_p = 0.35
        update_d_every = 4
        noise_std = 0.0

        # Load the dataset
        dataloader = pt_load_dataset(args.dataset, image_size, batch_size)

        dc_msgan = DCMusicSpectroGAN(device)
        netG, netD = dc_msgan.model(nz, ngf, nc, ndf, dropout_p, update_d_every, noise_std)

        # Train
        train_dcgan(
            device=device,
            nz=nz,
            lr_g=lr_g,
            lr_d=lr_d,
            beta1=beta1,
            netD=netD,
            netG=netG,
            dataloader=dataloader,
            num_epochs=num_epochs,
            result_path_numbered=result_path,
            lambda_l1=lambda_l1,
            lambda_mse=lambda_mse,
            lambda_fm=lambda_fm,
            dropout_p=dropout_p,
            update_d_every=update_d_every,
            noise_std=noise_std
        )

    if args.gan == "cgan":
        # Hyperparameters
        latent_dim = 100  # Dimension of the random latent vector
        num_epochs = 10  # Number of training epochs
        batch_size = 32  # Batch size for training

        # Load the dataset
        spectrograms, labels, num_classes = load_dataset(args.dataset, batch_size, 64, 64)
        image_size = spectrograms[0].shape

        c_msgan = CMusicSpectroGAN()
        discriminator = c_msgan.discriminator(image_size, num_classes)
        generator = c_msgan.generator(latent_dim, num_classes, (image_size[0] // 4, image_size[1] // 4))

        # Train
        train_cgan(spectrograms, labels, latent_dim, num_classes, num_epochs, batch_size, discriminator, generator, result_path)


if __name__ == "__main__":
    main()