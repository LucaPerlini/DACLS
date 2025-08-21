import optuna
import random

#from train import train_dcgan
from trainV2 import train_dcgan
from dcgan_modelV2 import DCMusicSpectroGAN
from utils import pt_load_dataset, create_numbered_folder
#from train import evaluate_discriminator
from trainV2 import evaluate_discriminator
import torch
import argparse
import os

# Set random seed for reproducibility
seed = 42
random.seed(seed)
#np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def objective(trial, dataset_path, validation_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparametri da ottimizzare
    beta1 = trial.suggest_float("beta1", 0.4, 0.65)
    lr_g = trial.suggest_float("lr_g", 1.9e-4, 2.3e-4, log=True)
    lr_d = trial.suggest_float("lr_d", 2e-4, 4e-4, log=True)
    lambda_l1 = trial.suggest_float("lambda_l1", 2.0, 7.0)
    lambda_mse = trial.suggest_float("lambda_mse", 5.0, 10.0)
    lambda_fm = trial.suggest_float("lambda_fm", 4.0, 9.0)
    ngf = trial.suggest_categorical("ngf", [64])
    ndf = trial.suggest_categorical("ndf", [64, 128])
    update_d_every = trial.suggest_categorical("update_d_every", [4, 5, 6, 7, 8, 9])
    dropout_p = trial.suggest_float("dropout_p", 0.35, 0.45)
    noise_std = trial.suggest_float("noise_std", 0.0, 0.1)

    # Carica dataset training
    dataloader = pt_load_dataset(dataset_path, image_size=(128, 128), batch_size=32)

    # Inizializza modello
    nz = 100
    nc = 1
    dcgan = DCMusicSpectroGAN(device)
    netG, netD = dcgan.model(nz=nz, ngf=ngf, nc=nc, ndf=ndf, dropout_p=dropout_p, update_d_every=update_d_every, noise_std=noise_std)

    # Crea cartella risultati numerata
    result_path = create_numbered_folder("OptunaResults9")

    # Train
    num_epochs = 100  # ridotto per velocità tuning
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
        update_d_every=update_d_every,
        dropout_p=dropout_p,
        noise_std=noise_std
    )

    # Valutazione sul validation set
    val_metrics = evaluate_discriminator(netD, validation_path, device)

    # Restituisci val_accuracy del discriminator per massimizzare
    val_accD = val_metrics[0]

    # Salvataggio risultati su file txt
    with open("optuna_results9.txt", "a") as f:
        f.write(
            f"Trial {trial.number} | Val_accD: {val_accD:.4f} | beta1: {beta1:.4f} | lr_g: {lr_g:.6f} | lr_d: {lr_d:.6f} | "
            f"lambda_l1: {lambda_l1:.2f} | lambda_mse: {lambda_mse:.2f} | lambda_fm: {lambda_fm:.2f} | "
            f"ngf: {ngf} | ndf: {ndf} | dropout_p: {dropout_p:.2f} | rapporto: {update_d_every} | noise_std: {noise_std:.3f} \n"
        )

    return val_accD

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to training dataset")
    parser.add_argument("--validation", required=True, help="Path to validation dataset")
    parser.add_argument("--n_trials", type=int, default=50, help="Numero di trial Optuna")
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args.dataset, args.validation), n_trials=args.n_trials)

    print("Migliori iperparametri trovati:")
    print(study.best_params)

if __name__ == "__main__":
    main()
