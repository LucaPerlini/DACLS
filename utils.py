import matplotlib.pyplot as plt
import numpy as np
import os

def create_numbered_folder(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    i = 1
    while True:
        new_path = os.path.join(base_path, str(i))
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
        i += 1

def pt_load_dataset(data_path, image_size, batch_size):
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from safe_imagefolder import SafeImageFolder  # ✅ Usa la versione sicura

    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])


    dataset = SafeImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader



def tf_load_dataset(dataset_path, batch_size, image_shape, class_mode, shuffle, color_mode):
    from keras.preprocessing.image import ImageDataGenerator

    dataset_generator = ImageDataGenerator()
    dataset_generator = dataset_generator.flow_from_directory(
                            dataset_path,
                            target_size=image_shape,
                            batch_size=batch_size,
                            class_mode=class_mode,
                            shuffle=shuffle,
                            color_mode=color_mode)
    return dataset_generator


def load_dataset(dataset_path, batch_size = 32, spectrogram_height = 512, spectrogram_width = 512):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize

    data_generator = datagen.flow_from_directory(
        dataset_path,
        color_mode="grayscale",
        target_size=(spectrogram_height, spectrogram_width),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    num_classes = data_generator.num_classes
    spectrograms = []
    labels = []

    for batch in data_generator:
        spectrograms.extend(batch[0])
        labels.extend(batch[1])

        # Break the loop when all data is extracted
        if len(spectrograms) >= data_generator.n:
            break

    spectrograms = np.array(spectrograms)
    labels = np.array(labels)

    return spectrograms, labels, num_classes


def latent_vector(latent_dim, n, n_cats=10):
    # Generate points in the latent space
    latent_input = np.random.randn(latent_dim * n)

    # Reshape into a batch of inputs for the network
    latent_input = latent_input.reshape(n, latent_dim)

    # Generate category labels 
    cat_labels = np.random.randint(0, n_cats, n)
    return [latent_input, cat_labels] 


def real_samples(dataset, categories, n):
    # Create a random list of indices
    indx = np.random.randint(0, dataset.shape[0], n)

    # Select real data samples (images and category labels) using the list of random indeces from above
    X, cat_labels = dataset[indx], categories[indx]

    cat_labels = np.array(cat_labels)
    if cat_labels.ndim > 1: # for one-hot encoded case
        cat_labels = np.argmax(cat_labels, axis=1)

    # Class labels
    y = np.ones((n, 1))
    return [X, cat_labels], y


def fake_samples(generator, latent_dim, n):
    # Draw latent variables
    latent_output, cat_labels = latent_vector(latent_dim, n)

    # Predict outputs (i.e., generate fake samples)
    X = generator.predict([latent_output, cat_labels])

    # Create class labels
    y = np.zeros((n, 1))
    return [X, cat_labels], y


def show_fakes(generator, latent_dim, n=10):
    # Get fake (generated) samples
    x_fake, y_fake = fake_samples(generator, latent_dim, n)

    # Rescale from [-1, 1] to [0, 1]
    X_tst = (x_fake[0] + 1) / 2.0

    # Display fake (generated) images
    _, axs = plt.subplots(2, 5, sharey=False, tight_layout=True, figsize=(12,6), facecolor='white')
    k=0
    for i in range(0,2):
        for j in range(0,5):
            axs[i,j].matshow(X_tst[k], cmap='gray')
            axs[i,j].set(title=x_fake[1][k])
            axs[i,j].axis('off')
            k=k+1
    plt.show()


# Generate and save example spectrograms during training
def generate_and_save_images(path, epoch, latent_dim, num_classes, generator):
    random_latent_vectors = np.random.normal(size=(10, latent_dim))
    random_labels = np.random.randint(0, num_classes, size=(10, num_classes))
    generated_spectrograms = generator.predict([random_latent_vectors, random_labels])

    _, axs = plt.subplots(2, 5, figsize=(12, 6))
    count = 0
    for i in range(2):
        for j in range(5):
            axs[i, j].imshow(generated_spectrograms[count, :, :, 0], cmap="gray")
            axs[i, j].axis("off")
            count += 1
    plt.tight_layout()
    plt.savefig(path + "\generated_spectrograms_epoch_{:04d}.png".format(epoch))
    plt.close()