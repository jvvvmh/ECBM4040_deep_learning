import tensorflow as tf 
import numpy as np
import scipy.misc
import os
import imageio
import skimage
import matplotlib.pyplot as plt
# http://www.cocoachina.com/articles/92307
from IPython.display import display

def load_mnist_data(batch_size=64,datasets='mnist',model_name=None):
    assert datasets in ['mnist','fashion_mnist','cifar10','cifar100'], "you should provided a datasets name in 'mnist','fashion_mnist' "
    if datasets=='mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    elif datasets=='fashion_mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    BUFFER_SIZE=train_images.shape[0]
    if model_name=='WGAN' or model_name == 'WGAN_GP':
        train_images = (train_images-127.5)/127.5
    else:
        train_images=(train_images)/255.0
    train_labels=tf.one_hot(train_labels,depth=10)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)).shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)
    return train_dataset


def inverse_transform(images):
    """
    Transform images whose values are (-1.0, 1.0) to (0.0, 1.0)
    by plusing 1 and then divided by 2.
    
    Return: The transformed images.
    """
    return (images+1.0)/2.0


def check_folder(log_dir):
    """
    Check if the 'log_dir' directory exists. 
    If not, make such a directory.
    
    Return: The path of the directory.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def imsave(images, size, image_path=None, show=False):
    """
    Save the images. 
    
    Args: 
    """
    images = np.squeeze(images)
    fig, axs = plt.subplots(size[0], size[1])
    for i, img in enumerate(images):
        row = i // size[1]
        col = i % size[1]
        axs[row, col].axis('off')
        axs[row, col].imshow(img, cmap='gray')
    if show:
        plt.show()
    else:
        plt.close()
    if image_path is not None:
        fig.savefig(image_path)


def check_args(args):
    """
    check arguments of initializing a GAN model.
    """
    # --checkpoint_dir
    check_folder(args['checkpoint_dir'])

    # --result_dir
    check_folder(args['result_dir'])

    # --result_dir
    check_folder(args['log_dir'])

    # --epoch
    assert args['epoch'] >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args['batch_size'] >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args['z_dim'] >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args


def generate_latent_points(
    latent_dim, n_samples, random_type, n_classes=10):
    """
    Generate 'n_samples' number of random latent variables and random labels.

    Args:
        random_type: 'uniform' or 'gaussian'.
    """
    # generate points in the latent space
    if random_type == 'uniform':
        x_input = np.random.uniform(
            -1, 1, latent_dim * n_samples).astype(np.float32)
    elif random_type == 'gaussian':
        x_input = randn(
            latent_dim * n_samples).astype(np.float32)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = np.random.randint(0, n_classes, n_samples)
    return [z_input, labels]

def batch_to_epoch(lst, n):
    """
    Get the mean of the statistics for each epoch.
    Args: 
        lst: lst[i] is the statistic in batch i.
        n: number of batches per epoch.
    """
    res = []
    for i in range(len(lst) // n):
        res.append(np.mean(lst[i * n: (i + 1) * n]))
    return res
