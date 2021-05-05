import tensorflow
import numpy as np
import warnings

def get_mean_cov_for_each_label(datasets='mnist'):
    """
    return a dictionary d
    example: d[0] = [mean_0, cov_0]
    """
    print(f"dataset {datasets}")
    assert datasets in ['mnist','fashion_mnist']
    if datasets=='mnist':
        (train_images, train_labels), (test_images, test_labels) = tensorflow.keras.datasets.mnist.load_data()
    elif datasets=='fashion_mnist':
        (train_images, train_labels), (test_images, test_labels) = tensorflow.keras.datasets.fashion_mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images) / 255.0 # values are in range [0, 1]
    d = {}
    nClass = 10
    for i in range(nClass):
        indexes = np.array(range(len(train_labels)))[train_labels == i]
        tmp_data = train_images[indexes]
        # print(f"label {i}, {len(tmp_data) / len(train_images):.3f}")
        tmp_data = tmp_data.reshape(tmp_data.shape[0], -1) # N x 784
        mean = tmp_data.mean(axis=0) # 784
        cov = np.cov(tmp_data.T) # 784 x 784
        d[i] = [mean, cov]
    return d


def frechet_distance(mean1, cov1, mean2, cov2):
	"""
	return:
	    FID between two disributions
	"""
	
	def check(x):
		neg = (x < 0)
		if neg.any():
			warnings.warn('Rank deficient covariance matrix, '
			'Frechet distance will not be accurate.', Warning)
		x[neg] = 0

    l1, v1 = np.linalg.eigh(cov1)
	check(l1)

	cov1_sqrt = (v1 * l1 ** 0.5).dot(v1.T)
	cov_prod = cov1_sqrt.dot(cov2).dot(cov1_sqrt)
	lp = np.linalg.eigvalsh(cov_prod)
	check(lp)

    d = mean1 - mean2
	tr = l1.sum() + np.trace(cov2) - 2 * np.sqrt(lp).sum()
	fid = tr + d.dot(d)

	return fid