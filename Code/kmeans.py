import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    c1 = generator.choice(n, 1)
    centers = np.array([c1[0]])
    for k in range(1, n_cluster):
        Distances = []
        for i in x:
            centers_dis = []
            centers_dis = np.append(centers_dis, [np.linalg.norm(i - x[c]) ** 2 for c in centers])
            Distances = np.append(Distances, np.min(centers_dis))
        P = Distances / np.sum(Distances)
        max_dis_index = np.argmax(P)
        centers = np.append(centers, max_dis_index)

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers.tolist()


def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():
    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"

        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        count = 1
        centroid = x[self.centers]
        distortion = 10**10
        # print("original centroid :", centroid)x
        while count <= self.max_iter:
            print("iteration count:", count)
            distance_of_all_points = np.array([np.linalg.norm(x-c, axis=1)**2 for c in centroid])
            # print("distance all:", distance_of_all_points[:10])
            # print("length of distances", distance_of_all_points.shape)
            y = np.array(np.argmin(distance_of_all_points, axis=0))
            wcss = 0.0
            for k in range(self.n_cluster):
                wcss += np.sum(distance_of_all_points[k][np.where(y == k)[0]], axis=0)
            avg_wcss = wcss/N
            # print("avg wcss:", avg_wcss)
            # print("tolerance :", distortion - avg_wcss)
            if distortion - avg_wcss <= self.e:
                break
            else:
                distortion = avg_wcss
                new_centroid = [np.average(x[np.where(y==k)[0]], axis=0) if len(np.where(y==k)[0]) > 0 else centroid[k]
                                for k in range(self.n_cluster)]
                centroid = new_centroid
            count = count + 1

        means = np.array(centroid)
        number_of_updates = count - 1
        membership = np.array([np.argmin([np.linalg.norm(i - c) ** 2 for c in centroid]) for i in x])
        return means, membership, number_of_updates

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #      'Implement fit function in KMeans class')

        # DO NOT CHANGE CODE BELOW THIS LINE
        # return centroids, y, self.max_iter


class KMeansClassifier():
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        k_means = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        centroids, membership, _ = k_means.fit(x, centroid_func)
        centroid_labels = np.array([np.argmax(np.bincount(y[np.where(membership == k)[0]])) if len(
            np.where(membership == k)[0]) > 0 else 0 for k in range(self.n_cluster)])

        print("centroid labels :", centroid_labels)

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        labels = self.centroid_labels[[np.argmin([np.linalg.norm(i - c) for c in self.centroids]) for i in x]]

        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE

    N, M = image.shape[:2]
    # print("N & M :", N, M)
    data = image.reshape(N * M, 3)
    new_im = np.array([code_vectors[np.argmin([np.linalg.norm(i - c) for c in code_vectors])] for i in
                       data]).reshape(N, M, 3)

    # print("new im:", new_im)

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im
