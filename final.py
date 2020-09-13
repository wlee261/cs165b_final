import cv2
import numpy as np
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import glob
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pandas as pd
import time

class Visual_BOW():
    def __init__(self, k=20, dictionary_size=50):
        self.k = k  # number of SIFT features to extract from every image
        self.dictionary_size = dictionary_size  # size of your "visual dictionary" (k in k-means)
        self.n_tests = 10  # how many times to re-run the same algorithm (to obtain average accuracy)

    def extract_sift_features(self):
        '''
        To Do:
            - load/read the Caltech-101 dataset
            - go through all the images and extract "k" SIFT features from every image
            - divide the data into training/testing (70% of images should go to the training set, 30% to testing)
        Useful:
            k: number of SIFT features to extract from every image
        Output:
            train_features: list/array of size n_images_train x k x feature_dim
            train_labels: list/array of size n_images_train
            test_features: list/array of size n_images_test x k x feature_dim
            test_labels: list/array of size n_images_test
        '''


        #os.system(pip3 install --user pandas)
        
        FILEPATH = "/Users/graceshin/Downloads/101_ObjectCategories/*"
        
        filenames = []
        images = []
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []
        rand = 0
        filenames = glob.glob(FILEPATH)
        
        for i in range(len(filenames)):
            images.append([])
            images[i] = glob.glob(filenames[i]+"/*")
            images[i].append(filenames[i])

       
        #print(len(descs[0]))
        for x in range(len(images)):
            for y in range(len(images[x])-1):
                rand = random.randint(0,100)
                print(images[x][y])
                img1 = cv2.imread(images[x][y])
                gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                sift = cv2.xfeatures2d.SIFT_create()
                (kp, descs) = sift.detectAndCompute(gray, None)

                if descs is None:
                    continue
                if (len(descs)>self.k):
                    descs = descs[0:self.k]
                    
                (head, tail) = os.path.split(images[x][len(images[x])-1])
                if(rand > 70):
                    test_features.append(descs)
                    test_labels.append(tail)
                else:
                    train_features.append(descs)
                    train_labels.append(tail)
                    
        '''print(len(train_features))
        print(len(train_features[0]))
        print(len(train_features[0][0]))
        print(train_labels)
        print(np.shape(train_features))'''
        
        return train_features, train_labels, test_features, test_labels

    def create_dictionary(self, features):
        '''
        To Do:
            - go through the list of features
            - flatten it to be of size (n_images x k) x feature_dim (from 3D to 2D)
            - use k-means algorithm to group features into "dictionary_size" groups
        Useful:
            dictionary_size: size of your "visual dictionary" (k in k-means)
        Input:
            features: list/array of size n_images x k x feature_dim
        Output:
            kmeans: trained k-means object (algorithm trained on the flattened feature list)
        '''

        flattened = []
        for i in range(len(features)):
            for x in range(len(features[i])):
                flattened.append(features[i][x])
        kmeans = MiniBatchKMeans(n_clusters = self.dictionary_size).fit(flattened)
        #print(kmeans.labels_)
        #print(np.shape(flattened))
        return kmeans

    def convert_features_using_dictionary(self, kmeans, features):
        '''
        To Do:
            - go through the list of features (images)
            - for every image go through "k" SIFT features that describes it
            - every image will be described by a single vector of length "dictionary_size"
            and every entry in that vector will indicate how many times a SIFT feature from a particular
            "visual group" (one of "dictionary_size") appears in the image. Non-appearing features are set to zeros.
        Input:
            features: list/array of size n_images x k x feature_dim
        Output:
            features_new: list/array of size n_images x dictionary_size
        '''
        clusters = kmeans.cluster_centers_
        #kmean_labels = []
        #kmean_labels = kmeans.labels_.tolist()
        prediction = []
        features_new = []
        counter = 0
        
        for z in range(len(features)):
            features_new.append([])
            for i in range(self.dictionary_size):
                features_new[z].append(0)


        flattened = []
        for a in range(len(features)):
            for b in range(len(features[a])):
                flattened.append(features[a][b])

        prediction = kmeans.predict(flattened)
        for x in range(len(features)):
            for y in range(len(features[x])):
                features_new[x][prediction[counter]] = features_new[x][prediction[counter]] + 1
                counter = counter+1

                
        #print(features_new)
        #print(np.shape(features))
        #print(len(kmean_labels))
        return features_new

    def train_svm(self, inputs, labels):
        '''
        To Do:
            - train an SVM classifier using the data
            - return the trained object
        Input:
            inputs: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        Output:
            clf: trained svm classifier object (algorithm trained on the inputs/labels data)
        '''
        clf = svm.SVC()
        clf.fit(inputs, labels)
        return clf            



    def test_svm(self, clf, inputs, labels):
        '''
        To Do:
            - test the previously trained SVM classifier using the data
            - calculate the accuracy of your model
        Input:
            clf: trained svm classifier object (algorithm trained on the inputs/labels data)
            inputs: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        Output:
            accuracy: percent of correctly predicted samples
        '''

        correct = 0
        total = 0
        changedim = []
        for x in range(len(inputs)):
            changedim.append([])
            changedim[x].append(inputs[x])

        
        for i in range(len(changedim)):
            if(clf.predict(changedim[i]) == labels[i]):
                correct = correct + 1
            total = total + 1

        accuracy = correct/total
        print(accuracy)
        return accuracy

    def save_plot(self, features, labels):
        '''
        To Do:
            - perform PCA on your features
            - use only 2 first Principle Components to visualize the data (scatter plot)
            - color-code the data according to the ground truth label
            - save the plot
        Input:
            features: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        '''

        SAVEPATH = '/Users/graceshin/Downloads/plot.png'

        N = len(labels)
        
        principal_axes = []
        reduced_data = []
        pca = PCA(n_components = 2)
        pca.fit(features)
        principal_axes = pca.components_
        principal_axes = np.transpose(principal_axes)
        reduced_data = np.dot(features, principal_axes)

        mydict = {}
        i = 0
        for item in labels:
            if(i>0 and item in mydict):
                continue
            else:
                i = i+1
                mydict[item] = i

        k = []
        for item in labels:
            k.append(mydict[item])

        colors = []
        for i in range(N):
            colors.append('#%06X' %random.randint(0, 0xFFFFFF))

        plt.scatter(reduced_data[:,0], reduced_data[:,1], c = k, s = 5, cmap = mpl.colors.ListedColormap(colors))
        plt.savefig(SAVEPATH)
        
        
        

############################################################################
################## DO NOT MODIFY ANYTHING BELOW THIS LINE ##################
############################################################################

    def algorithm(self):
        # This is the main function used to run the program
        # DO NOT MODIFY THIS FUNCTION
        accuracy = 0.0
        for i in range(self.n_tests):
            train_features, train_labels, test_features, test_labels = self.extract_sift_features()
            kmeans = self.create_dictionary(train_features)
            train_features_new = self.convert_features_using_dictionary(kmeans, train_features)
            classifier = self.train_svm(train_features_new, train_labels)
            test_features_new = self.convert_features_using_dictionary(kmeans, test_features)
            accuracy += self.test_svm(classifier, test_features_new, test_labels)
            self.save_plot(test_features_new, test_labels)
        accuracy /= self.n_tests
        return accuracy

if __name__ == "__main__":
    start_time = time.time()
    alg = Visual_BOW(k=20, dictionary_size=50)
    accuracy = alg.algorithm()
    print("Final accuracy of the model is:", accuracy)
    print("--- %s seconds ---" % (time.time() - start_time))
