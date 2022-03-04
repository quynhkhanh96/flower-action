import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from cifar import get_cifar_client_loader, get_cifar_test_loader
from mnist import get_mnist_client_loader, get_mnist_test_loader
from google_landmark_2020 import get_landmark_client_loader
from hmdb51 import get_hmdb51_client_loader
