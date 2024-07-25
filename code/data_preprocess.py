import json
import os
import random

import numpy as np
import torch
from PIL import Image
import class_names_cifar
from captions_utils import create_labels_from_index
import pickle


def take_N_random_data(x: list, y: list, random_samples: int) -> tuple:
    """
    From the data and its labels, randomly selects N data instances and their corresponding labels.
    @param x: the list of the images
    @param y: the list of the labels
    @param random_samples: the number which represents the amount of the selected instances. If it is smaller than 0, we consider all the data -> just randomly shuffle the data
    @return: the tuple which represents the new data instances and their corresponding labels
    """
    # Ensure that the number of samples does not exceed the size of the testing data
    if random_samples <= 0:
        random_samples = len(x)
    num_samples = min(random_samples, len(x))

    # Randomly select indices without replacement
    random_indices = np.random.choice(len(x), num_samples, replace=False)

    # Extract the random training data and labels
    random_x = [x[i] for i in random_indices]
    random_y = [y[i] for i in random_indices]

    return random_x, random_y


def unpickle(file: str) -> dict:
    """
    Function to unpickle a file.
    @param file: string which represents the path to the data file
    @return: the dictionary after extracting the data
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_cifarX_testing_data(type_cifar: str) -> tuple:
    """
    Get the cifar10/cifar100 testing data and also all the possible labels.
    @param type_cifar: string which represents the type of the dataset. It can be 'cifar10' or 'cifar100'
    @return: the tuple which represents 2 empty lists, the np array with images, their corresponding labels and the all possible labels
    """
    if type_cifar == 'cifar10':
        file = '../data/cifar-10-batches-py/test_batch'
        class_names = class_names_cifar.cifar10_class_names
        labels_string = b'labels'
    elif type_cifar == 'cifar100':
        file = '../data/cifar-100-python/test'
        class_names = class_names_cifar.cifar100_class_names
        labels_string = b'fine_labels'
    else:
        print("Wrong CIFAR. Choose cifar10 or cifar100 !!!")
        raise Exception("Wrong CIFAR. Choose cifar10 or cifar100 !!!")
    data_test = unpickle(file)
    data = data_test[b'data']
    data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)
    labels = data_test[labels_string]
    labels = create_labels_from_index(labels, class_names)
    return [], [], np.array(data), labels, class_names


def get_ImageNet_testing_data() -> tuple:
    """
    Gets the ImageNet validation data for the attacking and also all the possible labels.
    @return: the tuple which represents 2 empty lists (because we do not have testing data), the list of np array with images, their corresponding labels and the all possible labels
    """
    directory_string_data = '../data/imagenet/val'
    directory_string_labels = '../data/imagenet/'
    candidate_labels_json_file = directory_string_labels + 'candidate_labels.json'

    # Load the second JSON file (index to code_label and label)
    with open(directory_string_labels + 'image_labels_complete.json', 'r') as file_labels:
        image_to_info = json.load(file_labels)

    file_list = os.listdir(directory_string_data)
    # Number of entries you want to select
    num_entries_to_select = 10000
    # Use random.sample to select 10,000 random entries
    file_list = random.sample(file_list, num_entries_to_select)
    # Due to the memory limitation and speed, we will not process all 50.000 images, and we limit to only 10.000 images

    list_images = []
    labels_images = []
    with open(candidate_labels_json_file, 'r') as candidate_file:
        candidate_labels_data = json.load(candidate_file)
        list_candidate_labels = candidate_labels_data['candidate_labels']

    for filename in file_list:
        file_path = os.path.join(directory_string_data, filename)
        image = Image.open(file_path).convert('RGB')
        image_np = np.array(image)
        list_images.append(image_np)
        labels_images.append(int(image_to_info[filename][0]))
    labels_images = create_labels_from_index(labels_images, list_candidate_labels)
    return [], [], list_images, labels_images, list_candidate_labels

def load_data(dataset: str, N_samples_testing: int = -1) -> tuple:
    """
    Loads the specific requested data. If it is specified with a number greater than 0, we also shuffle the testing data
    @param dataset: the string which represents the dataset
    @param N_samples_testing: the number of the instances which we want to extract randomly from the testing
    @return: the tuple which represents:
    1. images for training and their labels
    2. images for testing and their labels
    3. all possible labels
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    class_names = []
    if dataset == 'cifar10' or dataset == 'cifar100':
        x_train, y_train, x_test, y_test, class_names = get_cifarX_testing_data(dataset)
    elif dataset == 'imagenet':
        x_train, y_train, x_test, y_test, class_names = get_ImageNet_testing_data()
    else:
        print("Wrong Dataset!")
        raise Exception("Wrong Dataset!")

    x_test, y_test = take_N_random_data(x_test, y_test, N_samples_testing)
    return x_train, y_train, x_test, y_test, class_names


def get_image_bounds(dataset: str) -> list:
    """
    Gets the image bounds for the 'cifar10' and 'cifar100' datasets.
    This function is also extendable for other datasets configurations.
    @param dataset: the string which represents the dataset and can have the values 'cifar10' and 'cifar100'
    @return: the list which represents the image bounds for the datasets
    The form is the following:[(x_min, x_max), (y_min, y_max), (r_min, r_max), (g_min, g_max), (b_min, b_max)]
    """
    if dataset == 'cifar10' or dataset == 'cifar100':
        # since the images are from dimension 0 to 31, even though their size is 32x32
        return [(0, 31), (0, 31), (0, 255), (0, 255), (0, 255)]
    else:
        raise Exception("Wrong Dataset!")


def load_preprocessed_data(pretrained_model: str, dataset: str, generate_captions: bool) -> tuple:
    """
    Load the preprocessed data for the attacking phased, based on the pretrained model, dataset and generate captions.
    Those images can be found in the '../data_s' directory and needs to follow the following structure:
    "data_s/MODEL/DATASET/generate_captions_True"  or "data_s/MODEL/DATASET/generate_captions_False"
    @param pretrained_model: the string which represents the model
    @param dataset: the string which represents the dataset
    @param generate_captions: the bool value which represents the 'generate_caption' parameter
    @return: the tuple which represents the list of preprocessed images, the list of their correct labels and all the possible labels
    """
    file_images = '../data_s/' + pretrained_model + '/' + dataset + '/generate_caption_' + str(
        generate_captions) + '/images'
    file_labels = '../data_s/' + pretrained_model + '/' + dataset + '/generate_caption_' + str(
        generate_captions) + '/labels.json'
    x_samples = torch.load(file_images)
    with open(file_labels, 'r') as file:
        data = json.load(file)
        y_samples = data['labels']
    if dataset == 'cifar10':
        class_names = class_names_cifar.cifar10_class_names
    elif dataset == 'cifar100':
        class_names = class_names_cifar.cifar100_class_names
    elif dataset == 'imagenet':
        with open('../data/imagenet/candidate_labels.json', 'r') as file:
            data = json.load(file)
            class_names = data['candidate_labels']
    else:
        raise Exception('We do not have the preprocessed data for this dataset')
    return x_samples, y_samples, class_names