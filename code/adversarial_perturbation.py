import logging

import numpy as np
import numpy.typing as npt
import torch
import time

from typing import List

from metrics import create_json_file
from model_dict import get_model

import log_config

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

from captions_utils import create_captions_list
from model_util import create_text_features, predict_probs_and_index_single_image_tensor, packing_pixel_values, \
    unpacking_pixel_values
from data_preprocess import load_data


def get_different_class_random(cs_toexclude: list, candidate_classes: list) -> object:
    """
    Function for which we generate a random class from the candidate_classes which is not in the cs_toexclude.
    @param cs_toexclude: the list of classes which have to be excluded
    @param candidate_classes: the list of the candidate classes
    @return: the chosen random class from the candidate_classes, which is not in the cs_toexlude list
    """
    classes_kept = [c for c in candidate_classes if c not in cs_toexclude]
    return np.random.choice(classes_kept)

def compute_acc2(list_predictions: List[float]) -> float:
    """
    Compute the accuracy metric using a list of predictions which contains only 0 and 1.
    @param list_predictions: the list of predictions which contains only 0 and 1
    @return: float which represents the accuracy metric
    """
    print(list_predictions)
    total_predictions = len(list_predictions)
    correct_predictions = sum(list_predictions)
    acc = correct_predictions / total_predictions if total_predictions > 0 else 0
    return acc


# This function is explicitly used for typographic att.
def testing_set_overall_accuracy(pretrained_model: str, dataset: str, generate_captions: bool = False,
                                 N_samples: int = -1):
    """
    Run the testing on the pretrained model. The idea is to check the performance of the pretrained model on the validation dataset.
    @param pretrained_model: the string which represents the model
    @param dataset:  the string which represents the dataset
    @param generate_captions: the boolean value which specify if the direct labels are used or captions
    @param N_samples: the number of samples extracted from the dataset
    """
    logging.info('Get the model...')
    model, processor, tokenizer = get_model(pretrained_model)
    model.to(device)

    logging.info('Get the data...')
    _, _, x_test, y_test, candidate_labels = load_data(dataset, N_samples)
    candidate_captions = candidate_labels
    captions_original_labels = y_test
    if generate_captions:
        candidate_captions = create_captions_list(candidate_labels)
        captions_original_labels = create_captions_list(y_test)
    if tokenizer is not None:
        with torch.no_grad():
            text_features = create_text_features(model, tokenizer, candidate_captions)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
    else:
        text_features = None

    logging.info('Start classification...')

    list_acc = []

    for i, img in enumerate(x_test):
        image_original = unpacking_pixel_values(**processor(images=x_test[i], return_tensors="pt")).to(device)
        most_similar_index, _ = predict_probs_and_index_single_image_tensor(model, image_original, text_features)
        if candidate_captions[most_similar_index] == captions_original_labels[i]:
            list_acc.append(1)
        else:
            list_acc.append(0)

    acc = compute_acc2(list_acc)

    entire_dataset = False
    if N_samples == -1:
        entire_dataset = True
    content = {"accuracy": acc,
               "pretrained_model": pretrained_model,
               "dataset": dataset,
               "generate_captions": generate_captions,
               "N_samples": len(captions_original_labels),
               "entire_dataset": entire_dataset}
    filename = 'test/'
    filename = filename + "Accuracy_test_data_" + dataset + "_" + pretrained_model + "_" + "generate_captions_" + str(
        generate_captions) + "_n_samples_" + str(len(captions_original_labels))
    create_json_file(content, filename)
    print(acc)

def generate_attacking_setup(x_samples: list, y_samples: List[str], candidate_labels: List[str], model: object,
                             tokenizer: object, candidate_captions: List[str]) -> tuple:
    """
    Generates the attacking setup for the images which are already preprocessed and checks if the preprocessed image has the same label as the one provided (extra assertion check).
    @param x_samples: the list of preprocessed the images
    @param y_samples: the list of strings which represents the correct labels of the images
    @param candidate_labels: the list of strings which represents the candidate labels
    @param model: the model which is used for attack
    @param tokenizer: the tokenizer which generates the textual features
    @param candidate_captions: the list of strings which represents all possible candidate captions
    @return: the tuple of: the list of images, their corresponding true labels and for each image the probability tensor for all the possible captions
    """
    logging.info('Starting the adversarial setup..')

    collecting_probs_list = []

    if tokenizer is not None:
        with torch.no_grad():
            text_features = create_text_features(model, tokenizer, candidate_captions)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
    else:
        text_features = None

    for i, img in enumerate(x_samples):
        image_original = img.to(device)
        most_similar_index, list_probs = predict_probs_and_index_single_image_tensor(model, image_original,
                                                                                     text_features)
        assert candidate_labels[most_similar_index] == y_samples[i]
        collecting_probs_list.append(list_probs)
    tensors_probs = torch.cat(collecting_probs_list, dim=0).to(device)
    return x_samples, y_samples, tensors_probs


def generate_attacking_samples_light(N_adversarial_samples: int, x: list, y: List[str], candidate_labels: List[str],
                                     model: object, processor: object, tokenizer: object,
                                     candidate_captions: List[str]) -> tuple:
    """
    Generates the list of the images and their correct labels which will be used in the attacking phase.
    The idea was to generate the attacking images, such that we will have everytime the same images to attack for a specific model  and dataset.
    @param N_adversarial_samples: the int representing the number of image samples which we want to generate
    @param x: the list of the images from the dataset
    @param y: the list of the correct labels from the dataset
    @param candidate_labels: list of candidate labels
    @param model: the model for which we want to generate the images for attacking phase
    @param processor: the processor of the model
    @param tokenizer: the tokenizer of the model
    @param candidate_captions: the list of the candidate captions
    @return: the tuple which represents the list of the images we want to attack and the list of their respective correct labels
    """
    # We have to ensure that x and y are already random
    samples = 0
    x_samples = []
    y_samples = []

    logging.info('Starting the adversarial selection...')

    if tokenizer is not None:
        with torch.no_grad():
            text_features = create_text_features(model, tokenizer, candidate_captions)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
    else:
        text_features = None

    for i, img in enumerate(x):
        image_original = unpacking_pixel_values(**processor(images=x[i], return_tensors="pt")).to(device)
        most_similar_index, list_probs = predict_probs_and_index_single_image_tensor(model, image_original,
                                                                                     text_features)
        if candidate_labels[most_similar_index] == y[i]:
            x_samples.append(image_original.detach().cpu())
            y_samples.append(y[i])
            samples = samples + 1

        if samples == N_adversarial_samples:
            return x_samples, y_samples


def generate_target_labels_second_similar(tensor_probs: torch.Tensor, candidate_labels: List[str]) -> List[str]:
    """
    Generates the target labels based on the second-best probability.
    @param tensor_probs: the tensor with the probabilities of each label
    @param candidate_labels: the list of the candidate labels
    @return: the list of target labels based on the second-best probability
    """
    target_labels = []
    v, topk_indices = torch.topk(tensor_probs, k=2, dim=1)
    # second probability: top_k[:, 1])
    # first probability: top_k[:, 0])
    second_largest_indices = topk_indices[:, 1]
    for index in second_largest_indices:
        target_labels.append(str(candidate_labels[index]))
    return target_labels


def generate_target_labels_random(y: List[str], candidate_labels: List[str]) -> list:
    """
    Generates the target labels based on the random choice.
    @param y: the list of the correct labels
    @param candidate_labels: the list of the candidate labels
    @return: the list of target labels based on the random choice
    """
    target_labels = []
    for original_label in y:
        target_label = get_different_class_random([original_label], candidate_labels)
        target_labels.append(target_label)
    return target_labels


def generate_target_labels(type_target: str, probs: torch.Tensor, candidate_labels: List[str],
                           y_samples: List[str]) -> list:
    """
    Generates the target labels based on the type of the target. This function is used only for the targeted attacks.
    @param type_target: the string which represent the type of the target, and it can be 'random' or 'second'
    @param probs: the tensor torch representing the probabilities
    @param candidate_labels:  the list of the candidate labels
    @param y_samples: the list of the correct labels
    @return: the list of target labels based on the target type
    """
    if type_target == 'random':
        return generate_target_labels_random(y_samples, candidate_labels)
    elif type_target == 'second':
        return generate_target_labels_second_similar(probs, candidate_labels)
    else:
        raise Exception("This type_target {} is not implemented! Use random or second .".format(type_target))