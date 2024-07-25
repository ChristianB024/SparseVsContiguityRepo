import json
from typing import List

import numpy as np
import torch


def compute_success_rate(list_success: list) -> float:
    """
    Compute the SR based on the list of success.
    @param list_success: the list of success - it contains only 0 and 1 (1 successful attack and 0 unsuccessful attack).
    @return: the value which represents the success rate - how many images were perturbed
    """
    positive = sum(list_success)
    if len(list_success) == 0:
        return 0
    return positive / len(list_success)


def create_list_success_confidence(list_probs: List[float], confidence_level: float) -> list:
    """
    Creates the list of success based on the confidence threshold - True if the probability is less or equal than confidence level.
    @param list_probs: the list of the probabilities
    @param confidence_level: the float which represents the confidence level
    @return: list of success based on the confidence threshold - True, if the attack was successful or False otherwise
    """
    list_success = []
    for x in list_probs:
        if x <= confidence_level:
            list_success.append(True)
        else:
            list_success.append(False)
    return list_success


def compute_success_rate_confidence(list_probs: list, confidence_level: float) -> float:
    """
    Computes the success rate based on the confidence threshold
    @param list_probs: the list of the probabilities
    @param confidence_level: the float value which represents the confidence level
    @return: the success rate based on the confidence threshold
    """
    list_success = create_list_success_confidence(list_probs, confidence_level)
    return compute_success_rate(list_success)


def compute_success_rate_confidence_and_labels(list_probs: list, confidence_level: float, list_success: list) -> float:
    """
    Computes the success rate based on the confidence level and the list of the success.
    @param list_probs: the list of the probabilities
    @param confidence_level: the float value which represents the confidence level
    @param list_success: the list which contains 0 and 1 and represents the success of the attack or not
    @return: the success rate based on the confidence and the list of the success
    """
    list_success_confidence = create_list_success_confidence(list_probs, confidence_level)
    list_success_confidence_np = np.array(list_success_confidence)
    list_success_np = np.array(list_success)

    # Perform element-wise OR operation
    result = np.logical_or(list_success_np, list_success_confidence_np).tolist()
    return compute_success_rate(result)


def compute_adversarial_confidence_probability(list_predicted_labels_targeted: list, list_probs_target: list) -> float:
    """
    Compute the Adversarial confidence  based on the predicted labels and the list of probabilities.
    The Adversarial Probability Labels (Confidence) is defined as:
    Accumulates the values of probability label of the target class for each successful perturbation,
    then divided by the total number of successful perturbations.
    The measure indicates the average confidence given by the target system when mis-classifying adversarial images.
    @param list_predicted_labels_targeted:  the list predicted labels
    @param list_probs_target:  the list of the probabilities
    @return: the float which represents the Adversarial Confidence
    """
    new_list = []
    for i in range(len(list_predicted_labels_targeted)):
        if list_predicted_labels_targeted[i]:
            new_list.append(list_probs_target[i])
    total_probability = sum(new_list)
    if len(new_list) == 0:
        return 0
    return total_probability / len(new_list)


def generate_numerical_results(predicted_labels_targeted: list, predicted_labels_non_targeted: list,
                               list_probs_original: list,
                               list_probs_target: list, confidence_level: float) -> dict:
    """
    Generates the numerical results of the attack, based on the predicted labels targeted and non-targeted,
    list probabilities of all the classes, and list probabilities of only targeted labels.
    @param predicted_labels_targeted: list of predicted labels targeted
    @param predicted_labels_non_targeted: list of predicted labels untargeted
    @param list_probs_original: list probabilities for all the classes
    @param list_probs_target: list probabilities of only target labels
    @param confidence_level: the float which represents the confidence level
    @return: a dictionary which contains all the computed metrics
    """
    acc_perturb_target = compute_success_rate(predicted_labels_targeted)
    acc_perturb_non_target = compute_success_rate(predicted_labels_non_targeted)
    acc_perturb_confidence = compute_success_rate_confidence(list_probs_original, confidence_level)
    acc_perturb_target_and_confidence = compute_success_rate_confidence_and_labels(list_probs_original,
                                                                                   confidence_level,
                                                                                   predicted_labels_targeted)
    acc_perturb_non_target_and_confidence = compute_success_rate_confidence_and_labels(list_probs_original,
                                                                                       confidence_level,
                                                                                       predicted_labels_non_targeted)
    adversarial_confidence_probability = compute_adversarial_confidence_probability(predicted_labels_targeted,
                                                                                    list_probs_target)

    results = {"success rate based on target label": acc_perturb_target,
               "success rate based on non-target label": acc_perturb_non_target,
               "success rate based on confidence level": acc_perturb_confidence,
               "success rate based on target label + confidence level": acc_perturb_target_and_confidence,
               "success rate based on non-target label + confidence level": acc_perturb_non_target_and_confidence,
               "adversarial confidence probability": adversarial_confidence_probability
               }
    return results

def create_results_json_v2(results: dict, config_str: str, type_attack: str, additional_info: dict, code: str) -> None:
    """
    Creates the Results json file to store the results.Those files are stored in the 'stats' folder. This is the V2 of the function and used for the one_pixel_attack2 and the patch attacks (the last versions of the Fit and Patch projects).
    @param results: the dictionary which contains the results
    @param config_str: the string which represents the configuration (used as the name of the file)
    @param type_attack: the string which represents the type of the attack
    @param additional_info: the dict which contains the additional info such as the model, dataset, crossover, mutation etc.
    @param code: the string which represents the code of the experiment, such that it can be easier identifiable
    """
    content = {"results": results,
               "additional info": additional_info,
               "code": code
               }
    filename = type_attack + '/'
    filename = (filename + "Res" + config_str)
    create_json_file(content, filename)


def create_json_file(content: dict, name: str) -> None:
    """
    Creates the json file in the 'results/stats' folder.
    @param content: the dict which contains the data which have to be stored
    @param name: the name of the json file
    """
    file = '../results/stats/' + name + '.json'
    with open(file, 'w') as json_file:
        json.dump(content, json_file, indent=4)
