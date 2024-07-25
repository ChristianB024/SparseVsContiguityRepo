from typing import List

import numpy as np


def create_captions(s: str) -> str:
    """
    Given the label, create the caption.
    @param s: the string which represents the label
    @return: the string which represents the captions
    """
    s = "a photo of {text}".format(text=s)
    return s


def create_captions_list(l: List[str]) -> List[str]:
    """
    Given a list of labels, create a list of captions from those labels.
    @param l: the list of the labels
    @return: the list of the corresponding captions for each label
    """
    nl = []
    for i in range(len(l)):
        nl.append(create_captions(l[i]))
    return nl

def create_labels_from_index(index_list: List[int], text_list: List[str]) -> List[str]:
    """
    Given the index list and the list of the all possible labels, create the list of the labels.
    @param index_list: the list of the indices (each index corresponds to an entry in the text_list)
    @param text_list: the list which represents all the possible labels in the dataset
    @return: the list of the labels for each entry in the index_list
    """
    labels = []
    for i in index_list:
        labels.append(text_list[i])
    return labels


def get_index_label(target_label: object, labels: List[object]) -> int:
    """
    Given the target_label, and the list of the labels, returns the corresponding index or -1 otherwise
    @param target_label: the string which represents the target label
    @param labels: the list of the labels
    @return: the index of the target label in the list of labels, or -1 otherwise
    """
    for i in range(len(labels)):
        if target_label == labels[i]:
            return i
    return -1
