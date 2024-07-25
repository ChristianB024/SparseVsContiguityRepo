from typing import List

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy.typing as npt
import numpy as np
import logging


# Function for plotting stuff
def v2_make_plot_from_preds(orig_preds: npt.NDArray, class_labels: List[str], colors: List[str] = ["forestgreen"]):
    """
    Make the plots for the predictions. Given the list of predictions.
    :param orig_preds: the original predictions
    :param class_labels: the list of the possible labels
    :param colors: the colors
    """
    width = 0.45

    for i in range(orig_preds.shape[0]):
        v_orig = orig_preds[i]
        alpha_orig = 0.6

        if np.argmax(orig_preds) == i:
            alpha_orig = 1.0

        plt.fill_between([0, v_orig], [i - width, i - width], [i + width, i + width], color=colors[0], alpha=alpha_orig)

    for i in range(len(class_labels)):
        text = class_labels[i]
        if orig_preds[i] > 0.3:
            text = text + " " + str(int(orig_preds[i] * 100)) + "%"
        plt.text(0, i - 0.3, text, fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks([], [])

    plt.xlabel("Probability", fontsize=16)
    plt.xlim([-0.015, 1.0])

def plot_adv_perturb_attack_single(original_preds: npt.NDArray, modified_preds_adv: npt.NDArray,
                                   original_label: List[str], modified_label: List[str],
                                   original_image: npt.NDArray, adv_image: npt.NDArray,
                                   perturbation_mask: npt.NDArray,
                                   class_names: List[str], index, attack_type: str, config: str = '') -> None:
    directory_perturb_plot = '../results/plots/' + attack_type
    # for id_to_show in range(modified_preds_adv.shape[0]):
    logging.info(
        f"Original Label {original_label} with probability {np.max(original_preds):.2f} and classified as {class_names[np.argmax(original_preds)]}")
    logging.info(
        f"Modified Label {modified_label} with probability {np.max(modified_preds_adv):.2f} and classified as {class_names[np.argmax(modified_preds_adv)]}")
    plt.figure(figsize=(6 * 3.5, 4.5), dpi=75)
    plt.subplot(1, 6, 2)
    plt.title("CIFAR-10 labels\nNo sticker", fontsize=16)
    v2_make_plot_from_preds(original_preds, class_names, colors=["forestgreen"])
    # plt.ylabel("Original dataset labels")


    # if adv_perturb_on_stickers:
    plt.subplot(1, 6, 4)
    plt.title("CIFAR-10 labels\nUsed to get adversary", fontsize=16)
    v2_make_plot_from_preds(modified_preds_adv, class_names, colors=["orangered"])
    # plt.ylabel("Original dataset labels")

    plt.subplot(1, 6, 1)
    plt.title("CIFAR-10 original image\n\"" + original_label + "\"",
              fontsize=16)

    plt.imshow(original_image.transpose([1, 2, 0]))
    plt.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.subplot(1, 6, 3)
    plt.title("Adversarial image\nFrom \"" + original_label + "\" to \"" +
              modified_label + "\"", fontsize=16)
    plt.imshow(adv_image.transpose([1, 2, 0]))
    plt.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.subplot(1, 6, 5)
    plt.title("Perturbation Mask", fontsize=16)
    plt.imshow(perturbation_mask.transpose([1, 2, 0]), cmap='viridis')
    plt.colorbar()
    plt.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.tight_layout()
    title = config + f'_{index}.png'
    plt.savefig(f'{directory_perturb_plot}/{title}')
    plt.clf()