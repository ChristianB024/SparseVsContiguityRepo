import logging

import torch

from model_util import get_logits_per_model_images_multi, get_probs_per_model_images_multi

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def hinge_loss_softmax(probs: torch.Tensor, target_class_index: int) -> torch.Tensor:
    """
    Implementation of the F4 fitness function. In the Patch project is noted as Ftar. It is a hinge function adapted from the ZOO paper.
    @param probs: the tensor with the probabilities for each label
    @param target_class_index: the index of the target class
    @return: the tensor with the fitness values
    """
    F_xt = probs[:, target_class_index]
    mask = torch.full_like(probs, float(0), dtype=torch.float32, device=device)
    mask[:, target_class_index] = float('-inf')
    masked_probs = probs + mask
    F_xi = masked_probs.max(dim=1).values
    return F_xt - F_xi


def hinge_loss_untargeted_softmax(probs: torch.Tensor, original_class_index: int) -> torch.Tensor:
    """
    Implementation of the Funtar (as it is noted in the Patch project). It is a hinge function adapted from the ZOO paper.
    @param probs: the tensor with the probabilities for each label
    @param original_class_index: the index of the target class
    @return:  the tensor with the fitness values
    """
    F_xt = probs[:, original_class_index]
    mask = torch.full_like(probs, float(0), dtype=torch.float32, device=device)
    mask[:, original_class_index] = float('-inf')
    masked_probs = probs + mask
    F_xi = masked_probs.max(dim=1).values
    return - F_xt + F_xi


def sum_weighted_loss_softmax(probs: torch.Tensor, target_class_index: int) -> torch.Tensor:
    """
    Implementation of the F5 fitness function.
    @param probs: the tensor with the probabilities for each label
    @param target_class_index: the index of the target class
    @return: the tensor with the fitness values
    """
    F_xt = probs[:, target_class_index]
    probs_squared = torch.pow(probs, 2)
    sum_value = torch.sum(probs_squared, dim=1) - probs_squared[:, target_class_index]
    return F_xt - sum_value


def sum_weighted_normalize_loss_softmax(probs: torch.Tensor, target_class_index: int) -> torch.Tensor:
    """
    Implementation of the F6 fitness function. Basically it is similar with F5, but we apply the normalized weights.
    @param probs: the tensor with the probabilities for each label
    @param target_class_index: the index of the target class
    @return: the tensor with the fitness values
    """
    F_xt = probs[:, target_class_index]
    F_xi = probs.clone()
    F_xi[:, target_class_index] = 0
    weights = torch.sum(F_xi, dim=1, keepdim=True)
    weights = F_xi / weights
    probs_weights = F_xi * weights
    sum_value = torch.sum(probs_weights, dim=1)
    return F_xt - sum_value


def loss_function1(probs: torch.Tensor, target_class_index: int) -> torch.Tensor:
    """
    Implementation of the F1 fitness function.
    @param probs: the tensor with the probabilities for each label
    @param target_class_index: the index of the target class
    @return: the tensor with the fitness values
    """
    return probs[:, target_class_index]


def loss_function2(probs: torch.Tensor, original_class_index: int) -> torch.Tensor:
    """
    Implementation of the F2 fitness function.
    @param probs: the tensor with the probabilities for each label
    @param original_class_index: the index of the original class
    @return: the tensor with the fitness values
    """
    return - probs[:, original_class_index]


def loss_function3(probs: torch.Tensor, target_class_index: int, original_class_index: int) -> torch.Tensor:
    """
    Implementation of the F3 fitness function.
    @param probs: the tensor with the probabilities for each label
    @param target_class_index: the index of the target class
    @param original_class_index: the index of the original class
    @return: the tensor with the fitness values
    """
    return probs[:, target_class_index] - probs[:, original_class_index]


def loss_function_least(probs: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the Fleast fitness function.
    @param probs: the tensor with the probabilities for each label
    @return: the tensor with the fitness values
    """
    min_probs, _ = torch.min(probs, dim=1)
    return min_probs


def loss_function_product(probs: torch.Tensor, target_class_index: int) -> torch.Tensor:
    """
    Implementation of the F7 fitness function.
    @param probs: the tensor with the probabilities for each label
    @param target_class_index: the index of the target class
    @return: the tensor with the fitness values
    """
    F_xt = probs[:, target_class_index]
    prod_value = torch.prod(probs, dim=1) / probs[:, target_class_index]
    return F_xt - prod_value


def loss_function_combine_static_weights(probs: torch.Tensor, target_class_index: int, rv: int) -> torch.Tensor:
    """
    Implementation of the F10 fitness function.
    @param probs: the tensor with the probabilities for each label
    @param target_class_index: the index of the target class
    @param rv: the random value which for the F10 represents the iteration step
    @return: the tensor with the fitness values
    """
    if rv <= 50:
        # If we are in the first 50 iterations out of 100, we are applying the F5 fitness function
        return sum_weighted_loss_softmax(probs, target_class_index)
    else:
        # Otherwise we apply the F7 fitness function
        return loss_function_product(probs, target_class_index)


def loss_function_combine_variable_weights_11(probs: torch.Tensor, target_class_index: int, rv: int) -> torch.Tensor:
    """
    Implementation of the F11 fitness function.
    @param probs: the tensor with the probabilities for each label
    @param target_class_index: the index of the target class
    @param rv: the random value which for the F11 represents the iteration step
    @return: the tensor with the fitness values
    """
    F_xt = probs[:, target_class_index]
    prod_value = torch.prod(probs, dim=1) / probs[:, target_class_index]
    probs_squared = torch.pow(probs, 2)
    sum_value = torch.sum(probs_squared, dim=1) - probs_squared[:, target_class_index]
    rv = rv / 100
    result = F_xt - rv * (sum_value) - (1 - rv) * prod_value
    return result


def loss_function_combine_variable_weights_12(probs: torch.Tensor, target_class_index: int, rv: int) -> torch.Tensor:
    """
    Implementation of the F12 fitness function.
    @param probs: the tensor with the probabilities for each label
    @param target_class_index: the index of the target class
    @param rv: the random value which for the F12 represents the iteration step
    @return: the tensor with the fitness values
    """
    F_xt = probs[:, target_class_index]
    prod_value = torch.prod(probs, dim=1) / probs[:, target_class_index]
    probs_squared = torch.pow(probs, 2)
    sum_value = torch.sum(probs_squared, dim=1) - probs_squared[:, target_class_index]
    rv = rv / 100
    result = F_xt - rv * (prod_value) - (1 - rv) * sum_value
    return result


def check_gpu_variable(x: object):
    """
    This function is used for debugging purposes, to check if a specific variable is on GPU or CPU.
    @param x: the variable for which we test if it is stored on the GPU or CPU
    """
    if x.is_cuda:
        logging.info("Variable is on GPU")
    else:
        logging.info("Variable is on CPU")


def multi_loss_functions_official_paper(probs: torch.Tensor, target_class_index: int, original_class_index: int,
                                        random_function_id: int) -> torch.Tensor:
    """
    Implementation of the F8 fitness function. This function at each iteration is choosing a specific fitness function. The function is the same as the original paper.
    @param probs: the tensor with the probabilities for each label
    @param target_class_index: the index of the target class
    @param original_class_index: the index of the original class
    @param random_function_id: the random value which for the F8 represents the index of the function which needs to be used
    @return: the tensor with the fitness values
    """
    if random_function_id == 0:
        value = loss_function1(probs, target_class_index)
    elif random_function_id == 1:
        value = loss_function2(probs, original_class_index)
    elif random_function_id == 2:
        value = loss_function3(probs, target_class_index, original_class_index)
    elif random_function_id == 3:
        value = loss_function_least(probs)
    else:
        raise Exception('We did not implement any other loss functions!!!!')
    return value


def multi_loss_functions_custom(probs, target_class_index, original_class_index, random_function_id):
    """
    Implementation of the F9 fitness function. This function at each iteration is choosing a specific fitness function. The function is adapted from the original paper one.
    @param probs: the tensor with the probabilities for each label
    @param target_class_index: the index of the target class
    @param original_class_index: the index of the original class
    @param random_function_id: the random value which for the F8 represents the index of the function which needs to be used
    @return: the tensor with the fitness values
    """
    if random_function_id == 0:
        value = loss_function1(probs, target_class_index)
    elif random_function_id == 1:
        value = loss_function2(probs, original_class_index)
    elif random_function_id == 2:
        value = loss_function3(probs, target_class_index, original_class_index)
    elif random_function_id == 3:
        value = loss_function_least(probs)
    elif random_function_id == 4:
        value = hinge_loss_softmax(probs, target_class_index)
    elif random_function_id == 5:
        value = sum_weighted_loss_softmax(probs, target_class_index)
    elif random_function_id == 6:
        value = sum_weighted_normalize_loss_softmax(probs, target_class_index)
    elif random_function_id == 7:
        value = loss_function_product(probs, target_class_index)
    else:
        raise Exception('We did not implement any other loss functions!!!!')
    return value


def get_output_model(fitness_function: str, imgs_perturbated, model: object, text_features: object) -> torch.Tensor:
    """
    Returns the output from the model, for the inference stage. The function is optimized to work with images in batches.
    @param fitness_function: the string which represent the fitness function. The fitness function must contain as a substring
    'logits' for accessing the last layer of the model (white-box), 'softmax' for receiving the probabilities of the classes (black-box)
    or 'logprobs' for accessing the log probabilities (black-box)
    @param imgs_perturbated: the tensor which represents the images which are perturbed and need to be fetched into the model for classification
    @param model: the model for which the inference is done
    @param text_features: the tensor which represents the textual features
    @return: the tensor which represents the probabilities/log probabilities or logits values, depending on the fitness function
    """
    if 'logits' in fitness_function:
        probs = get_logits_per_model_images_multi(imgs_perturbated, model, text_features)
    elif 'softmax' in fitness_function:
        probs = get_probs_per_model_images_multi(imgs_perturbated, model, text_features)
    elif 'logprobs' in fitness_function:
        probs = get_probs_per_model_images_multi(imgs_perturbated, model, text_features)
        # Adding a regularization term for stability - 1e-30
        probs = torch.log(probs + 1e-30)
    else:
        raise Exception('We did not implement any other fitness functions...')
    return probs


def fitness_function_version(fitness_function: str, random_value: int, imgs_perturbed: torch.Tensor, model: object,
                             text_features: torch.Tensor, target_class_index: int, original_class_index: int) -> torch.Tensor:
    """
    Computes the fitness values for all images fetched to the model
    @param fitness_function: the string which represents the fitness function. It should contain substrings:
    For the type of the attack:
        'logits' for accessing the last layer of the model (white-box),
        'softmax' for receiving the probabilities of the classes (black-box)
        or 'logprobs' for accessing the log probabilities (black-box)
    For the function encoding:
        'loss10' for F10
        'loss11' for F11
        'loss12' for F12
        'loss13' for Funtar
        'loss1' for F1
        'loss2' for F2
        'loss3' for F3
        'loss4' for F4/Ftar
        'loss5' for F5
        'loss6' for F6
        'loss7' for F7
        'loss8' for F8
        'loss9' for F9
    @param random_value: the integer which represents the random value and used for specific fitness functions
    @param imgs_perturbed: the tensor which represents all the images which need to be fed to the model
    @param model: the model for which the inference is done
    @param text_features: the tensor which represnts the textual features
    @param target_class_index: the index of the target class
    @param original_class_index: the index of the original class
    @return: the tensor which represents the fitness values
    """
    probs = get_output_model(fitness_function, imgs_perturbed, model, text_features)
    # check_gpu_variable(probs)
    # F10
    if 'loss10' in fitness_function:
        value = loss_function_combine_static_weights(probs, target_class_index, random_value)
    # F11
    elif 'loss11' in fitness_function:
        value = loss_function_combine_variable_weights_11(probs, target_class_index, random_value)
    # F12
    elif 'loss12' in fitness_function:
        value = loss_function_combine_variable_weights_12(probs, target_class_index, random_value)
    # Untargeted version of softmax4
    elif 'loss13' in fitness_function:
        value = hinge_loss_untargeted_softmax(probs, original_class_index)
    # F1
    elif 'loss1' in fitness_function:
        value = loss_function1(probs, target_class_index)
    # F2
    elif 'loss2' in fitness_function:
        value = loss_function2(probs, original_class_index)
    # F3
    elif 'loss3' in fitness_function:
        value = loss_function3(probs, target_class_index, original_class_index)
    # F4
    elif 'loss4' in fitness_function:
        value = hinge_loss_softmax(probs, target_class_index)
    # F5
    elif 'loss5' in fitness_function:
        value = sum_weighted_loss_softmax(probs, target_class_index)
    # F6
    elif 'loss6' in fitness_function:
        value = sum_weighted_normalize_loss_softmax(probs, target_class_index)
    # F7
    elif 'loss7' in fitness_function:
        value = loss_function_product(probs, target_class_index)
    # F8
    elif 'loss8' in fitness_function:
        value = multi_loss_functions_official_paper(probs, target_class_index, original_class_index, random_value)
    # F9
    elif 'loss9' in fitness_function:
        value = multi_loss_functions_custom(probs, target_class_index, original_class_index, random_value)
    else:
        raise Exception('We did not implement any fitness functions with the specified loss and index')
    return value


def get_random_vector(fitness_function: str) -> list:
    """
    Create the random vector which can be used for some functions.
    1. Most of the functions which are not using any weight or randomness, are having an empty random vector
    2. Loss10, Loss11, Loss12 are the losses functions which are using the weights for each iteration. Therefore,
    their random functions will represent the indices of the max iterations.
    3, Loss8, Loss9 are using the random functions at each iteration and the random vector list contains
    the list of the indices of the fitness functions which can be used.
    @param fitness_function: the string which represents the fitness function
    @return: the list which represents the random vector
    """
    if 'loss10' in fitness_function:
        numbers_list = [i for i in range(1, 101)]
        return numbers_list
    elif 'loss11' in fitness_function:
        numbers_list = [i for i in range(1, 101)]
        return numbers_list
    elif 'loss12' in fitness_function:
        numbers_list = [i for i in range(1, 101)]
        return numbers_list
    # Untargeted version of softmax4
    elif 'loss13' in fitness_function:
        return []
    elif 'loss1' in fitness_function:
        return []
    elif 'loss2' in fitness_function:
        return []
    elif 'loss3' in fitness_function:
        return []
    elif 'loss4' in fitness_function:
        return []
    elif 'loss5' in fitness_function:
        return []
    elif 'loss6' in fitness_function:
        return []
    elif 'loss7' in fitness_function:
        return []
    elif 'loss8' in fitness_function:
        return [0, 1, 2, 3]
    elif 'loss9' in fitness_function:
        return [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        raise Exception('We did not implement any fitness functions with the specified loss and index')


def imperceptible_function0(imgs_perturbed: torch.Tensor, img_original: torch.Tensor) -> torch.Tensor:
    """
    Compute the difference (L2 norm) between the perturbed image and the original one.
    The computation of L2 norm is done all on 3 dimensions.
    @param imgs_perturbed: the tensor which represents the perturbed image
    @param img_original: the tensor which represents the original image
    @return: the tensor which represents the -L2 norms
    """
    l2_norms = torch.norm(imgs_perturbed - img_original, p=2, dim=(1, 2, 3))
    return -l2_norms
