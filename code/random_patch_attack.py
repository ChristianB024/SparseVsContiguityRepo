import numpy as np

import torch

from adversarial_perturbation import generate_target_labels, generate_attacking_setup
from captions_utils import get_index_label, create_captions_list
import log_config
from fit import get_output_model
from data_preprocess import load_preprocessed_data
from model_util import classify_image_and_probs

torch.manual_seed(42)

from model_dict import get_model, get_model_image_processor_bounds

from plots import plot_adv_perturb_attack_single
from metrics import generate_numerical_results, create_results_json_v2

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

from random_evolution import random_evolution
import logging


###############################################################################################################
# This is the correct implementation of the Random Contiguous Attack - attacking preprocessed images
#################################################################################################################


def perturb_image_patch(xs: torch.Tensor, img: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """
    Creates the perturbed images based on the agents and the original image.
    This function does the Patch Attack.
    @param xs: the tensor which represents all the agents
    @param img: the tensor which represents the original images
    @param w: the integer which represents the width
    @param h: the integer which represents the height
    @return: the tensor which represents all the perturbed images derived from the agents and the original image
    """
    if xs.ndim < 2:
        xs = xs.unsqueeze(0)
    batch = xs.size(0)

    imgs = img.repeat(batch, 1, 1, 1)

    count = 0
    for x in xs:
        x_pos = int(x[0])
        y_pos = int(x[1])
        pixels = x[2:]
        pixels = torch.chunk(pixels, len(pixels) // 3)
        j = y_pos
        index = 0
        while j < y_pos + h:
            i = x_pos
            while i < x_pos + w:
                r, g, b = pixels[index]
                index = index + 1
                imgs[count, 0, j, i] = (r / 255.0 - 0.4914) / 0.2023
                imgs[count, 1, j, i] = (g / 255.0 - 0.4822) / 0.1994
                imgs[count, 2, j, i] = (b / 255.0 - 0.4465) / 0.2010
                i = i + 1
            j = j + 1
        count += 1
    return imgs


def perturb_image_row(xs, img):
    """
    Creates the perturbed images based on the agents and the original image.
    This function does the Row Attack.
    @param xs: the tensor which represents all the agents
    @param img: the tensor which represents the original images
    @return: the tensor which represents all the perturbed images derived from the agents and the original image
    """
    if xs.ndim < 2:
        xs = xs.unsqueeze(0)
    batch = xs.size(0)
    imgs = img.repeat(batch, 1, 1, 1)
    count = 0
    for x in xs:
        x_pos = int(x[0])
        y_pos = int(x[1])
        pixels = x[2:]
        pixels = torch.chunk(pixels, len(pixels) // 3)
        for p in pixels:
            r, g, b = p
            imgs[count, 0, y_pos, x_pos] = (r / 255.0 - 0.4914) / 0.2023
            imgs[count, 1, y_pos, x_pos] = (g / 255.0 - 0.4822) / 0.1994
            imgs[count, 2, y_pos, x_pos] = (b / 255.0 - 0.4465) / 0.2010
            x_pos = x_pos + 1
        count += 1
    return imgs


def perturb_image_column(xs, img):
    """
    Creates the perturbed images based on the agents and the original image.
    This function does the Column Attack.
    @param xs: the tensor which represents all the agents
    @param img: the tensor which represents the original images
    @return: the tensor which represents all the perturbed images derived from the agents and the original image
    """
    if xs.ndim < 2:
        xs = xs.unsqueeze(0)
    batch = xs.size(0)
    imgs = img.repeat(batch, 1, 1, 1)
    count = 0
    for x in xs:
        x_pos = int(x[0])
        y_pos = int(x[1])
        pixels = x[2:]
        pixels = torch.chunk(pixels, len(pixels) // 3)
        for p in pixels:
            r, g, b = p
            imgs[count, 0, y_pos, x_pos] = (r / 255.0 - 0.4914) / 0.2023
            imgs[count, 1, y_pos, x_pos] = (g / 255.0 - 0.4822) / 0.1994
            imgs[count, 2, y_pos, x_pos] = (b / 255.0 - 0.4465) / 0.2010
            y_pos = y_pos + 1
        count += 1
    return imgs


def perturb_image_diag(xs, img):
    """
    Creates the perturbed images based on the agents and the original image.
    This function does the Diagonal Attack.
    @param xs: the tensor which represents all the agents
    @param img: the tensor which represents the original images
    @return: the tensor which represents all the perturbed images derived from the agents and the original image
    """
    if xs.ndim < 2:
        xs = xs.unsqueeze(0)
    batch = xs.size(0)
    imgs = img.repeat(batch, 1, 1, 1)
    count = 0
    for x in xs:
        x_pos = int(x[0])
        y_pos = int(x[1])
        pixels = x[2:]
        pixels = torch.chunk(pixels, len(pixels) // 3)
        for p in pixels:
            r, g, b = p
            imgs[count, 0, y_pos, x_pos] = (r / 255.0 - 0.4914) / 0.2023
            imgs[count, 1, y_pos, x_pos] = (g / 255.0 - 0.4822) / 0.1994
            imgs[count, 2, y_pos, x_pos] = (b / 255.0 - 0.4465) / 0.2010
            y_pos = y_pos + 1
            x_pos = x_pos + 1
        count += 1
    return imgs


def perturb_image_anti_diag(xs, img):
    """
    Creates the perturbed images based on the agents and the original image.
    This function does the Anti-Diagonal Attack.
    @param xs: the tensor which represents all the agents
    @param img: the tensor which represents the original images
    @return: the tensor which represents all the perturbed images derived from the agents and the original image
    """
    if xs.ndim < 2:
        xs = xs.unsqueeze(0)
    batch = xs.size(0)
    imgs = img.repeat(batch, 1, 1, 1)
    count = 0
    for x in xs:
        x_pos = int(x[0])
        y_pos = int(x[1])
        pixels = x[2:]
        pixels = torch.chunk(pixels, len(pixels) // 3)
        for p in pixels:
            r, g, b = p
            imgs[count, 0, y_pos, x_pos] = (r / 255.0 - 0.4914) / 0.2023
            imgs[count, 1, y_pos, x_pos] = (g / 255.0 - 0.4822) / 0.1994
            imgs[count, 2, y_pos, x_pos] = (b / 255.0 - 0.4465) / 0.2010
            y_pos = y_pos - 1
            x_pos = x_pos + 1
        count += 1
    return imgs


def predict_classes_multi(type_attack: str, xs: torch.Tensor, img: torch.Tensor, target_class_index: int,
                          original_class_index: int, targeted: bool, model,
                          text_features, fitness_function: str, w: int, h: int) -> tuple:
    """
    Checks if the random attack is successful or not, and if that is the case it returns the index of the best perturbation.
    @param type_attack: the string which represents the type of the attack (random_patch, random_row etc.')
    @param xs: the perturbations encodings
    @param img: the tensor which represents the preprocessed original image
    @param target_class_index: the index of the target class
    @param original_class_index: the index of the original class
    @param targeted: the bool which refers to the targeted or non-targeted attacks
    @param model: the model which we attack
    @param text_features: the text features
    @param fitness_function: the string which represents the fitness function
    @param w: the integer which represents the width
    @param h: the integer which represents the height
    @return: the tensor which represents the probabilities of the classes for each perturbed image
    """
    if type_attack == 'random_patch':
        imgs_perturbed = perturb_image_patch(xs, img.clone(), w, h)
    elif type_attack == 'random_row':
        imgs_perturbed = perturb_image_row(xs, img.clone())
    elif type_attack == 'random_column':
        imgs_perturbed = perturb_image_column(xs, img.clone())
    elif type_attack == 'random_diag':
        imgs_perturbed = perturb_image_diag(xs, img.clone())
    elif type_attack == 'random_anti_diag':
        imgs_perturbed = perturb_image_anti_diag(xs, img.clone())
    else:
        raise Exception('This attack is not yet implemented')
    model.zero_grad()
    with torch.no_grad():
        probs = get_output_model(fitness_function, imgs_perturbed, model, text_features)
        probs = probs.to(device)
        best_indices = torch.argmax(probs, dim=1)

        if targeted:
            tensor_target_class_index = torch.tensor(target_class_index, device=device)
            indices_target = torch.nonzero(best_indices == tensor_target_class_index, as_tuple=False).flatten()
            if indices_target.numel() != 0:
                return True, indices_target[0].item()
            else:
                return False, 0
        else:
            tensor_original_class_index = torch.tensor(original_class_index, device=device)
            indices_original = torch.nonzero(best_indices != tensor_original_class_index, as_tuple=False).flatten()
            if indices_original.numel() != 0:
                return True, indices_original[0]
            else:
                return False, 0


def random_attack_multiple(type_attack: str, model, tokenizer, images: list, image_bounds: list,
                           candidate_captions: list,
                           original_captions: list, target_captions: list, targeted: bool, config_string: str,
                           pixels: int, popsize: int, maxiter: int, fitness_function: str,
                           save_plot: bool = True, w: int = 0, h: int = 0):
    """
    Launches the Random one-pixel attacks (perturbations on the preprocessed image) for multiple images.
    @param type_attack: the string which represents the type of the attack (random_patch, random_row etc.')
    @param model: the model which we attack
    @param tokenizer: the textual encoder (tokenizer)
    @param images: the list of the images which we attack
    @param image_bounds: the list which represents the image bounds
    @param candidate_captions: the list of the candidate captions
    @param original_captions: the list of the original captions
    @param target_captions: the list of the target captions
    @param targeted: the bool value which indicates targeted attacks (True) and non-targeted attacks (False)
    @param config_string: the string which represents the config string for plots
    @param pixels: the number of perturbed pixels
    @param popsize: the population size
    @param maxiter: the number of max iterations
    @param save_plot: the boolean value which is True if we want to save the plots, False otherwise
    @param w: the integer which represents the width
    @param h: the integer which represents the height
    @return: the tuple which represents the list of predicted targeted labels, the list of predicted non-target, list of original probabilities and list of the perturbed probabilities
   """
    if tokenizer is not None:
        with torch.no_grad():
            inputs_text = tokenizer(candidate_captions, padding=True, return_tensors="pt").to(device)
            text_features = model.get_text_features(**inputs_text)
            text_features_normalized = text_features / text_features.norm(dim=1, keepdim=True)
    else:
        text_features_normalized = None
    logging.info('The text_features are built for the one pixel attack')
    for p in model.parameters():
        p.requires_grad = False

    list_predicted_labels_targeted = []
    list_predicted_labels_non_targeted = []
    list_probs_original = []
    list_probs_target = []
    for i in range(len(target_captions)):
        image_gpu = images[i].clone().to(device=device)
        target_label_index = get_index_label(target_captions[i], candidate_captions)
        original_label_index = get_index_label(original_captions[i], candidate_captions)
        adv_image_tensor, attack_success = random_attack_single(type_attack, model, text_features_normalized, image_gpu,
                                                                image_bounds,
                                                                target_label_index, original_label_index, targeted,
                                                                pixels, popsize, maxiter, fitness_function, w, h)
        # Check if they have the same values
        # adv_image_tensor_process = unpacking_pixel_values(**processor(images=adv_image_tensor, return_tensors="pt")).to(
        #     device)
        # image_gpu_process = unpacking_pixel_values(**processor(images=images[i], return_tensors="pt")).to(device)
        result_perturbation = image_gpu - adv_image_tensor

        prediction_original, probs_original = classify_image_and_probs(model, text_features_normalized,
                                                                       image_gpu, candidate_captions)
        prediction_adv, probs_adv = classify_image_and_probs(model, text_features_normalized, adv_image_tensor,
                                                             candidate_captions)
        # # Transform everything into numpy for the plots
        image_np = np.concatenate(image_gpu.detach().cpu().numpy(), axis=0)
        adv_image_np = np.concatenate(adv_image_tensor.detach().cpu().numpy(), axis=0)
        result_perturbation_np = np.concatenate(result_perturbation.detach().cpu().numpy(), axis=0)
        probs_original_np = np.concatenate(probs_original.detach().cpu().numpy(), axis=0)
        probs_adv_np = np.concatenate(probs_adv.detach().cpu().numpy(), axis=0)

        if prediction_adv == target_captions[i]:
            list_predicted_labels_targeted.append(True)
        else:
            list_predicted_labels_targeted.append(False)
        if prediction_adv != prediction_original:
            list_predicted_labels_non_targeted.append(True)
        else:
            list_predicted_labels_non_targeted.append(False)

        list_probs_original.append(probs_adv_np[original_label_index])
        list_probs_target.append(probs_adv_np[target_label_index])
        if save_plot:
            plot_adv_perturb_attack_single(probs_original_np, probs_adv_np, original_captions[i], target_captions[i],
                                           image_np, adv_image_np, result_perturbation_np, candidate_captions, i,
                                           type_attack, config_string)
    return list_predicted_labels_targeted, list_predicted_labels_non_targeted, list_probs_original, list_probs_target


def random_attack_single(type_attack: str, model, text_features, image, image_bounds: list, target_caption_index: int,
                         original_caption_index, targeted: bool, pixels: int, popsize: int, maxiter: int,
                         fitness_function: str, w: int = 0, h: int = 0) -> tuple:
    """
    Launches the Random one-pixel attacks (perturbations on the preprocessed image) for multiple images.
    @param type_attack: the string which represents the type of the attack (random_patch, random_row etc.')
    @param model: the model which we attack
    @param text_features: the textual features from the captions
    @param image: the image which we attack
    @param image_bounds: the list which represents the image bounds
    @param target_caption_index: the index of the target captions
    @param original_caption_index: the index of the original captions
    @param targeted: the bool value which indicates targeted attacks (True) and non-targeted attacks (False)
    @param pixels: the number of perturbed pixels
    @param popsize: the population size
    @param maxiter: the number of max iterations
    @param fitness_function: the string which represents the fitness function
    @param w: the integer which represents the width
    @param h: the integer which represents the height
    @return: a tuple which represents the best perturbed image and also the attack success
    """
    pixel_positions_bounds = [image_bounds[0], image_bounds[1]]
    color_positions_bounds = [image_bounds[2], image_bounds[3], image_bounds[4]]
    patch_pixels = pixels
    color_bounds = torch.tensor(color_positions_bounds * patch_pixels).to(device)
    positions_bounds = torch.tensor(pixel_positions_bounds).to(device)
    bounds = torch.cat((positions_bounds, color_bounds), dim=0)
    predict_fn = lambda xs: predict_classes_multi(type_attack,
                                                  xs, image, target_caption_index, original_caption_index, targeted,
                                                  model, text_features, fitness_function, w, h)

    attack_success, attack_result = random_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popsize)
    if type_attack == 'random_patch':
        attack_image = perturb_image_patch(attack_result, image.clone(), w, h)
    elif type_attack == 'random_row':
        attack_image = perturb_image_row(attack_result, image.clone())
    elif type_attack == 'random_column':
        attack_image = perturb_image_column(attack_result, image.clone())
    elif type_attack == 'random_diag':
        attack_image = perturb_image_diag(attack_result, image.clone())
    elif type_attack == 'random_anti_diag':
        attack_image = perturb_image_anti_diag(attack_result, image.clone())
    else:
        raise Exception('This type of the attack is not implemented yet!!!')
    attack_image = attack_image.to(device)

    return attack_image, attack_success


def configuration_string_plot(code: str, pretrained_model: str, dataset: str, generate_captions: bool, type_target: str,
                              targeted: bool, pixels: int, popsize: int, maxiter: int, fitness_function:str, w:int,
                              h:int) -> str:
    """
    Creates the configuration string for the plots.
    @param code: the string which represents the code of the experiment, such that it can be easier identifiable
    @param pretrained_model: the string which represents the pretrained model
    @param dataset: the string which represents the dataset
    @param generate_captions: bool value which is True if captions where generated, False otherwise
    @param type_target: the string which represents the type of the target -> 'second'/'random'
    @param targeted: the bool value which indicates targeted attacks (True) and non-targeted attacks (False)
    @param pixels: the number of perturbed pixels
    @param popsize: the population size
    @param maxiter: the number of max iterations
    @param fitness_function: the string which represents the fitness function
    @param w: the integer which represents the width
    @param h: the integer which represents the height
    @return: the configuration string which will be used in the plots
    """
    config = f"{pretrained_model}_{dataset}_gc_{generate_captions}_tt_{type_target}_tar_{targeted}_px_{pixels}_pops_{popsize}_maxit_{maxiter}_fit_{fitness_function}"
    if w != 0:
        config = config + f"_w_{w}"
    if h != 0:
        config = config + f"_h_{h}"
    config = config + "_" + code
    return config


def create_dictionary_random(pretrained_model:str, dataset:str, generate_captions:bool, type_target:str, targeted:bool, save_plot:bool, pixels:int, popsize:int, maxiter:int, fitness_function:str) -> dict:
    """
    Creates the dictionary used in the results for the Random Contiguous one pixel attack v2 (attacking preprocessed images).
    @param pretrained_model: the string which represents the pretrained model
    @param dataset: the string which represents the dataset
    @param generate_captions: bool value which is True if captions where generated, False otherwise
    @param type_target: the string which represents the type of the target -> 'second'/'random'
    @param targeted: the bool value which indicates targeted attacks (True) and non-targeted attacks (False)
    @param save_plot: the boolean value which is True if we want to save the plots, False otherwise
    @param pixels: the number of perturbed pixels
    @param popsize: the population size
    @param maxiter: the number of max iterations
    @param fitness_function: the string which represents the fitness function
    @return: the dictionary which contains the configuration of the attack
    """
    dict_add = dict()
    dict_add['pretrained_model'] = pretrained_model
    dict_add['dataset'] = dataset
    dict_add['generate_captions'] = generate_captions
    dict_add['type_target'] = type_target
    dict_add['targeted'] = targeted
    dict_add['pixels'] = pixels
    dict_add['popsize'] = popsize
    dict_add['maxiter'] = maxiter
    dict_add['fitness_function'] = fitness_function
    dict_add['save_plot'] = save_plot
    return dict_add


def get_image_bounds_rectangle_shrink(image_bounds: list, w: int, h: int) -> list:
    """
    Creates the image bounds in case we attack using the Patch.
    @param image_bounds: the list which represents the image bounds
    @param w: the integer which represents the width
    @param h: the integer which represents the height
    @return: the list which represents the new image bounds
    """
    modified_list = []
    for i, (x, y) in enumerate(image_bounds):
        if i == 0:
            modified_list.append((x, y - w))
        elif i == 1:
            modified_list.append((x, y - h))
        else:
            modified_list.append((x, y))
    return modified_list


def get_image_bounds_anti_diag(image_bounds: list, pixels: int) -> list:
    """
    Creates the image bounds in case we attack using the Anti-Diagonal.
    @param image_bounds: the list which represents the image bounds
    @param pixels: the number of pixels we want to perturb
    @return: the list which represents the new image bounds
    """
    modified_list = []
    for i, (x, y) in enumerate(image_bounds):
        if i == 0:
            modified_list.append((x, y - pixels))
        elif i == 1:
            modified_list.append((x + pixels, y))
        else:
            modified_list.append((x, y))
    return modified_list


def attack(type_attack:str, pretrained_model: str, dataset: str, generate_captions: bool, type_target: str,
           save_plot: bool, pixels: int, popsize: int, maxiter: int, fitness_function:str, targeted:bool=True,
           w:int=0, h:int=0, code:str='0')-> None:
    """
    The main attack function for the Random Contiguous attack (one-pixel attacking the preprocessed images using shapes).
    @param type_attack: the string which represents the type of the attack ('one_patch, one_row etc.')
    @param pretrained_model: the string which represents the model
    @param dataset: the string which represents the dataset
    @param generate_captions: bool value which is True if captions where generated, False otherwise
    @param type_target: the string which represents the type of the target -> 'second'/'random'
    @param save_plot: the boolean value which is True if we want to save the plots, False otherwise
    @param pixels: the number of perturbed pixels
    @param popsize: the population size
    @param maxiter: the number of max iterations
    @param fitness_function: the string which represents the fitness function
    @param targeted: the bool value which indicates targeted attacks (True) and non-targeted attacks (False)
    @param w: the integer which represents the width
    @param h: the integer which represents the height
    @param code: the string which represents the code of the experiment, such that it can be easier identifiable
    """
    logging.info('Starting getting the model...')
    model, processor, tokenizer = get_model(pretrained_model)
    if tokenizer is None:
        generate_captions = False

    image_bounds = get_model_image_processor_bounds(pretrained_model)
    if type_attack == 'random_row':
        image_bounds = get_image_bounds_rectangle_shrink(image_bounds, pixels, 1)
        # get_image_bounds_rectangle_shrink(image_bounds, 1, pixels)
    elif type_attack == 'random_column':
        image_bounds = get_image_bounds_rectangle_shrink(image_bounds, 1, pixels)
    elif type_attack == 'random_patch':
        pixels = w * h
        image_bounds = get_image_bounds_rectangle_shrink(image_bounds, w, h)
    elif type_attack == 'random_diag':
        image_bounds = get_image_bounds_rectangle_shrink(image_bounds, pixels, pixels)
    elif type_attack == 'random_anti_diag':
        image_bounds = get_image_bounds_anti_diag(image_bounds, pixels)
    else:
        raise Exception('We did not implement this type of the attack')
    model.to(device)
    logging.info('Finalizing getting the model')

    logging.info('Starting getting the data...')
    x_samples, y_samples, candidate_labels = load_preprocessed_data(pretrained_model, dataset, generate_captions)
    candidate_captions = candidate_labels
    # # For testing locally in the future
    # x_samples = x_samples[:2]
    # y_samples = y_samples[:2]

    if generate_captions:
        candidate_captions = create_captions_list(candidate_labels)
    logging.info('Finalizing getting the data...')
    logging.info('Starting generate the attacking samples...')
    x_samples, y_samples, probs = generate_attacking_setup(x_samples, y_samples, candidate_labels, model, tokenizer,
                                                           candidate_captions)
    logging.info('Finalizing generate the attacking samples...')
    target_labels = generate_target_labels(type_target, probs, candidate_labels, y_samples)

    captions_original_labels = y_samples
    captions_target_labels = target_labels
    if generate_captions:
        captions_target_labels = create_captions_list(target_labels)
        captions_original_labels = create_captions_list(y_samples)
    logging.info('Create the configuration string')
    config_str = configuration_string_plot(code, pretrained_model, dataset, generate_captions, type_target, targeted,
                                           pixels, popsize, maxiter, fitness_function, w, h)
    logging.info('Launch the attack.....')
    predicted_labels_targeted, predicted_labels_non_targeted, list_probs_original, list_probs_target = random_attack_multiple(
        type_attack, model, tokenizer, x_samples, image_bounds, candidate_captions, captions_original_labels,
        captions_target_labels, targeted, config_str, pixels, popsize, maxiter, fitness_function, save_plot, w, h)
    logging.info("Done with the attack")
    logging.info("Compute the results")
    results = generate_numerical_results(predicted_labels_targeted, predicted_labels_non_targeted, list_probs_original,
                                         list_probs_target, 0.3)
    add_dictionary = create_dictionary_random(pretrained_model, dataset, generate_captions, type_target, targeted,
                                              save_plot, pixels, popsize, maxiter, fitness_function)

    create_results_json_v2(results, config_str, type_attack, add_dictionary, code)
    logging.info("Finalizing the experiments")


if __name__ == "__main__":
    # pretrained_model = 'CLIP_ViT-B32'
    pretrained_model = "CLIP_ViT-B32"
    dataset = "cifar10"
    generate_captions = True
    type_target = "second"

    pixels = 2
    popsize = 20
    maxiter = 10
    crossover = 0.8
    mutation = 0.9
    save_plot = True
    fitness_function = "softmax"
    log_file = "row_t"
    log_type = "info"
    code = "row_t"
    targeted = True
    type_attack = 'random_patch'

    log_config.configure_log_info(log_file)
    attack(type_attack, pretrained_model, dataset, generate_captions, type_target, save_plot, pixels, popsize, maxiter,
           fitness_function, targeted, 10, 10, code)
