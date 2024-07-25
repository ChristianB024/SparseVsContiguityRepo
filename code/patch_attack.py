import numpy as np

import torch

from adversarial_perturbation import generate_target_labels, generate_attacking_setup
from captions_utils import get_index_label, create_captions_list
from fit import get_random_vector, fitness_function_version
from data_preprocess import load_preprocessed_data
from model_util import classify_image_and_probs, get_probs_per_model_images_multi

torch.manual_seed(42)

from model_dict import get_model, get_model_image_processor_bounds

from plots import plot_adv_perturb_attack_single
from metrics import generate_numerical_results, create_results_json_v2

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

from differential_evolution import differential_evolution

import logging


###############################################################################################################
# This is the correct implementation of the Contiguous Attack - attacking preprocessed images
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


def perturb_image_column(xs:torch.Tensor, img:torch.Tensor) -> torch.Tensor:
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


def predict_classes_multi(xs: torch.Tensor, random_value: int, img: torch.Tensor, target_class_index: int,
                          original_class_index: int, targeted: bool, model,
                          text_features, fitness_function: str, type_attack: str, w: int, h: int) -> torch.Tensor:
    """
    Computes the probabilities of the classes for the perturbed images.
    @param xs: the perturbations encodings
    @param random_value: the integer which represents the random value and used for specific fitness functions
    @param img: the tensor which represents the preprocessed original image
    @param target_class_index: the index of the target class
    @param original_class_index: the index of the original class
    @param targeted: the bool which refers to the targeted or non-targeted attacks
    @param model: the model which we attack
    @param text_features: the text features
    @param fitness_function: the string which represents the fitness function
    @param type_attack: the string which represents the type of the attack ('one_patch, one_row etc.')
    @param w: the integer which represents the width
    @param h: the integer which represents the height
    @return: the tensor which represents the probabilities of the classes for each perturbed image
    """
    if type_attack == 'one_patch':
        imgs_perturbed = perturb_image_patch(xs, img.clone(), w, h)
    elif type_attack == 'one_row':
        imgs_perturbed = perturb_image_row(xs, img.clone())
    elif type_attack == 'one_column':
        imgs_perturbed = perturb_image_column(xs, img.clone())
    # check_gpu_variable(imgs_perturbed)
    elif type_attack == 'one_diag':
        imgs_perturbed = perturb_image_diag(xs, img.clone())
    elif type_attack == 'one_anti_diag':
        imgs_perturbed = perturb_image_anti_diag(xs, img.clone())
    else:
        raise Exception('This attack is not yet implemented')
    model.zero_grad()
    with torch.no_grad():
        probs = fitness_function_version(fitness_function, random_value, imgs_perturbed, model, text_features,
                                         target_class_index, original_class_index)
        probs = probs.to(device)
        return probs


def attack_success(x, img, target_index: int, original_index: int, targeted: bool, model, text_features,
                   type_attack: str, w: int, h: int) -> bool:
    """
    The attack success function used for stopping condition in the DE.
    Hence checking if we have a targeted or non-targeted missclassification.
    @param x: the perturbations encodings
    @param img: the tensor which represents the preprocessed original image
    @param target_index: the index of the target class
    @param original_index: the index of the original class
    @param targeted: the boolean which is True if we launch targeted attacks, False if we launch untargeted attacks
    @param model: the model which we attack
    @param text_features: the text features
    @param type_attack:the string which represents the type of the attack ('one_patch, one_row etc.')
    @param w: the integer which represents the width
    @param h: the integer which represents the height
    @return: return True, if the attack was successful (target attack), False otherwise
    """
    with torch.no_grad():
        if type_attack == 'one_patch':
            attack_image = perturb_image_patch(x, img.clone(), w, h)
        elif type_attack == 'one_row':
            attack_image = perturb_image_row(x, img.clone())
        elif type_attack == 'one_column':
            attack_image = perturb_image_column(x, img.clone())
            # check_gpu_variable(imgs_perturbed)
        elif type_attack == 'one_diag':
            attack_image = perturb_image_diag(x, img.clone())
        elif type_attack == 'one_anti_diag':
            attack_image = perturb_image_anti_diag(x, img.clone())
        else:
            raise Exception('This attack is not yet implemented')
        # check_gpu_variable(attack_image)
        # New version
        probs = get_probs_per_model_images_multi(attack_image, model, text_features)
        predicted_index = torch.argmax(probs)
        if targeted:
            if predicted_index == target_index:
                return True
            else:
                return False
        else:
            if predicted_index != original_index:
                return True
            else:
                return False


def patch_attack_multiple(type_attack: str, model, tokenizer, images: list, image_bounds: list,
                          candidate_captions: list, original_captions: list, target_captions: list, targeted: bool,
                          config_string: str, pixels: int, popsize: int, maxiter: int, crossover: float,
                          mutation: float, fitness_function: str, strategy: str = 'best1bin', polish: bool = False,
                          max_iter_lbfgs: int = 100, save_plot: bool = True, stats: bool = False, w: int = 0,
                          h: int = 0):
    """
    Launches the Contiguous one-pixel attacks (perturbations on the preprocessed image) for multiple images.
    @param type_attack: the string which represents the type of the attack ('one_patch, one_row etc.')
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
    @param crossover: the crossover rate
    @param mutation: the mutation rate
    @param fitness_function: the string which represents the fitness function
    @param strategy: the string which represents the strategy used in the DE
    @param polish: the boolean value which is True if the LBFGS was used, False otherwise
    @param max_iter_lbfgs: the maximum number of iterations in the LBGFS
    @param save_plot: the boolean value which is True if we want to save the plots, False otherwise
    @param stats: the boolean value which is True if we want to save the stats, False otherwise
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
    list_indx_stats_fails = []
    list_stats_all = []

    random_vector = get_random_vector(fitness_function)

    for i in range(len(target_captions)):
        image_gpu = images[i].clone().to(device=device)
        target_label_index = get_index_label(target_captions[i], candidate_captions)
        original_label_index = get_index_label(original_captions[i], candidate_captions)

        adv_image_tensor, list_stats = patch_attack_single(type_attack, model, text_features_normalized, image_gpu,
                                                           image_bounds, target_label_index, original_label_index,
                                                           targeted, pixels, popsize, maxiter, crossover, mutation,
                                                           fitness_function, strategy, polish, max_iter_lbfgs, stats,
                                                           random_vector, w, h)
        # Check if they have the same values

        # adv_image_tensor_process = unpacking_pixel_values(**processor(images=adv_image_tensor, return_tensors="pt")).to(
        #     device)
        # image_gpu_process = unpacking_pixel_values(**processor(images=images[i], return_tensors="pt")).to(device)
        result_perturbation = image_gpu - adv_image_tensor

        prediction_original, probs_original = classify_image_and_probs(model, text_features_normalized,
                                                                       image_gpu, candidate_captions)
        prediction_adv, probs_adv = classify_image_and_probs(model, text_features_normalized, adv_image_tensor,
                                                             candidate_captions)
        # images[i] = images[i].detach()
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
        if stats:
            tensor_stats = torch.stack(list_stats)
            if len(list_stats) == maxiter + 1:
                list_indx_stats_fails.append(i)
            list_stats_all.append(tensor_stats)
    if stats:
        # Assuming you have a list of 100 tensors, each of size 300x100
        file_path = '../results/fits/' + type_attack + '/' + config_string + '_fit_values.pt'
        torch.save(list_stats_all, file_path)
    return list_predicted_labels_targeted, list_predicted_labels_non_targeted, list_probs_original, list_probs_target


def create_inits(inits: list, image_bounds: list, bounds: list) -> list:
    """
    Creates the initial encodings.
    @param inits: the list which represents the initial agents
    @param image_bounds: the list which represents the image bounds
    @param bounds: the list which represents the position and the color bounds
    @return: the initial encodings
    """
    for init in inits:
        for i in range(len(bounds)):
            if i == 0:
                init[i] = np.random.randint(low=image_bounds[0][0], high=image_bounds[0][1] + 1)
            elif i == 1:
                init[i] = np.random.randint(low=image_bounds[1][0], high=image_bounds[1][1] + 1)
            else:
                init[i] = np.random.normal(128, 127)
    return inits


def patch_attack_single(type_attack: str, model, text_features, image, image_bounds: list, target_caption_index: int,
                        original_caption_index: int, targeted: bool, pixels: int, popsize: int, maxiter: int,
                        crossover: float, mutation: float, fitness_function: str, strategy: str = 'best1bin',
                        polish: bool = False, max_iter_lbfgs: int = 100, stats: bool = False,
                        random_vector: list = [], w: int = 0, h: int = 0) -> tuple:
    """
    Launches the Contiguous one-pixel attacks (perturbations on the preprocessed image) for single images.
    @param type_attack: the string which represents the type of the attack ('one_patch, one_row etc.')
    @param model: the model which we attack
    @param text_features: the textual features from the captions
    @param image: the image which we attack
    @param image_bounds:the list which represents the image bounds
    @param target_caption_index: the index of the target captions
    @param original_caption_index: the index of the original captions
    @param targeted: the bool value which indicates targeted attacks (True) and non-targeted attacks (False)
    @param pixels: the number of perturbed pixels
    @param popsize: the population size
    @param maxiter: the number of max iterations
    @param crossover: the crossover rate
    @param mutation: the mutation rate
    @param fitness_function: the string which represents the fitness function
    @param strategy: the string which represents the strategy used in the DE
    @param polish: the boolean value which is True if the LBFGS was used, False otherwise
    @param max_iter_lbfgs: the maximum number of iterations in the LBGFS
    @param stats: the boolean value which is True if we want to save the stats, False otherwise
    @param random_vector: the list which represents the random vector
    @param w: the integer which represents the width
    @param h: the integer which represents the height
    @return: a tuple which represents the best perturbed image and also the stats during the attack
    """
    pixel_positions_bounds = [image_bounds[0], image_bounds[1]]
    color_positions_bounds = [image_bounds[2], image_bounds[3], image_bounds[4]]
    patch_pixels = pixels
    color_bounds = torch.tensor(color_positions_bounds * patch_pixels).to(device)
    positions_bounds = torch.tensor(pixel_positions_bounds).to(device)
    bounds = torch.cat((positions_bounds, color_bounds), dim=0)

    predict_fn = lambda xs, random_value: predict_classes_multi(
        xs, random_value, image, target_caption_index, original_caption_index, targeted, model, text_features,
        fitness_function, type_attack, w, h)
    callback_fn = lambda x: attack_success(
        x, image, target_caption_index, original_caption_index, targeted, model, text_features, type_attack, w, h)

    inits = torch.zeros((popsize, len(bounds)), dtype=torch.float32, device=device)
    inits = create_inits(inits, image_bounds, bounds)

    attack_result, list_stats = differential_evolution(predict_fn, bounds, inits, maxiter=maxiter, popsize=popsize,
                                                       crossover=crossover, callback_fn=callback_fn, mutation=mutation,
                                                       polish=polish, max_iter_lbfgs=max_iter_lbfgs, strategy=strategy,
                                                       stats=stats, random_vec=random_vector)

    if type_attack == 'one_patch':
        attack_image = perturb_image_patch(attack_result, image.clone(), w, h)
    elif type_attack == 'one_row':
        attack_image = perturb_image_row(attack_result, image.clone())
    elif type_attack == 'one_column':
        attack_image = perturb_image_column(attack_result, image.clone())
    elif type_attack == 'one_diag':
        attack_image = perturb_image_diag(attack_result, image.clone())
    elif type_attack == 'one_anti_diag':
        attack_image = perturb_image_anti_diag(attack_result, image.clone())
        # check_gpu_variable(imgs_perturbed)
    else:
        raise Exception('This type of the attack is not implemented yet!!!')
    attack_image = attack_image.to(device)

    return attack_image, list_stats


def configuration_string_plot(code: str, pretrained_model: str, dataset: str, generate_captions: bool, type_target: str,
                              targeted: bool, pixels: int, popsize: int, maxiter: int, crossover: float,
                              mutation: float, strategy: str, fitness_function: str, w: int, h: int) -> str:
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
    @param crossover: the crossover rate
    @param mutation: the mutation rate
    @param strategy: the string which represents the strategy used in the DE
    @param fitness_function: the string which represents the fitness function
    @param w: the integer which represents the width
    @param h: the integer which represents the height
    @return: the configuration string which will be used in the plots
    """
    config = f"{pretrained_model}_{dataset}_gc_{generate_captions}_tt_{type_target}_tar_{targeted}_px_{pixels}_pops_{popsize}_maxit_{maxiter}_cross_{crossover:.2f}_mut_{mutation:.2f}_strat_{strategy}_fit_{fitness_function}"
    if w != 0:
        config = config + f"_w_{w}"
    if h != 0:
        config = config + f"_h_{h}"
    config = config + "_" + code
    return config


def create_dictionary_patch(pretrained_model: str, dataset: str, generate_captions: bool, type_target: str,
                            save_plot: bool, pixels: int, popsize: int, maxiter: int, crossover: float,
                            mutation: float, fitness_function: str, strategy: str, polish: bool, targeted: bool,
                            max_iter_lbfgs: int, stats: bool, w: int, h: int) -> dict:
    """
    Creates the dictionary used in the results for the Contiguous attack v2 (attacking preprocessed images).
    @param pretrained_model: the string which represents the pretrained model
    @param dataset: the string which represents the dataset
    @param generate_captions: bool value which is True if captions where generated, False otherwise
    @param type_target: the string which represents the type of the target -> 'second'/'random'
    @param save_plot: the boolean value which is True if we want to save the plots, False otherwise
    @param pixels: the number of perturbed pixels
    @param popsize: the population size
    @param maxiter: the number of max iterations
    @param crossover: the crossover rate
    @param mutation: the mutation rate
    @param fitness_function: the string which represents the fitness function
    @param strategy: the string which represents the strategy used in the DE
    @param polish: the boolean value which is True if the LBFGS was used, False otherwise
    @param max_iter_lbfgs: the maximum number of iterations in the LBGFS
    @param w: the integer which represents the width
    @param h: the integer which represents the height
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
    dict_add['crossover'] = crossover
    dict_add['mutation'] = mutation
    dict_add['fitness_function'] = fitness_function
    dict_add['w'] = w
    dict_add['h'] = h
    dict_add['strategy'] = strategy
    dict_add['polish'] = polish
    dict_add['max_iter_lbfgs'] = max_iter_lbfgs
    dict_add['save_plot'] = save_plot
    dict_add['stats'] = stats
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
           save_plot: bool, pixels: int, popsize: int, maxiter: int, crossover: float, mutation: float,
           fitness_function: str, strategy: str = 'best1bin', polish: bool = False, max_iter_lbfgs: int = 100,
           targeted: bool = True, stats: bool = False, w:int=0, h:int=0, code:str='0') -> None:
    """
    The main attack function for the Contiguous attack (one-pixel attacking the preprocessed images using shapes).
    @param type_attack: the string which represents the type of the attack ('one_patch, one_row etc.')
    @param pretrained_model: the string which represents the model
    @param dataset: the string which represents the dataset
    @param generate_captions: bool value which is True if captions where generated, False otherwise
    @param type_target: the string which represents the type of the target -> 'second'/'random'
    @param save_plot: the boolean value which is True if we want to save the plots, False otherwise
    @param pixels: the number of perturbed pixels
    @param popsize: the population size
    @param maxiter: the number of max iterations
    @param crossover: the crossover rate
    @param mutation: the mutation rate
    @param fitness_function: the string which represents the fitness function
    @param strategy:  the string which represents the strategy used in the DE
    @param polish: the boolean value which is True if the LBFGS was used, False otherwise
    @param max_iter_lbfgs: the maximum number of iterations in the LBGFS
    @param targeted: the bool value which indicates targeted attacks (True) and non-targeted attacks (False)
    @param stats: the boolean value which is True if we want to save the stats, False otherwise
    @param w: the integer which represents the width
    @param h: the integer which represents the height
    @param code: the string which represents the code of the experiment, such that it can be easier identifiable
    """
    logging.info('Starting getting the model...')
    model, processor, tokenizer = get_model(pretrained_model)
    if tokenizer is None:
        generate_captions = False
    image_bounds = get_model_image_processor_bounds(pretrained_model)
    if type_attack == 'one_row':
        image_bounds = get_image_bounds_rectangle_shrink(image_bounds, pixels, 1)
        # get_image_bounds_rectangle_shrink(image_bounds, 1, pixels)
    elif type_attack == 'one_column':
        image_bounds = get_image_bounds_rectangle_shrink(image_bounds, 1, pixels)
    elif type_attack == 'one_patch':
        pixels = w * h
        image_bounds = get_image_bounds_rectangle_shrink(image_bounds, w, h)
    elif type_attack == 'one_diag':
        image_bounds = get_image_bounds_rectangle_shrink(image_bounds, pixels, pixels)
    elif type_attack == 'one_anti_diag':
        image_bounds = get_image_bounds_anti_diag(image_bounds, pixels)
    else:
        raise Exception('We did not implement this type of the attack')
    model.to(device)
    logging.info('Finalizing getting the model')

    logging.info('Starting getting the data...')
    x_samples, y_samples, candidate_labels = load_preprocessed_data(pretrained_model, dataset, generate_captions)
    candidate_captions = candidate_labels

    # For testing locally in the future
    # x_samples = x_samples[:2]
    # y_samples = y_samples[:2]
    if generate_captions:
        candidate_captions = create_captions_list(candidate_labels)
    logging.info('Finalizing getting the data...')
    logging.info('Starting generate the attacking samples...')
    x_samples, y_samples, probs = generate_attacking_setup(x_samples, y_samples, candidate_labels, model, tokenizer,
                                                           candidate_captions)
    # logging.info('Finalizing generate the attacking samples...')
    target_labels = generate_target_labels(type_target, probs, candidate_labels, y_samples)

    captions_original_labels = y_samples
    captions_target_labels = target_labels
    if generate_captions:
        captions_target_labels = create_captions_list(target_labels)
        captions_original_labels = create_captions_list(y_samples)
    logging.info('Create the configuration string')

    config_str = configuration_string_plot(code, pretrained_model, dataset, generate_captions, type_target, targeted,
                                           pixels, popsize, maxiter, crossover, mutation, strategy, fitness_function, w,
                                           h)

    logging.info('Launch the attack.....')
    predicted_labels_targeted, predicted_labels_non_targeted, list_probs_original, list_probs_target = patch_attack_multiple(
        type_attack, model, tokenizer, x_samples, image_bounds, candidate_captions, captions_original_labels,
        captions_target_labels, targeted, config_str, pixels, popsize, maxiter, crossover, mutation, fitness_function,
        strategy, polish, max_iter_lbfgs, save_plot, stats, w, h)

    logging.info("Done with the attack")
    logging.info("Compute the results")
    results = generate_numerical_results(predicted_labels_targeted, predicted_labels_non_targeted, list_probs_original,
                                         list_probs_target, 0.3)
    add_dictionary = create_dictionary_patch(pretrained_model, dataset, generate_captions, type_target, save_plot,
                                             pixels, popsize, maxiter, crossover, mutation, fitness_function, strategy,
                                             polish, targeted, max_iter_lbfgs, stats, w, h)

    create_results_json_v2(results, config_str, type_attack, add_dictionary, code)
    logging.info("Finalizing the experiments")

if __name__ == "__main__":
    pretrained_model = 'CLIP_ViT-B32'
    # pretrained_model = 'ALIGN'
    # pretrained_model = 'AltCLIP'
    # pretrained_model = 'GroupViT'
    # pretrained_model = 'BLIP'
    # pretrained_model = "VAN-base"
    # pretrained_model = "resnet-50-imagenet"
    dataset = "cifar10"
    # dataset = "imagenet"
    generate_captions = True
    type_target = "second"

    targeted = True
    save_plot = True
    stats = True

    pixels = 4
    popsize = 300
    maxiter = 100

    # pixels = 4
    # popsize = 10
    # maxiter = 10
    crossover = 0.8
    mutation = 0.55
    strategy = "best2exp"
    w = 2
    h = 2
    fitness_function = 'softmax_loss1'
    # fitness_function = 'logprobs_loss4'
    # fitness_function = 'logits_loss4'
    log_file = 'z_patch0_dnn'
    log_type = 'info'
    code = 'z_patch0_dnn'
    type_attack = 'one_patch'

    # pretrained_model = "StreetCLIP"
    attack(type_attack, pretrained_model, dataset, generate_captions, type_target, save_plot, pixels, popsize, maxiter,
           crossover, mutation, fitness_function, strategy='best1bin', polish=False, max_iter_lbfgs=100,
           targeted=targeted, stats=stats, w=w, h=h, code=code)
