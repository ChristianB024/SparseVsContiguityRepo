import argparse
import json
import logging
import log_config
import one_pixel_attack2
import patch_attack
import random_one_attack
import random_patch_attack


def process_configuration_json_file(config: json):
    """
    Process the configuration for running the experiment
    :param config: the json file which contains the configuration
    :return: pretrained_model, size_images, dataset, N_adversarial_images, adv_perturb_on_stickers
    """
    # Establish the optional fields
    # Extract values from the configuration dictionary
    # Setting up the logs
    log_file = config.get("log_file")
    log_type = config.get("log_type")
    if log_type == 'info':
        log_config.configure_log_info(log_file)
    elif log_type == 'critical':
        log_config.configure_log_critical(log_file)
    elif log_type == 'debug':
        log_config.configure_log_debug(log_file)
    else:
        raise Exception('We did not implement this kind of logger')
    logging.critical('Setup the log - Successfully with the type {}'.format(log_type))

    # Setting up the model and the dataset settings
    pretrained_model = config.get("pretrained_model")
    dataset = config.get("dataset")
    generate_captions = config.get('generate_captions')
    logging.info("Model: {}".format(pretrained_model))
    logging.info("Dataset: {}".format(dataset))
    logging.info("generate_captions: {}".format(generate_captions))
    N_sample_random = -1

    # Check the type of the attack
    type_attack = config.get('type_attack')
    logging.info("type_attack: {}".format(type_attack))

    # Add more attacks here
    if type_attack == 'one_pixel2':
        return process_type_attack_one_pixel2(config, pretrained_model, dataset, generate_captions)
    elif (type_attack == 'random_patch' or type_attack == 'random_row' or type_attack == 'random_column'
          or type_attack == 'random_diag' or type_attack == 'random_anti_diag'):
        return process_type_attack_random_patch(config, pretrained_model, dataset, generate_captions)
    elif type_attack == 'random_one':
        return process_type_attack_random_one(config, pretrained_model, dataset, generate_captions)
    elif (type_attack == 'one_patch' or type_attack == 'one_row' or type_attack == 'one_column'
          or type_attack == 'one_diag' or type_attack == 'one_anti_diag'):
        return process_type_attack_one_patch(config, pretrained_model, dataset, generate_captions)
    else:
        logging.critical("This type_attack was not yet implemented!")
        raise Exception("This type_attack was not yet implemented!")


def process_type_attack_one_pixel2(config, pretrained_model, dataset, generate_captions):
    # Setups default values
    code = '0'
    strategy = 'best1bin'
    polish = False
    max_iter_lbfgs = 100
    stats = False
    targeted = True

    type_target = config.get('type_target')
    logging.info("type_target: {}".format(type_target))
    save_plot = config.get('save_plot')
    logging.info("save_plot: {}".format(save_plot))
    pixels = config.get('pixels')
    logging.info("pixels: {}".format(pixels))
    popsize = config.get('popsize')
    logging.info("popsize: {}".format(popsize))
    maxiter = config.get('maxiter')
    logging.info("maxiter: {}".format(maxiter))
    crossover = config.get('crossover')
    logging.info("crossover: {}".format(crossover))
    mutation = config.get('mutation')
    logging.info("mutation: {}".format(mutation))
    fitness_function = config.get('fitness_function')
    logging.info("fitness function: {}".format(fitness_function))

    # Optional parameters
    if 'strategy' in config:
        strategy = config.get('strategy')
        logging.info("strategy: {}".format(strategy))
    if 'polish' in config:
        polish = config.get('polish')
        logging.info("polish: {}".format(polish))
    if 'max_iter_lbgfs' in config:
        max_iter_lbfgs = config.get('max_iter_lbfgs')
        logging.info("max_iter_lbfgs: {}".format(max_iter_lbfgs))
    if 'code' in config:
        code = config.get('code')
        logging.info("code: {}".format(code))
    if 'stats' in config:
        stats = config.get('stats')
        logging.info("stats: {}".format(stats))
    if 'targeted' in config:
        targeted = config.get('targeted')
        logging.info("targeted: {}".format(targeted))
    one_pixel_attack2.attack(pretrained_model, dataset, generate_captions, type_target, save_plot, pixels, popsize,
                             maxiter, crossover, mutation, fitness_function, strategy, polish,
                             max_iter_lbfgs, targeted, stats, code)
def process_type_attack_random_patch(config, pretrained_model, dataset, generate_captions):
    # Setups default values
    code = '0'
    targeted = True
    w = 0
    h = 0

    type_target = config.get('type_target')
    logging.info("type_target: {}".format(type_target))
    save_plot = config.get('save_plot')
    logging.info("save_plot: {}".format(save_plot))
    pixels = config.get('pixels')
    logging.info("pixels: {}".format(pixels))
    popsize = config.get('popsize')
    logging.info("popsize: {}".format(popsize))
    maxiter = config.get('maxiter')
    logging.info("maxiter: {}".format(maxiter))
    fitness_function = config.get('fitness_function')
    logging.info("fitness function: {}".format(fitness_function))
    type_attack = config.get('type_attack')
    logging.info("type attack for the random functions: {}".format(type_attack))

    # Optional parameters
    if 'code' in config:
        code = config.get('code')
        logging.info("code: {}".format(code))
    if 'targeted' in config:
        targeted = config.get('targeted')
        logging.info("targeted: {}".format(targeted))
    if 'w' in config:
        w = config.get('w')
        logging.info("w: {}".format(w))
    if 'h' in config:
        h = config.get('h')
        logging.info("h: {}".format(h))
    random_patch_attack.attack(type_attack, pretrained_model, dataset, generate_captions, type_target, save_plot,
                               pixels,
                               popsize, maxiter, fitness_function, targeted, w, h, code)


def process_type_attack_one_patch(config, pretrained_model, dataset, generate_captions):
    # Setups default values
    targeted = True
    w = 0
    h = 0
    strategy = 'best1bin'
    polish = False
    max_iter_lbfgs = 100
    stats = False
    code = '0'

    type_target = config.get('type_target')
    logging.info("type_target: {}".format(type_target))
    save_plot = config.get('save_plot')
    logging.info("save_plot: {}".format(save_plot))
    pixels = config.get('pixels')
    logging.info("pixels: {}".format(pixels))
    popsize = config.get('popsize')
    logging.info("popsize: {}".format(popsize))
    maxiter = config.get('maxiter')
    logging.info("maxiter: {}".format(maxiter))
    crossover = config.get('crossover')
    logging.info("crossover: {}".format(crossover))
    mutation = config.get('mutation')
    logging.info("mutation: {}".format(mutation))
    fitness_function = config.get('fitness_function')
    logging.info("fitness function: {}".format(fitness_function))
    type_attack = config.get('type_attack')
    logging.info("type attack for the patch attack: {}".format(type_attack))

    # Optional parameters
    if 'strategy' in config:
        strategy = config.get('strategy')
        logging.info("strategy: {}".format(strategy))
    if 'polish' in config:
        polish = config.get('polish')
        logging.info("polish: {}".format(polish))
    if 'max_iter_lbgfs' in config:
        max_iter_lbfgs = config.get('max_iter_lbfgs')
        logging.info("max_iter_lbfgs: {}".format(max_iter_lbfgs))
    if 'w' in config:
        w = config.get('w')
        logging.info("w: {}".format(w))
    if 'h' in config:
        h = config.get('h')
        logging.info("h: {}".format(h))
    if 'targeted' in config:
        targeted = config.get('targeted')
        logging.info("targeted: {}".format(targeted))
    if 'stats' in config:
        stats = config.get('stats')
        logging.info("stats: {}".format(stats))
    if 'code' in config:
        code = config.get('code')
        logging.info(code)

    patch_attack.attack(type_attack, pretrained_model, dataset, generate_captions, type_target, save_plot, pixels,
                        popsize, maxiter, crossover, mutation, fitness_function, strategy, polish, max_iter_lbfgs, targeted, stats, w, h, code)


def process_type_attack_random_one(config, pretrained_model, dataset, generate_captions):
    # Setups default values
    code = '0'
    targeted = True

    type_target = config.get('type_target')
    logging.info("type_target: {}".format(type_target))
    save_plot = config.get('save_plot')
    logging.info("save_plot: {}".format(save_plot))
    pixels = config.get('pixels')
    logging.info("pixels: {}".format(pixels))
    popsize = config.get('popsize')
    logging.info("popsize: {}".format(popsize))
    maxiter = config.get('maxiter')
    logging.info("maxiter: {}".format(maxiter))
    fitness_function = config.get('fitness_function')
    logging.info("fitness function: {}".format(fitness_function))

    # Optional parameters
    if 'code' in config:
        code = config.get('code')
        logging.info("code: {}".format(code))
    if 'targeted' in config:
        targeted = config.get('targeted')
        logging.info("targeted: {}".format(targeted))
    random_one_attack.attack(pretrained_model, dataset, generate_captions, type_target, save_plot, pixels, popsize,
                             maxiter, fitness_function, targeted, code)


def main():
    """
    All the json files dictionaries which represents the setup for the attack, should be added in the directory configurations.
    """
    parser = argparse.ArgumentParser(description="Read and process a JSON configuration file.")
    parser.add_argument("config_file", help="Path to the JSON configuration file")

    args = parser.parse_args()
    config_file = '../configurations/' + args.config_file + '.json'
    try:
        with open(config_file, "r") as json_file:
            config = json.load(json_file)
            process_configuration_json_file(config)
    except FileNotFoundError:
        print("JSON configuration file not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON in the configuration file.")
    except Exception as e:
        print("An error occurred:", str(e))