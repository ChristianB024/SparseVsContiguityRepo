import json
from typing import List

import torch

from adversarial_perturbation import generate_attacking_samples_light
from captions_utils import create_captions_list
from data_preprocess import load_data

torch.manual_seed(42)

from model_dict import get_model

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def create_y_labels_json(pretrained_model: str, dataset: str, generate_captions: bool, y_samples: List[str]) -> None:
    """
    Creates the list of the correct labels and saved them in the ../data_s/PRETRAIN_MODEL/DATASET/generate_captions_BOOL/
    @param pretrained_model: the string which represents the pretrained model
    @param dataset: the string which represents the dataset
    @param generate_captions: the bool which show if we want to attack using captions or not
    @param y_samples: the list of the correct labels for each image
    """
    filename = pretrained_model + '/' + dataset + '/generate_caption_' + str(generate_captions) + '/labels'
    content = {}
    content['labels'] = y_samples
    create_json_file(content, filename)


def create_json_file(content: dict, name: str) -> None:
    """
    Creates the json file based on the dictionary and the name in the ../data_s/ directory.
    @param content: the dictionary which contains the content
    @param name: the name of the json file which will be created.It can also represent a path
    """
    file = '../data_s/' + name + '.json'
    with open(file, 'w') as json_file:
        json.dump(content, json_file, indent=4)


def generate_dataset(pretrained_model: str, dataset: str, N_samples: int, generate_captions: bool) -> None:
    """
    Generates the dataset for the attacking phase (preprocessed images) saved them in the ../data_s/PRETRAIN_MODEL/DATASET/generate_captions_BOOL/.
    @param pretrained_model: the string which represents the pretrained model
    @param dataset: the string which represents the dataset
    @param N_samples: the integer which represents the number of samples that are generated from the original dataset
    @param generate_captions: the bool which show if we want to attack using captions or not
    """
    print('Starting getting the model...')
    model, processor, tokenizer = get_model(pretrained_model)
    model.to(device)
    print('Finalizing getting the model')

    print('Starting getting the data...')
    _, _, x_samples, y_samples, candidate_labels = load_data(dataset, N_samples)
    candidate_captions = candidate_labels
    if generate_captions:
        candidate_captions = create_captions_list(candidate_labels)
    print('Finalizing getting the data...')
    print('Starting generate the attacking samples...')
    x_samples, y_samples = generate_attacking_samples_light(100, x_samples, y_samples,
                                                            candidate_labels, model, processor, tokenizer,
                                                            candidate_captions)
    create_y_labels_json(pretrained_model, dataset, generate_captions, y_samples)
    file = '../data_s/' + pretrained_model + '/' + dataset + '/generate_caption_' + str(generate_captions) + '/images'
    torch.save(x_samples, file)
    print('I am done with generating for one setup')
    # How to load the generated dataset -> it contains the model image size (1, 3, 289, 289) for ALIGN. So Be careful


if __name__ == "__main__":
    clip_model = 'CLIP_ViT-B32'
    align_model = 'ALIGN'
    van_model = 'VAN-base'
    resnet_model = 'resnet-50-imagenet'
    altclip_model = 'AltCLIP'
    group_vit_model = 'GroupViT'
    blip_model = 'BLIP'

    cifar10 = 'cifar10'
    cifar100 = 'cifar100'
    imagenet = 'imagenet'

    # generate_dataset(altclip_model, cifar10, -1, False)
    # exit(0)
    # generate_dataset(altclip_model, cifar10, -1, True)
    #
    # generate_dataset(clip_model, cifar100, -1, False)
    # generate_dataset(clip_model, cifar100, -1, True)
    #
    # generate_dataset(clip_model, imagenet, -1, False)
    # generate_dataset(clip_model, imagenet, -1, True)

    # generate_dataset(clip_model, cifar10, -1, False)
    # generate_dataset(clip_model, cifar10, -1, True)
    #
    # generate_dataset(clip_model, cifar100, -1, False)
    # generate_dataset(clip_model, cifar100, -1, True)
    #
    # generate_dataset(clip_model, imagenet, -1, False)
    # generate_dataset(clip_model, imagenet, -1, True)
    #
    # # print(#####################################################)
    #
    # generate_dataset(align_model, cifar10, -1, False)
    # generate_dataset(align_model, cifar10, -1, True)
    #
    # generate_dataset(align_model, cifar100, -1, False)
    # generate_dataset(align_model, cifar100, -1, True)
    #
    # generate_dataset(align_model, imagenet, -1, False)
    # generate_dataset(align_model, imagenet, -1, True)

    # generate_dataset(resnet_model, imagenet, -1, False)
    # generate_dataset(van_model, imagenet, -1, False)
    #
    # generate_dataset(altclip_model, cifar10, -1, False)
    # generate_dataset(altclip_model, cifar10, -1, True)
    #
    # generate_dataset(altclip_model, cifar100, -1, False)
    # generate_dataset(altclip_model, cifar100, -1, True)
    #
    # generate_dataset(altclip_model, imagenet, -1, False)
    # generate_dataset(altclip_model, imagenet, -1, True)
    #
    # # generate_dataset(blip_model, cifar10, -1, False)
    # # generate_dataset(blip_model, cifar10, -1, True)
    # #
    # # generate_dataset(blip_model, cifar100, -1, False)
    # # generate_dataset(blip_model, cifar100, -1, True)
    # #
    # # generate_dataset(blip_model, imagenet, -1, False)
    # # generate_dataset(blip_model, imagenet, -1, True)
    #
    # generate_dataset(group_vit_model, cifar10, -1, False)
    # generate_dataset(group_vit_model, cifar10, -1, True)
    #
    # generate_dataset(group_vit_model, cifar100, -1, False)
    # generate_dataset(group_vit_model, cifar100, -1, True)
    #
    # generate_dataset(group_vit_model, imagenet, -1, False)
    # generate_dataset(group_vit_model, imagenet, -1, True)
