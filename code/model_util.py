from typing import List
import torch
import torch.nn.functional as F
import gc

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

import logging


def unpacking_pixel_values(pixel_values=None):
    """
    Function for extracting the pixel_values, since the processor will process the image and return a tensor,
    where in the field 'pixel_values', we actually have the image.
    @param pixel_values: the values of the pixel, which are encapsulated in the field 'pixel_values'
    @return:  the values of the pixels, which can be used for image perturbation
    """
    return pixel_values


def packing_pixel_values(image) -> dict:
    """
    Function for transform an image, into a batch encoding which is required as an input by the model.
    In this manner, we can give to the model any kind of perturbed image
    @param image: the input image which we want to encapsulate
    @return: batch encoding, since this is the only way the model accept the image
    """
    batch_encoding = {'pixel_values': image}
    return batch_encoding

def create_text_features(model, tokenizer, list_captions: List[str]):
    """
    Computes the normalized text features.
    @param model: the model which we want to attack
    @param tokenizer: the text encoder (tokenizer)
    @param list_captions: the list of captions
    @return: the text features generated for all the captions
    """
    inputs_text = tokenizer(text=list_captions, padding=True, return_tensors="pt").to(device)
    text_features = model.get_text_features(**inputs_text)
    return text_features

def predict_probs_and_index_single_image_tensor_multimodal(model, image, text_features):
    """
    Generates the probabilities of each class for a single preprocessed image for MULTIMODAL models.
    @param model: the model (MULTIMODAL) which we want to attack
    @param image: the preprocessed image
    @param text_features: the text features
    @return: the probabilities of each class for a single preprocessed image for MULTIMODAL models
    """
    with torch.no_grad():
        image = image.to(device)
        image_features = model.get_image_features(pixel_values=image)
        # Code from the original CLIP:
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        # logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        logit_scale = 100.0
        logits_per_image = logit_scale * image_features @ text_features.t()

        # logits_per_text = logits_per_image.t() // For now, we do not need it, maybe later

        # if it is the same also if I use the dim 1 or -1. It is tested, but keep it 1.
        # When you specify dim=1, the softmax function will be applied along the second dimension of the tensor.
        # If your tensor has shape (batch_size, num_classes) (common in classification tasks),
        # then using dim=1 would apply the softmax operation to each row (each batch element) independently.
        # probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()
        probs = logits_per_image.softmax(dim=1).detach().to(device)
        most_similar_index = torch.argmax(probs)

        return most_similar_index, probs


def predict_probs_and_index_single_image_tensor_dnn(model, image):
    """
    Generates the probabilities of each class for a single preprocessed image for DNN models.
    @param model: the model (DNN) which we want to attack
    @param image: the preprocessed image
    @return: the probabilities of each class for a single preprocessed image for DNN models
    """
    with torch.no_grad():
        image = image.to(device)
        logits_per_image = model(pixel_values=image).logits
        probs = logits_per_image.softmax(dim=1).detach().to(device)
        most_similar_index = torch.argmax(probs)
        return most_similar_index, probs


def predict_probs_and_index_single_image_tensor(model, image, text_features):
    """
    Generates the probabilities of each class for a single preprocessed image for any model (MULTIMODAL/DNN).
    If it is a MULTIMODAL, the text_features are not None.
    If it is a DNN, then the text_features are None.
    @param model: the model which we want to attack
    @param image: the preprocessed image
    @param text_features: the text features, which are None in case of DNNs, and not if case of MULTIMODAL
    @return: the probabilities of each class for a single preprocessed image for DNN models
    """
    if text_features is None:
        return predict_probs_and_index_single_image_tensor_dnn(model, image)
    else:
        return predict_probs_and_index_single_image_tensor_multimodal(model, image, text_features)


def classify_image_and_probs_multimodal(model, text_features, image, candidate_captions: list,
                                        brute_image: bool = False,
                                        processor=None) -> tuple:
    """
    Classifies the image and also returns the probabilities tensor for multiple images (images in batches).
    This function is used for MULTIMODAL models.
    @param model: the model (MULTIMODAL) which we want to attack
    @param text_features: the text features
    @param image: the images which we want to attack
    @param candidate_captions: the list of the candidate labels
    @param brute_image: the boolean value which is True if the provided images are brute (original), False otherwise
    @param processor: the image encoder processor
    @return: the tuple which contains the list of the predicted captions, and the tensor with all the classes probabilities
    """
    logging.info('Starting the classification...')
    with torch.no_grad():

        # text_features = create_text_features(model, tokenizer, candidate_captions)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        if brute_image:
            image = processor(image=image, return_tensors="pt")
            image = image.to(device)
            image_features = model.get_image_features(**image)
        else:
            image = image.to(device)
            image_features = model.get_image_features(pixel_values=image)

        # Code from the original CLIP:
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        # logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        # Value to obtain probabilities from 0 to 100
        logit_scale = 100.0
        logits_per_image = logit_scale * image_features @ text_features.t()

        # logits_per_text = logits_per_image.t() // For now, we do not need it, maybe later

        # if it is the same also if I use the dim 1 or -1. It is tested, but keep it 1.
        # When you specify dim=1, the softmax function will be applied along the second dimension of the tensor.
        # If your tensor has shape (batch_size, num_classes) (common in classification tasks),
        # then using dim=1 would apply the softmax operation to each row (each batch element) independently.
        probs = logits_per_image.softmax(dim=1).detach()
        # most_similar_index = torch.argmax(logits_per_image)
        most_similar_index = torch.argmax(probs)
        del logits_per_image
        del image_features

        return candidate_captions[most_similar_index], probs


def classify_image_and_probs_dnn(model, image, candidate_captions, brute_image=False,
                                 processor=None):
    """
    Classifies the image and also returns the probabilities tensor for multiple images (images in batches).
    This function is used for DNN models.
    @param model: the model (DNN) which we want to attack
    @param image: the images which we want to attack
    @param candidate_captions: the list of the candidate labels
    @param brute_image: the boolean value which is True if the provided images are brute (original), False otherwise
    @param processor: the image encoder processor
    @return: the tuple which contains the list of the predicted captions, and the tensor with all the classes probabilities
    """
    logging.info('Starting the classification...')
    with torch.no_grad():
        if brute_image:
            image = processor(image=image, return_tensors="pt")
            image = image.to(device)
            logits_per_image = model(**image).logits
        else:
            image = image.to(device)
            logits_per_image = model(pixel_values=image).logits
        probs = logits_per_image.softmax(dim=1).detach()
        # most_similar_index = torch.argmax(logits_per_image)
        most_similar_index = torch.argmax(probs)
        del logits_per_image

        return candidate_captions[most_similar_index], probs


def classify_image_and_probs(model, text_features, image, candidate_captions, brute_image=False,
                             processor=None) -> tuple:
    """
    Given the model setup, images and the candidate captions, this function does the prediction and outputs the predicted labels and the probabilities.
    @param model: the model which is used for attack
    @param text_features: the processor which generates the image features. If it is None, then we are attacking a DNN model
    @param image: the list of the images which needs to be classified
    @param candidate_captions: the all possible candidate captions - the model will chose the most similar one for each image
    @param brute_image: the boolean value which is True if the provided images are brute (original), False otherwise
    @param processor: the image encoder of the model
    @return: a tuple which represents the list of predicted captions, and a numpy array which represents the list of probs,
    for each prediction the latter list can be used for the figures/graphs to show the confidence levels for each class/captions
    """
    if text_features is not None:
        return classify_image_and_probs_multimodal(model, text_features, image, candidate_captions, brute_image,
                                                   processor)
    else:
        return classify_image_and_probs_dnn(model, image, candidate_captions, brute_image, processor)


def get_logits_per_model_images_multi_multimodal(img, model, text_features):
    """
    Computes the logits for preprocessed images for MULTIMODAL models. It is used with multiple processed images on the same time.
    @param img: the preprocessed images used for classification
    @param model: the model (MULTIMODAL) which is used for attack
    @param text_features: the textual features
    @return: the logits for preprocessed images
    """
    image_features = model.get_image_features(pixel_values=img).detach()

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logit_scale = 100.0
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_image = logits_per_image.detach()
    del image_features
    gc.collect()
    return logits_per_image


def get_logits_per_model_images_multi_dnn(img, model):
    """
    Computes the logits for preprocessed images for DNN models. It is used with multiple processed images on the same time.
    @param img: the preprocessed images used for classification
    @param model: the model (DNN) which is used for attack
    @return: the logits for preprocessed images
    """
    logits_per_image = model(pixel_values=img).logits
    gc.collect()
    return logits_per_image


def get_logits_per_model_images_multi(img, model, text_features):
    """
    Computes the logits for preprocessed images for MULTIMODAL/DNN models. It is used with multiple processed images on the same time.
    If it is a MULTIMODAL, the text_features are not None.
    If it is a DNN, then the text_features are None.
    @param img: the preprocessed images used for classification
    @param model: the model which we want to attack
    @param text_features: the text features, which are None in case of DNNs, and not if case of MULTIMODAL
    @return: the logits for preprocessed images
    """
    if text_features is None:
        return get_logits_per_model_images_multi_dnn(img, model)
    else:
        return get_logits_per_model_images_multi_multimodal(img, model, text_features)



def get_probs_per_model_images_multi(img, model, text_features):
    """
    Computes the probabilities for the preprocessed images.  It is used with multiple processed images on the same time.
    @param img: the preprocessed images
    @param model: the model we want to attack
    @param text_features: the textual features
    @return: the probabilities of the preprocessed images
    """
    logits_per_image = get_logits_per_model_images_multi(img, model, text_features)
    probs = logits_per_image.softmax(dim=1).detach()
    del logits_per_image
    gc.collect()
    return probs
