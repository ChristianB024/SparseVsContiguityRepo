from transformers import CLIPModel, AutoTokenizer, AutoProcessor, AlignModel, VanForImageClassification,\
    AutoFeatureExtractor, ResNetForImageClassification, AutoImageProcessor, AltCLIPModel, GroupViTModel, CLIPTokenizer


# Here new models, can be added, based on the needs
# The code works for the multimodals models which are on the https://huggingface.co
def get_model(name_model: str) -> tuple:
    """
    Returns the model, processor and the tokenizer, depending on the chosen model.
    Possible models and configurations (value for the name_model parameter):
    - CLIP_ViT-B/32
    - ALIGN
    - AltCLIP
    - GroupViT
    - VAN-base
    - resnet-50-imagenet
    @param name_model: the string which represents the keyword for a specific model
    @return: the tuple which represents the model, processor and the tokenizer, depending on the chosen model
    """
    if name_model == 'CLIP_ViT-B32':
        return CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"), AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    elif name_model == 'ALIGN':
        return AlignModel.from_pretrained("kakaobrain/align-base"), AutoProcessor.from_pretrained(
            "kakaobrain/align-base"), AutoTokenizer.from_pretrained("kakaobrain/align-base")
    elif name_model == 'AltCLIP':
        return AltCLIPModel.from_pretrained("BAAI/AltCLIP"),AutoProcessor.from_pretrained("BAAI/AltCLIP"),AutoProcessor.from_pretrained("BAAI/AltCLIP")
    elif name_model == 'GroupViT':
        return  GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc"), AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc"), CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")
    elif name_model == 'VAN-base':
        return VanForImageClassification.from_pretrained(
            "Visual-Attention-Network/van-base"), AutoFeatureExtractor.from_pretrained(
            "Visual-Attention-Network/van-base"), None
    elif name_model == 'resnet-50-imagenet':
        return ResNetForImageClassification.from_pretrained("microsoft/resnet-50"), AutoImageProcessor.from_pretrained(
            "microsoft/resnet-50"), None
    else:
        print("Wrong model configuration which does not exist!!!See the available models in the model_dict!")
        raise Exception("Wrong model configuration which does not exist!!!See the available models in the model_dict!")


def get_model_image_processor_bounds(name_model: str) -> list:
    """
    Returns the list which represents the image bounds as follows: [(x_min, x_max), (y_min, y_max), (r_min, r_max), (g_min, g_max), (b_min, b_max)].
    Possible models and configurations (value for the name_model parameter):
    - CLIP_ViT-B/32
    - ALIGN
    - AltCLIP
    - GroupViT
    - VAN-base
    - resnet-50-imagenet
    @param name_model: the string which represents the keyword for a specific model
    @return: the list which represents the dimension of the preprocessed image accepted by the model
    The form is the following:[(x_min, x_max), (y_min, y_max), (r_min, r_max), (g_min, g_max), (b_min, b_max)]
    """
    if name_model == 'CLIP_ViT-B32':
        return [(0, 223), (0, 223), (0, 255), (0, 255), (0, 255)]
    elif name_model == 'ALIGN':
        return [(0, 288), (0, 288), (0, 255), (0, 255), (0, 255)]
    elif name_model == 'AltCLIP':
        return [(0, 223), (0, 223), (0, 255), (0, 255), (0, 255)]
    elif name_model == 'GroupViT':
        return [(0, 223), (0, 223), (0, 255), (0, 255), (0, 255)]
    elif name_model == 'VAN-base':
        return [(0, 223), (0, 223), (0, 255), (0, 255), (0, 255)]
    elif name_model == 'resnet-50-imagenet':
        return [(0, 223), (0, 223), (0, 255), (0, 255), (0, 255)]
    else:
        print("Wrong model configuration which does not exist!!!See the available models in the model_dict!")
        raise Exception("Wrong model configuration which does not exist!!!See the available models in the model_dict!")
