from models.csp import CSPInterface
import argparse
from PIL import Image
import numpy as np

import clip
import pandas as pd
import torch
import torch.nn as nn
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from datasets.composition_dataset import transform_image, ImageLoader


model = None

def csp_init(
    config,
    device,
    prompt_template="a photo of X X",
):

    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    allattrs = ['undercooked', 'golden', 'overcooked'] # train_dataset.attrs
    allobj = ['marshmallow'] # train_dataset.objs

    # cleaning the classes and the attributes
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]

    tokenized = torch.cat(
        [
            clip.tokenize(tok, context_length=config.context_length)
            for tok in attributes + classes
        ]
    )

    orig_token_embedding = clip_model.token_embedding(tokenized.to(device))

    soft_embedding = torch.zeros(
        (len(attributes) + len(classes), orig_token_embedding.size(-1)),
    )
    for idx, rep in enumerate(orig_token_embedding):
        eos_idx = tokenized[idx].argmax()
        soft_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

    soft_embedding = nn.Parameter(soft_embedding)
    print(attributes)

    class_token_ids = clip.tokenize(
        [prompt_template],
        context_length=config.context_length,
    )
    offset = len(attributes)

    return (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    )

def get_csp(config, device):

    (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    ) = csp_init(config, device)

    interface = CSPInterface(
        clip_model,
        config,
        offset,
        soft_embedding,
        class_token_ids,
        device,
        attr_dropout=config.attr_dropout
    )

    return interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="name of the dataset", type=str)
    parser.add_argument(
        "--lr", help="learning rate", type=float, default=1e-04
    )
    parser.add_argument(
        "--weight_decay", help="weight decay", type=float, default=1e-05
    )
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-B/32"
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=64, type=int
    )
    parser.add_argument(
        "--experiment_name",
        help="name of the experiment",
        type=str,
    )
    parser.add_argument(
        "--evaluate_only",
        help="directly evaluate on the" "dataset without any training",
        action="store_true",
    )
    parser.add_argument(
        "--context_length",
        help="sets the context length of the clip model",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--attr_dropout",
        help="add dropout to attributes",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--open_world",
        help="evaluate on open world setup",
        action="store_true",
    )
    parser.add_argument(
        "--bias",
        help="eval bias",
        type=float,
        default=1e3,
    )
    parser.add_argument(
        "--topk",
        help="eval topk",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--soft_embeddings",
        help="location for softembeddings",
        type=str,
        default="./soft_embeddings.pt",
    )

    parser.add_argument(
        "--text_encoder_batch_size",
        help="batch size of the text encoder",
        default=16,
        type=int,
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help="optional threshold"
    )
    parser.add_argument(
        '--threshold_trials',
        type=int,
        default=50,
        help="how many threshold values to try"
    )

    config = parser.parse_args()

    # set the seed value
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = get_csp(config, device)

def call_this(img):

    text_idx = torch.tensor([[0, 0], [1, 0], [2, 0]])

    
    transform = transform_image(split='test')
    b = transform(img)
    b = torch.stack([b, b])
    print(b.shape)

    encoded = model.encode_image(b)
    
    a = model(encoded, text_idx)
    choice = np.argmin(a[0].detach().numpy())
    return choice

image = Image.open('data/mowly/images/burnt marshmallow/symbol.png').convert('RGB')
print(jeffrey_call_this(image))