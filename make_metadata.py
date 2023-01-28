import torch

lst = [
    {
        'image': 'burnt marshmallow/symbol.png',
        'attr': 'burnt',
        'obj': 'marshmallow',
        'set': 'train'
    },
    {
        'image': 'uncooked marshmallow/symbol.png',
        'attr': 'uncooked',
        'obj': 'marshmallow',
        'set': 'val'
    },
    {
        'image': 'overcooked marshmallow/symbol.png',
        'attr': 'overcooked',
        'obj': 'marshmallow',
        'set': 'test'
    }
]

torch.save(lst, 'metadata_compositional-split-natural.t7')
