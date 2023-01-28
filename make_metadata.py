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
        'set': 'train'
    },
    {
        'image': 'overcooked marshmallow/symbol.png',
        'attr': 'overcooked',
        'obj': 'marshmallow',
        'set': 'train'
    },
    {
        'image': 'burnt marshmallow/symbol.png',
        'attr': 'burnt',
        'obj': 'marshmallow',
        'set': 'val'
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
        'set': 'val'
    },
    {
        'image': 'burnt marshmallow/symbol.png',
        'attr': 'burnt',
        'obj': 'marshmallow',
        'set': 'test'
    },
    {
        'image': 'uncooked marshmallow/symbol.png',
        'attr': 'uncooked',
        'obj': 'marshmallow',
        'set': 'test'
    },
    {
        'image': 'overcooked marshmallow/symbol.png',
        'attr': 'overcooked',
        'obj': 'marshmallow',
        'set': 'test'
    }
]

torch.save(lst, 'metadata_compositional-split-natural.t7')
