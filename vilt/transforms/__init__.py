from .pixelbert import (
    pixelbert_transform,
    pixelbert_transform_randaug,
    normal_transform,
)

_transforms = {
    "normal_transform": normal_transform,
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
}


def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]
