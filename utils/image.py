import torch


def check_range(image: torch.Tensor, low: float, high: float):
    return torch.ge(image, low).all() and torch.le(image, high).all()


def image_float_to_uint8(image: torch.Tensor):
    """ [0, 1] -> [0, 255] """
    assert image.dtype == torch.float32
    assert check_range(image, 0, 1)
    return (image * 255).to(dtype=torch.uint8)


def image_norm_to_float(image: torch.Tensor):
    """ [-1, 1] -> [0, 1] """
    assert image.dtype == torch.float32
    assert check_range(image, -1, 1)
    return (image + 1) / 2


def image_norm_to_uint8(image: torch.Tensor):
    """ [-1, 1] -> [0, 255] """
    assert image.dtype == torch.float32
    assert check_range(image, -1, 1)
    return ((image + 1) / 2 * 255).to(dtype=torch.uint8)
