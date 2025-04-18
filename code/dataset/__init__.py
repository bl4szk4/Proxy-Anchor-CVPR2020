from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .import utils
from .base import BaseDataset
from .LogoDataset import LogoDataset

_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP,
    'logos': LogoDataset
}
def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)