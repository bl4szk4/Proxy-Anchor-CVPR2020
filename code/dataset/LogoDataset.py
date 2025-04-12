import os
import PIL.Image

from dataset.base import BaseDataset
class LogoDataset(BaseDataset):
    def __init__(self, root, mode, transform=None):
        super(LogoDataset, self).__init__(root, mode, transform)
        self.classes = sorted([d for d in os.listdir(os.path.join(root, mode))
                               if os.path.isdir(os.path.join(root, mode, d))])
        class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        for cls in self.classes:
            cls_folder = os.path.join(root, mode, cls)
            for file in os.listdir(cls_folder):
                if file.startswith('.'):
                    continue
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.im_paths.append(os.path.join(cls_folder, file))
                    self.ys.append(class_to_idx[cls])
                    self.I.append(len(self.I))

    def nb_classes(self):
        return len(self.classes)