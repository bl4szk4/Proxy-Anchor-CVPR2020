from __future__ import print_function
from __future__ import division

from torchvision import transforms
import PIL.Image
import torch
import losses
from tqdm import tqdm
import torch.nn.functional as F


def std_per_channel(images):
    images = torch.stack(images, dim = 0)
    return images.view(3, -1).std(dim = 1)


def mean_per_channel(images):
    images = torch.stack(images, dim = 0)
    return images.view(3, -1).mean(dim = 1)


class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im

class print_shape():
    def __call__(self, im):
        print(im.size)
        return im

class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im

class pad_shorter():
    def __call__(self, im):
        h,w = im.size[-2:]
        s = max(h, w) 
        new_im = PIL.Image.new("RGB", (s, s))
        new_im.paste(im, ((s-h)//2, (s-w)//2))
        return new_im    

    
class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul_(255)
        return tensor

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor


def make_transform(is_train = True, is_inception = False):
    # Resolution Resize List : 256, 292, 361, 512
    # Resolution Crop List: 224, 256, 324, 448
    
    resnet_sz_resize = 256
    resnet_sz_crop = 224 
    resnet_mean = [0.485, 0.456, 0.406]
    resnet_std = [0.229, 0.224, 0.225]
    resnet_transform = transforms.Compose([
        transforms.RandomResizedCrop(resnet_sz_crop) if is_train else Identity(),
        transforms.RandomHorizontalFlip() if is_train else Identity(),
        transforms.Resize(resnet_sz_resize) if not is_train else Identity(),
        transforms.CenterCrop(resnet_sz_crop) if not is_train else Identity(),
        transforms.ToTensor(),
        transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])

    inception_sz_resize = 256
    inception_sz_crop = 224
    inception_mean = [104, 117, 128]
    inception_std = [1, 1, 1]
    inception_transform = transforms.Compose(
       [
        RGBToBGR(),
        transforms.RandomResizedCrop(inception_sz_crop) if is_train else Identity(),
        transforms.RandomHorizontalFlip() if is_train else Identity(),
        transforms.Resize(inception_sz_resize) if not is_train else Identity(),
        transforms.CenterCrop(inception_sz_crop) if not is_train else Identity(),
        transforms.ToTensor(),
        ScaleIntensities([0, 1], [0, 255]),
        transforms.Normalize(mean=inception_mean, std=inception_std)
       ])
    
    return inception_transform if is_inception else resnet_transform


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    s = 0
    for t, y in zip(T, Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))

def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()

    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: images, i = 1: labels, (opcjonalnie i = 2: indices)
                if i == 0:
                    # move images to device of model
                    J = model(J.cuda())
                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state

    return [torch.stack(A[i]) for i in range(len(A))]

def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)
    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])
    return proxy_mean

def evaluate_cos(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    # get predictions by assigning nearest neighbors using cosine similarity
    K = 32
    cos_sim = F.linear(X, X)
    # Poprawka: przenosimy indeksy na CPU przed indeksowaniem T
    Y = T[cos_sim.topk(1 + K)[1].cpu()[:, 1:]]
    Y = Y.float().cpu()

    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return recall

def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader):
    nb_classes = query_dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)

    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)

    # get predictions by assigning nearest neighbors using cosine similarity
    K = 50
    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0
        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]
            thresh = torch.max(pos_sim).item()
            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1
        return match_counter / m

    recall = []
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

def evaluate_cos_SOP(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest neighbors using cosine similarity
    K = 1000
    Y = []
    xs = []
    for x in X:
        xs.append(x)
        if len(xs) >= 10000:
            xs = torch.stack(xs, dim=0)
            cos_sim = F.linear(xs, X)
            # Poprawka: przenosimy indeksy na CPU przed indeksowaniem T
            y = T[cos_sim.topk(1 + K)[1].cpu()[:, 1:]]
            Y.append(y.float().cpu())
            xs = []
    if xs:
        xs = torch.stack(xs, dim=0)
        cos_sim = F.linear(xs, X)
        y = T[cos_sim.topk(1 + K)[1].cpu()[:, 1:]]
        Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    recall = []
    for k in [1, 10, 100, 1000]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return recall