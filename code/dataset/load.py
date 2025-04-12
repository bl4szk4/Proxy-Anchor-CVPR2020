def load(name, root, mode, transform=None):
    if name.lower() == 'logos':
        from datasets.LogoDataset import LogoDataset
        return LogoDataset(root, mode, transform)
    raise ValueError("Dataset {} not supported.".format(name))