import ml_collections

def get_base_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.classifier = 'seg'
    config.patch_size = 16
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 3
    config.activation = 'softmax'
    return config



def get_igt_config():
    config = get_base_config()
    config.patches.grid = (16, 16)
    config.classifier = 'seg'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 3
    config.n_skip = 3
    config.activation = 'softmax'
    return config
