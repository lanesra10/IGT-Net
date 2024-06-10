import ttach as tta
from typing import Tuple
import torch
import numpy as np

def SUE_TTA(model, image_path, batch: torch.tensor, last_layer: bool) -> Tuple[np.ndarray, np.ndarray]:
    r"""Interface of Binary Segmentation Uncertainty Estimation with Test-Time Augmentations (TTA) method for 1 2D slice.
            Inputs supposed to be in range [0, data_range].
            Args:
                model: Trained model.
                batch: Tensor with shape (1, C, H, W).
                last_layer: Flag whether there is Sigmoid as a last NN layer
     """
    model.eval()
    transforms = tta.Compose(
        [
            tta.VerticalFlip(),
            # tta.HorizontalFlip(),
            # tta.Rotate90(angles=[0, 180]),
            # tta.Scale(scales=[1, 2, 4]),
            tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )
    predicted = []
    sp = []
    head = []
    i = 0
    j = 100
    # dropout_layer = nn.Dropout(p=0.2)
    for transformer in transforms:
        augmented_image = transformer.augment_image(batch)
        model_output = model(augmented_image)
        deaug_mask = transformer.deaugment_mask(model_output)
        prediction = torch.softmax(
            deaug_mask, dim=1).cpu().detach().numpy() if last_layer else deaug_mask.cpu().detach().numpy()

        sp_pred = prediction[:, 1, :, :]
        head_pred = prediction[:, 2, :, :]

        sp.extend(sp_pred)
        head.extend(head_pred)
        predicted.append(prediction)


    import matplotlib.colors as colors

    earth_yellow = (15 / 255, 219 / 255, 224 / 255)
    earth_green = (15 / 255, 246 / 255, 175 / 255)
    earth_blue = (213 / 255, 217 / 255, 29 / 255)
    earth_red = (0 / 255, 0 / 255, 220 / 255)
    camp_colors = [(0, 'black'), (0.6, earth_yellow), (0.7, earth_green), (0.8, earth_blue), (0.9, earth_red), (1, 'black')]
    camp = colors.LinearSegmentedColormap.from_list('custom_camp', camp_colors)
    substring = image_path[0].rsplit('\\', 1)[-1]
    sp_pred_np = np.exp(np.mean(np.log(np.stack(sp)), axis=0))
    sp_pred_np = 1 - sp_pred_np

    import matplotlib.pyplot as plt
    plt.imshow(sp_pred_np, cmap=camp, interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./sp_hotmap/" + substring)
    plt.clf()
    # plt.show()

    plt.imshow(sp_pred_np, cmap=camp, interpolation='nearest')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./sp_output/" + substring, dpi=69.5, bbox_inches='tight', pad_inches=0)
    plt.clf()

    head_pred_np = np.exp(np.mean(np.log(np.stack(head)), axis=0))
    head_pred_np = 1 - head_pred_np

    import matplotlib.pyplot as plt
    plt.imshow(head_pred_np, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./hotmap/" + substring)
    plt.clf()

    plt.imshow(head_pred_np, cmap=camp, interpolation='nearest')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./head_output/" + substring, dpi=69.5, bbox_inches='tight', pad_inches=0)
    plt.clf()

    return np.exp(np.mean(np.log(np.stack(predicted)), axis=0))
