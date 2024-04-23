# SPDX-License-Identifier: MIT
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from train_ball_model import MODEL_CONFIGURATIONS, load_data


def plot_ball_detections(model1, model2, num_images: int = 8, indices=None):
    train, val, test = load_data()
    del train
    del val
    test = test.batch(len(test)).get_single_element()
    pos_mask = test[1].numpy()[:,0] > 0
    images = test[0].numpy()[pos_mask]
    del test

    results1 = np.load('weights/' + '/'.join(str(c.value) for c in model1) + '/results_test.npz')['circles'][0,pos_mask]
    results2 = np.load('weights/' + '/'.join(str(c.value) for c in model2) + '/results_test.npz')['circles'][0,pos_mask]

    if not indices:
        indices = np.random.permutation(images.shape[0])[:num_images]
        print(indices)

    fig, axes = plt.subplots(2, num_images)
    fig.set_figwidth(20)
    fig.set_figheight(4)
    fig.tight_layout()
    for res, row_axes in zip((results1, results2), axes):
        for ax, image, circle in zip(row_axes, images[indices], res[indices]):
            ax.set_axis_off()
            ax.imshow(image, cmap='Greys_r')
            ax.add_patch(matplotlib.patches.Circle(circle[:2],circle[2], fill=False, color='red', linewidth=2))
    plt.savefig('ball_detections.pdf')



if __name__ == '__main__':
    plot_ball_detections(MODEL_CONFIGURATIONS[3], MODEL_CONFIGURATIONS[7])
