import glob
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from utils import get_data


def visual(ground_truth, predictions):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    - predictions [list[dict]]: model predictions
    """
    paths = glob.glob('./data/images/*')

    # mapping to access data faster
    gtdic = {}
    for gt in ground_truth:
        gtdic[gt['filename']] = gt

    # color mapping of classes
    colormap = {2: [1, 0, 0], 1: [0, 1, 0], 4: [0, 0, 1]}

    f, ax = plt.subplots()


    filename = os.path.basename(paths[0])
    img = Image.open(paths[0])
    ax.imshow(img)

    gt_bboxes = gtdic[filename]['boxes']
    classes = gtdic[filename]['classes']
    for cl, bb in zip(classes, gt_bboxes):
            x1, y1, x2, y2 = bb
            rec = Rectangle((x1, y1), x2 - x1, y2-y1, facecolor='none',
                            edgecolor=colormap[cl],lw=2)
            ax.add_patch(rec)
    ## TODOs
    ## Add prediction boxes
    plt.show()


if __name__ == "__main__": 
    ground_truth, predictions = get_data()
    visual(ground_truth,predictions)
