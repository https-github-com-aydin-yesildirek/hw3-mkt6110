import json

from utils import calculate_iou


def nms(predictions):
    """
    non max suppression
    args:
    - predictions [dict]: predictions dict 
    returns:
    - filtered [list]: filtered bboxes and scores
    """

    data = []
    for bb, sc in zip(predictions['boxes'], predictions['scores']):
        data.append([bb, sc])

    data_sorted = sorted(data, key = lambda k: k[1])[::-1]
    filtered = []
    for i, bi in enumerate(data_sorted):
        discard = False
        for j, bj in enumerate(data_sorted):
            ## TODOs
            # add NMS to keep bi or just discard the bbox
            
            filtered.append(bi)
    return filtered


if __name__ == '__main__':
    with open('./data/predictions_nms.json', 'r') as f:
        predictions = json.load(f)
    
    filtered = nms(predictions)
print(filtered)