import numpy as np


def rand_bbox(size, lam):
    '''
    the location of the box to be cut
    '''
    W = size[2]
    H = size[3]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W) # uniform
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    '''
    change the criterion accordingly
    '''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
