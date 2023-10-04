import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def postprocess(img_ori, pred, THRESHOLD, EROSION):
    h, w = img_ori.shape[:2]
    
    mask_ori = (pred.squeeze()[:, :, 1] > THRESHOLD).astype(np.uint8)
    max_size = max(h, w)
    result_mask = cv2.resize(mask_ori, dsize=(max_size, max_size))

    if h >= w:
        diff = (max_size - w) // 2
        if diff > 0:
            result_mask = result_mask[:, diff:-diff]
    else:
        diff = (max_size - h) // 2
        if diff > 0:
            result_mask = result_mask[diff:-diff, :]
        
    result_mask = cv2.resize(result_mask, dsize=(w, h))
    
    # fill holes
    #cv2.floodFill(result_mask, mask=np.zeros((h+2, w+2), np.uint8), seedPoint=(0, 0), newVal=255)
    #result_mask = cv2.bitwise_not(result_mask)
    result_mask *= 255

    # erode image
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*EROSION + 1, 2*EROSION+1), (EROSION, EROSION))
    result_mask = cv2.erode(result_mask, element)

    # smoothen edges
    result_mask = cv2.GaussianBlur(result_mask, ksize=(9, 9), sigmaX=5, sigmaY=5)
    return result_mask