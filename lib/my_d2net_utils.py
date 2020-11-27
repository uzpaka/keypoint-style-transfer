from lib.utils import preprocess_image
from lib.mypyramid2 import process_multiscale
import torch.nn.functional as F
import cv2
import numpy as np
import torch

def get_d2net(image, d2net_model, device, args_multiscale=False, args_scales=[.5, 1, 2], args_maxedge = 1600, args_maxsumedges = 2800, args_preprocessing = 'caffe', args_userelu = True):
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    resized_image = image
    if max(resized_image.shape) > args_maxedge:
        fac = args_maxedge / max(resized_image.shape)
        resized_image = cv2.resize(resized_image,None,fx=fac, fy=fac,interpolation=cv2.INTER_AREA).astype('float')

    if sum(resized_image.shape[: 2]) > args_maxsumedges:
        fac = args_maxsumedges / sum(resized_image.shape[: 2])
        resized_image = cv2.resize(resized_image,None,fx=fac, fy=fac,interpolation=cv2.INTER_AREA).astype('float')

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(
        resized_image,
        preprocessing=args_preprocessing
    )
    with torch.no_grad():
        if args_multiscale:
            keypoints, scores, descriptors, dense_features = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                d2net_model,
                scales=args_scales
            )
        else:
            keypoints, scores, descriptors, dense_features = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                d2net_model,
                scales=[1]
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]

    keypoints = keypoints[:, 0:2] # Get rid of the scale information, for the visuallocalization colmap code

    return keypoints, descriptors, dense_features


def interpolate_dense_features_gpu(pos, dense_features):
    device = pos.device

    ids = torch.arange(0, pos.size(0), device=device)

    _, h, w = dense_features.size()

    i = pos[:, 1]
    j = pos[:, 0]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    if ids.size(0) == 0:
        print("Error, empty tensor")
        sys.exit(0)

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    descriptors = (
        w_top_left * dense_features[:, i_top_left, j_top_left] +
        w_top_right * dense_features[:, i_top_right, j_top_right] +
        w_bottom_left * dense_features[:, i_bottom_left, j_bottom_left] +
        w_bottom_right * dense_features[:, i_bottom_right, j_bottom_right]
    )

    descriptors = F.normalize(descriptors, dim=0)
    return descriptors
