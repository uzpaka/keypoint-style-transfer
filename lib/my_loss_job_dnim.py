import sys
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import pickle
import torchvision

import lib.ns_utils
from lib.my_d2net_utils import get_d2net, interpolate_dense_features_gpu
import skimage.measure
import skimage.transform

from lib.utils import (
    grid_positions,
    upscale_positions,
    downscale_positions,
    savefig,
    imshow_image,
    preprocess_image
)
from lib.exceptions import NoGradientError, EmptyTensorError

from lib.my_r2d2 import r2d2_extract_keypoints_and_sparse_descriptors

matplotlib.use('Agg')

kp1_initial = None
kp2_initial = None

kp1_unmatched = None
kp2_unmatched = None
original_mma = None
r2d2_model = None
r2d2_mma = None
original_matches = None
r2d2_matches = None

torch_mse_loss = torch.nn.MSELoss()

def loss_function(model, ns_transformer, ns_vgg, d2net_test_model, features_tin, gram_style, transformer_input, batch, device, margin=1, safe_radius=4, scaling_steps=3, 
                    my_global_step=0, num_steps_per_image=800, content_steps = 400, output_path = "temp_output/"):
    global kp1_initial, kp2_initial, torch_mse_loss, kp1_unmatched, kp2_unmatched, original_mma, r2d2_model, r2d2_mma, original_matches, r2d2_matches

    transformer_output = ns_transformer(transformer_input)
    transformer_output_vgg = lib.ns_utils.normalize_batch(transformer_output)
    features_tout = ns_vgg(transformer_output_vgg)
    content_loss = 1.0 * torch_mse_loss(features_tout.relu2_2, features_tin.relu2_2)

    #Prepare the transformer output for d2-net. Caffe preprocessing -> needs BGR and not RGB!    
    transformer_output = torch.clamp(transformer_output, 0.0, 255.0)
    transformer_output = transformer_output[:, [2,1,0], :,:] #RGB to BGR
    
    if my_global_step==0 or my_global_step == (num_steps_per_image-1):
        
        if my_global_step==0:
            r2d2_matches, kp1_r2d2_unmatched, kp2_r2d2_unmatched = get_r2d2_matches2(batch['image1_rgbraw'][0], batch['image2_rgbraw'][0])
            kp1_unmatched = kp1_r2d2_unmatched.astype('float')
            kp2_unmatched = kp2_r2d2_unmatched.astype('float')

            kp1_initial, kp2_initial, original_matches = keypoint_matches(batch['image1_rgbraw'][0], batch['image2_rgbraw'][0], kp1_unmatched, kp2_unmatched, d2net_test_model, device, do_fundamental=True) 
                                
        if my_global_step == (num_steps_per_image-1):
            transformer_output_cv2 = transformer_output[0].cpu().detach().numpy().transpose(1, 2, 0).astype("uint8")             
            translator_matches = get_mutual_nn(cv2.cvtColor(transformer_output_cv2, cv2.COLOR_BGR2RGB), batch['image2_rgbraw'][0], kp1_unmatched, kp2_unmatched, d2net_test_model, device)
            
            
            output_translator_path = output_path + "translator_data_{}_{}.npz".format(batch["image_name1"].replace("/","_"), batch["image_name2"].replace("/","_"))
            
            translated_imname = output_path + "translated_image_{}_{}.png".format(batch["image_name1"].replace("/","_"), batch["image_name2"].replace("/","_"))


            n_original_inliers = kp1_initial.shape[0]
            kp1_initial_cv2 = [cv2.KeyPoint(point[0], point[1], 1) for point in kp1_initial]
            kp2_initial_cv2 = [cv2.KeyPoint(point[0], point[1], 1) for point in kp2_initial]
            placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_original_inliers)]
            original_images_matched = cv2.drawMatches(batch['image1_rgbraw'][0], kp1_initial_cv2, batch['image2_rgbraw'][0], kp2_initial_cv2, placeholder_matches, None)
            original_images_matched = cv2.cvtColor(original_images_matched, cv2.COLOR_RGB2BGR)

            output_original_images_matched = output_path + "original_matches_{}_{}_ninliers_{}.png".format(batch["image_name1"].replace("/","_"), batch["image_name2"].replace("/","_"), n_original_inliers)

            
            #Display matches for the output of the transformer:
            kp1_matched = kp1_unmatched[translator_matches[:, 0], : 2]
            kp2_matched = kp2_unmatched[translator_matches[:, 1], : 2]

            np.random.seed(seed=1)
            _, inliers = skimage.measure.ransac( (kp1_matched, kp2_matched), skimage.transform.FundamentalMatrixTransform, min_samples=8, residual_threshold=4, max_trials=10000)
            kp1_ret = kp1_matched[inliers]
            kp2_ret = kp2_matched[inliers]    
            n_inliers = np.sum(inliers)
            kp1_inliers = [cv2.KeyPoint(point[0], point[1], 1) for point in kp1_matched[inliers]]
            kp2_inliers = [cv2.KeyPoint(point[0], point[1], 1) for point in kp2_matched[inliers]]
            placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
            new_images_matched = cv2.drawMatches(transformer_output_cv2, kp1_inliers, cv2.cvtColor(batch['image2_rgbraw'][0], cv2.COLOR_RGB2BGR), kp2_inliers, placeholder_matches, None)

            output_new_images_matched = output_path + "new_matches_{}_{}_ninliers_{}.png".format(batch["image_name1"].replace("/","_"), batch["image_name2"].replace("/","_"), n_inliers)

            if (kp1_unmatched is None) or (kp2_unmatched is None) or (translator_matches is None) or (original_matches is None):
                
                print("Some output variables were None! Exit...")
                
                print(kp1_unmatched)
                print(kp2_unmatched)
                print(translator_matches)
                print(original_matches)                
                sys.exit()

            else:
                
                with open(output_translator_path, 'wb') as output_file:
                    np.savez(
                        output_file,
                        kp1_r2d2 = kp1_unmatched,
                        kp2_r2d2 = kp2_unmatched,
                        translator_matches = translator_matches,
                        original_d2net_matches = original_matches,
                        )
                    
                cv2.imwrite(translated_imname, transformer_output_cv2)
                cv2.imwrite(output_new_images_matched, new_images_matched)
                cv2.imwrite(output_original_images_matched, original_images_matched)
            
    transformer_output = transformer_output - torch.from_numpy(np.array([103.939, 116.779, 123.68])).view(-1, 1, 1).cuda() # remove caffe mean

    if my_global_step < content_steps:
        return content_loss.unsqueeze(0)

    n_batch = 1
    style_loss = 0.0
    for ft_y, gm_s in zip(features_tout, gram_style):
        gm_y = lib.ns_utils.gram_matrix(ft_y)
        style_loss += torch_mse_loss(gm_y, gm_s[:n_batch, :, :])

#    output = model({
#        'image1': transformer_output.float(),
#        'image2': batch['image2'].float().to(device)
#    })
    
    output1 = model({
        'image1': transformer_output.float(),
        'image2': None
    })
    
    output2 = model({
        'image1': batch['image2'].float().to(device),
        'image2': None
    })

    loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
    has_grad = False

    n_valid_samples = 0
    for idx_in_batch in range(batch['image1'].size(0)):

        # Network output
        dense_features1 = output1['dense_features1'][idx_in_batch]
        c, h1, w1 = dense_features1.size()
        scores1 = output1['scores1'][idx_in_batch].view(-1)

        dense_features2 = output2['dense_features1'][idx_in_batch]
        _, h2, w2 = dense_features2.size()
        scores2 = output2['scores1'][idx_in_batch]

        all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
        all_descriptors2 = F.normalize(dense_features2.view(c, -1), dim=0)

        #################### BEGINNING OF SELF-SUPERVISION CHANGE
        #Here you could actually interpolate as well
        #image: [height,width,3]
        #dense_features: [512,h~,w~]
        #kp_initial: [N, 2] in original pixel coordinates and in numpy

        image_height1 = batch['image1_rgbraw'][idx_in_batch].shape[0]
        image_width1 = batch['image1_rgbraw'][idx_in_batch].shape[1]
        factor_height1 = (dense_features1.shape[1]+0.0)/image_height1
        factor_width1 = (dense_features1.shape[2]+0.0)/image_width1
        fmap_pos1 = np.copy(kp1_initial)
        fmap_pos1[:,0] = np.minimum(fmap_pos1[:,0]*factor_width1, dense_features1.shape[2]-1)
        fmap_pos1[:,1] = np.minimum(fmap_pos1[:,1]*factor_height1, dense_features1.shape[1]-1)
        fmap_pos1[:,0] = np.maximum(fmap_pos1[:,0], 0)
        fmap_pos1[:,1] = np.maximum(fmap_pos1[:,1], 0)
        fmap_pos1 = fmap_pos1[:, [1, 0]] # store as height,width instead of width,height
        fmap_pos1 = np.transpose(fmap_pos1) # [2,N] instead of [N,2]
        fmap_pos1 = torch.round(torch.from_numpy(fmap_pos1)).long().cuda()


        image_height2 = batch['image2_rgbraw'][idx_in_batch].shape[0]
        image_width2 = batch['image2_rgbraw'][idx_in_batch].shape[1]
        factor_height2 = (dense_features2.shape[1]+0.0)/image_height2
        factor_width2 = (dense_features2.shape[2]+0.0)/image_width2
        fmap_pos2 = np.copy(kp2_initial)
        fmap_pos2[:,0] = np.minimum(fmap_pos2[:,0]*factor_width2, dense_features2.shape[2]-1)
        fmap_pos2[:,1] = np.minimum(fmap_pos2[:,1]*factor_height2, dense_features2.shape[1]-1)
        fmap_pos2[:,0] = np.maximum(fmap_pos2[:,0], 0)
        fmap_pos2[:,1] = np.maximum(fmap_pos2[:,1], 0)
        fmap_pos2 = fmap_pos2[:, [1, 0]] # store as height,width instead of width,height
        fmap_pos2 = np.transpose(fmap_pos2) # [2,N] instead of [N,2]
        fmap_pos2 = torch.round(torch.from_numpy(fmap_pos2)).long().cuda()
    
        descriptors1 = F.normalize(
            dense_features1[:, fmap_pos1[0, :], fmap_pos1[1, :]],
            dim=0
        )

        descriptors2 = F.normalize(
            dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]],
            dim=0
        )

        scores1 = output1['scores1'][idx_in_batch][fmap_pos1[0, :], fmap_pos1[1, :]]

        #################### END OF SELF-SUPERVISION CHANGE

        positive_distance = 2 - 2 * (
            descriptors1.t().unsqueeze(1) @ descriptors2.t().unsqueeze(2)
        ).squeeze()

        all_fmap_pos2 = grid_positions(h2, w2, device)

        position_distance = torch.max(
            torch.abs(
                fmap_pos2.unsqueeze(2).float() -
                all_fmap_pos2.unsqueeze(1)
            ),
            dim=0
        )[0]

        is_out_of_safe_radius = position_distance > safe_radius
        distance_matrix = 2 - 2 * (descriptors1.t() @ all_descriptors2)
        negative_distance2 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
            dim=1
        )[0]

        all_fmap_pos1 = grid_positions(h1, w1, device)
        position_distance = torch.max(
            torch.abs(
                fmap_pos1.unsqueeze(2).float() -
                all_fmap_pos1.unsqueeze(1)
            ),
            dim=0
        )[0]
        is_out_of_safe_radius = position_distance > safe_radius
        distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)
        negative_distance1 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
            dim=1
        )[0]

        diff = positive_distance - torch.min(
            negative_distance1, negative_distance2
        )

        scores2 = scores2[fmap_pos2[0, :], fmap_pos2[1, :]]

        loss = loss + (
            torch.sum(scores1 * scores2 * F.relu(margin + diff)) /
            torch.sum(scores1 * scores2)
        )

        has_grad = True
        n_valid_samples += 1

    if not has_grad:
        raise NoGradientError

    loss = loss / n_valid_samples

    style_loss *= 1e5 #was 1e4. I did this in order for the style loss to be of the same scale as the other losses
    
    if my_global_step < content_steps:
        #This branch is just a placeholder, in this case the function would have returned already
        weight_matching = 0.0
        weight_content = 1.0        
        weight_style = 0.0
    else:
        weight_matching = 1.0
        #weight_matching = 0
        weight_content = 0.2  
        weight_style = 0.2
        #weight_style = 0

  #  weight_matching, weight_content, weight_style = 1.0, 1.0, 1.0 #
    #print("\nD2-net, content, style losses : {}  |  {}  |  {}".format(loss.item(),content_loss,style_loss))
    return weight_matching*loss + weight_content*content_loss + weight_style*style_loss

def interpolate_depth(pos, depth):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    h, w = depth.size()

    i = pos[0, :]
    j = pos[1, :]

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
        raise EmptyTensorError

    # Valid depth
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    ids = ids[valid_depth]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]


def uv_to_pos(uv):
    return torch.cat([uv[1, :].view(1, -1), uv[0, :].view(1, -1)], dim=0)


def warp(
        pos1,
        depth1, intrinsics1, pose1, bbox1,
        depth2, intrinsics2, pose2, bbox2
):
    device = pos1.device

    Z1, pos1, ids = interpolate_depth(pos1, depth1)

    # COLMAP convention
    u1 = pos1[1, :] + bbox1[1] + .5
    v1 = pos1[0, :] + bbox1[0] + .5

    X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
    Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

    XYZ1_hom = torch.cat([
        X1.view(1, -1),
        Y1.view(1, -1),
        Z1.view(1, -1),
        torch.ones(1, Z1.size(0), device=device)
    ], dim=0)
    XYZ2_hom = torch.chain_matmul(pose2, torch.inverse(pose1), XYZ1_hom)
    XYZ2 = XYZ2_hom[: -1, :] / XYZ2_hom[-1, :].view(1, -1)

    uv2_hom = torch.matmul(intrinsics2, XYZ2)
    uv2 = uv2_hom[: -1, :] / uv2_hom[-1, :].view(1, -1)

    u2 = uv2[0, :] - bbox2[1] - .5
    v2 = uv2[1, :] - bbox2[0] - .5
    uv2 = torch.cat([u2.view(1, -1),  v2.view(1, -1)], dim=0)

    annotated_depth, pos2, new_ids = interpolate_depth(uv_to_pos(uv2), depth2)

    ids = ids[new_ids]
    pos1 = pos1[:, new_ids]
    estimated_depth = XYZ2[2, new_ids]

    inlier_mask = torch.abs(estimated_depth - annotated_depth) < 0.05

    ids = ids[inlier_mask]
    if ids.size(0) == 0:
        raise EmptyTensorError

    pos2 = pos2[:, inlier_mask]
    pos1 = pos1[:, inlier_mask]

    return pos1, pos2, ids


def tocuda(data):
    # Convert tensor data in dictionary to cuda when it is a tensor
    for key in data.keys():
        if type(data[key]) == torch.Tensor:
            data[key] = data[key].cuda()
    return data

def interpolate_helper(image_height, image_width, dense_features, keypoints, device):
    #image: [height,width,3]
    #dense_features: [512,h~,w~]
    #keypoints: [n, 2] in original pixel coordinates and in numpy
    factor_height = (dense_features.shape[1]+0.0)/image_height
    factor_width = (dense_features.shape[2]+0.0)/image_width
    kp_scaled = np.copy(keypoints)
    kp_scaled[:,0] = np.minimum(kp_scaled[:,0]*factor_width, dense_features.shape[2]-1)
    kp_scaled[:,1] = np.minimum(kp_scaled[:,1]*factor_height, dense_features.shape[1]-1)
    kp_scaled[:,0] = np.maximum(kp_scaled[:,0], 0)
    kp_scaled[:,1] = np.maximum(kp_scaled[:,1], 0)
    sparse_descriptors = interpolate_dense_features_gpu(torch.from_numpy(kp_scaled).to(device), torch.from_numpy(dense_features).to(device))
    sparse_descriptors = sparse_descriptors.t() # transpose
    sparse_descriptors = sparse_descriptors.float()

    return sparse_descriptors # [n, 512]

def mutual_nn_matcher(descriptors1, descriptors2):
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    return matches.data.cpu().numpy()

def nn_matcher(descriptors1, descriptors2):
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()
    nn12 = torch.max(sim, dim=1)[1] #[N1]
    return nn12.cpu().numpy()

def keypoint_matches(img1_rgb, img2_rgb, kp1, kp2, d2net_test_model, device, do_fundamental=True): 
    _, _, dense_features1 = get_d2net(img1_rgb, d2net_test_model, device) #numpy
    sparse_features1 = interpolate_helper(img1_rgb.shape[0], img1_rgb.shape[1], dense_features1, kp1, device) #torch gpu

    _, _, dense_features2 = get_d2net(img2_rgb, d2net_test_model, device)
    sparse_features2 = interpolate_helper(img2_rgb.shape[0], img2_rgb.shape[1], dense_features2, kp2, device) #torch gpu

    matches = mutual_nn_matcher(sparse_features1, sparse_features2).astype(np.uint32)
    kp1_matched = kp1[matches[:, 0], : 2]
    kp2_matched = kp2[matches[:, 1], : 2]

    #Note: when matching (nearest or mutual nearest), kp1_matched.shape[0] must be equal to kp2_matched.shape[0]
    kp1_ret = None
    kp2_ret = None

    if do_fundamental:
        if kp1_matched.shape[0] < 8:
            return None, None
        try:
            np.random.seed(seed=1)
            _, inliers = skimage.measure.ransac( (kp1_matched, kp2_matched), skimage.transform.FundamentalMatrixTransform, min_samples=8, residual_threshold=4, max_trials=10000)
        except np.linalg.LinAlgError as err:
            return None, None

        kp1_ret = kp1_matched[inliers]
        kp2_ret = kp2_matched[inliers]    

        #n_inliers = np.sum(inliers)
        #kp1_inliers = [cv2.KeyPoint(point[0], point[1], 1) for point in kp1_matched[inliers]]
        #kp2_inliers = [cv2.KeyPoint(point[0], point[1], 1) for point in kp2_matched[inliers]]
        #placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
        #image_matched = cv2.drawMatches(img1_rgb, kp1_inliers, img2_rgb, kp2_inliers, placeholder_matches, None)
    else:
        #The following computes all the (mutual) matches, and not just the fundamental matrix inliers. 
        kp1_inliers = [cv2.KeyPoint(point[0], point[1], 1) for point in kp1_matched]
        kp2_inliers = [cv2.KeyPoint(point[0], point[1], 1) for point in kp2_matched] 
        #placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(kp1_matched.shape[0])]
        #n_inliers = len(kp1_matched)
        #image_matched = cv2.drawMatches(img1_rgb, kp1_inliers, img2_rgb, kp2_inliers, placeholder_matches, None)

        kp1_ret = kp1_matched
        kp2_ret = kp2_matched

    return kp1_ret, kp2_ret, matches


def get_mutual_nn(img1_rgb, img2_rgb, kp1, kp2, d2net_test_model, device):
    _, _, dense_features1 = get_d2net(img1_rgb, d2net_test_model, device) #numpy
    sparse_features1 = interpolate_helper(img1_rgb.shape[0], img1_rgb.shape[1], dense_features1, kp1, device) #torch gpu

    _, _, dense_features2 = get_d2net(img2_rgb, d2net_test_model, device)
    sparse_features2 = interpolate_helper(img2_rgb.shape[0], img2_rgb.shape[1], dense_features2, kp2, device) #torch gpu

    matches = mutual_nn_matcher(sparse_features1, sparse_features2).astype(np.uint32)
    #kp1_matched = kp1[matches[:, 0], : 2]
    #kp2_matched = kp2[matches[:, 1], : 2]
    return matches

def get_r2d2_matches2(img1_rgb, img2_rgb):
    global r2d2_model
    
    if r2d2_model is None:
        sparse_features1, kp1, r2d2_model = r2d2_extract_keypoints_and_sparse_descriptors(img_rgb_numpy=img1_rgb)
    else:
        sparse_features1, kp1, _ = r2d2_extract_keypoints_and_sparse_descriptors(img_rgb_numpy=img1_rgb, net=r2d2_model)
        
    sparse_features2, kp2, _ = r2d2_extract_keypoints_and_sparse_descriptors(img_rgb_numpy=img2_rgb, net=r2d2_model)
    
    matches = mutual_nn_matcher(sparse_features1, sparse_features2).astype(np.uint32)
    
    return matches, kp1, kp2
