import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
import random
import logging
from skimage import measure
import math


def attention_crop_drop(attention_maps, input_image):
    """
    # https://github.com/wvinzh/WS_DAN_PyTorch/blob/master/utils/attention.py
    """
    # start = time.time()
    B, N, W, H = input_image.shape
    input_tensor = input_image
    batch_size, num_parts, height, width = attention_maps.shape
    attention_maps = torch.nn.functional.interpolate(attention_maps.detach(), size=(W, H), mode='bilinear')
    part_weights = F.avg_pool2d(attention_maps.detach(), (W, H)).reshape(batch_size,-1)
    part_weights = torch.add(torch.sqrt(part_weights), 1e-12)
    part_weights = torch.div(part_weights, torch.sum(part_weights,dim=1).unsqueeze(1)).cpu()
    part_weights = part_weights.numpy()
    # print(part_weights.shape)
    ret_imgs = []
    masks = []
    # print(part_weights[3])
    for i in range(batch_size):
        attention_map = attention_maps[i]
        part_weight = part_weights[i]
        selected_index = np.random.choice(np.arange(0, num_parts), 1, p=part_weight)[0]
        selected_index2 = np.random.choice(np.arange(0, num_parts), 1, p=part_weight)[0]
        ## create crop imgs
        mask = attention_map[selected_index, :, :]
        # mask = (mask-mask.min())/(mask.max()-mask.min())
        threshold = torch.tensor(np.random.uniform(0.4, 0.6), device=mask.device)
        # threshold = 0.5
        itemindex = torch.where(mask >= mask.max()*threshold)
        # print(itemindex.shape)
        # itemindex = torch.nonzero(mask >= threshold*mask.max())
        padding_h = int(0.1*H)
        padding_w = int(0.1*W)
        height_min = itemindex[0].min()
        height_min = max(0,height_min-padding_h)
        height_max = itemindex[0].max() + padding_h
        width_min = itemindex[1].min()
        width_min = max(0,width_min-padding_w)
        width_max = itemindex[1].max() + padding_w
        # print('numpy',height_min,height_max,width_min,width_max)
        out_img = input_tensor[i][:,height_min:height_max,width_min:width_max].unsqueeze(0)
        out_img = torch.nn.functional.interpolate(out_img,size=(W,H),mode='bilinear',align_corners=True)
        out_img = out_img.squeeze(0)
        ret_imgs.append(out_img)

        ## create drop imgs
        # mask2 = attention_map[selected_index2:selected_index2 + 1, :, :]
        # threshold = random.uniform(0.2, 0.5)
        # mask2 = (mask2 < threshold * mask2.max()).float()
        # masks.append(mask2)
    # bboxes = np.asarray(bboxes, np.float32)
    crop_imgs = torch.stack(ret_imgs)
    # masks = torch.stack(masks)
    # drop_imgs = input_tensor*masks
    return crop_imgs  # (crop_imgs,drop_imgs)


class APCNNCrop(nn.Module):
    """
    https://github.com/PRIS-CV/AP-CNN_Pytorch-master
    """
    def __init__(self, input_img_H, input_img_W):
        super(APCNNCrop, self). __init__()
        self.input_img_H = input_img_H
        self.input_img_W = input_img_W
    
    def get_att_roi(self, att_mask, feature_stride, anchor_size, img_h, img_w, iou_thred=0.2, topk=1):
        """generation multi-leve ROIs upon spatial attention masks with NMS method"""
        with torch.no_grad():
            roi_ret_nms = []
            n, c, h, w = att_mask.size()
            att_corner_unmask = torch.zeros_like(att_mask).cuda()
            att_corner_unmask[:, :, int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)] = 1
            att_mask = att_mask * att_corner_unmask
            feat_anchor = self.generate_anchors_single_pyramid([anchor_size], [1], [h, w], feature_stride, 1)
            feat_new_cls = att_mask.clone()
            for i in range(n):
                boxes = feat_anchor.clone().float()
                scores = feat_new_cls[i].view(-1)
                score_thred_index = scores > scores.mean()
                boxes = boxes[score_thred_index, :]
                scores = scores[score_thred_index]
                # nms
                # nms_index = pth_nms(torch.cat([boxes, scores.unsqueeze(1)], dim=1), iou_thred)[:topk]
                nms_index = nms(boxes, scores, iou_thred)[:topk]
                boxes_nms = boxes[nms_index, :]
                if len(boxes_nms.size()) == 1:
                    boxes_nms = boxes_nms.unsqueeze(0)
                # boxes_nms = pth_nms_merge(torch.cat([boxes, scores.unsqueeze(1)], dim=1), iou_thred, topk).cuda()
                boxes_nms[:, 0] = torch.clamp(boxes_nms[:, 0], min=0)
                boxes_nms[:, 1] = torch.clamp(boxes_nms[:, 1], min=0)
                boxes_nms[:, 2] = torch.clamp(boxes_nms[:, 2], max=img_w - 1)
                boxes_nms[:, 3] = torch.clamp(boxes_nms[:, 3], max=img_h - 1)
                roi_ret_nms.append(torch.cat([torch.FloatTensor([i] * boxes_nms.size(0)).unsqueeze(1).cuda(), boxes_nms], 1))

            return torch.cat(roi_ret_nms, 0)

    def generate_anchors_single_pyramid(self, scales, ratios, shape, feature_stride, anchor_stride):
        """
        scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
        shape: [height, width] spatial shape of the feature map over which
                to generate anchors.
        feature_stride: Stride of the feature map relative to the image in pixels.
        anchor_stride: Stride of anchors on the feature map. For example, if the
            value is 2 then generate anchors for every other feature map pixel.
        """
        # Get all combinations of scales and ratios
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()

        # Enumerate heights and widths from scales and ratios
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)

        # Enumerate shifts in feature space
        shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        box_centers = np.stack(
            [box_centers_x, box_centers_y], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_widths, box_heights], axis=2).reshape([-1, 2])

        # Convert to corner coordinates (x1, y1, x2, y2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
        return torch.from_numpy(boxes).cuda()

    def get_roi_crop_feat(self, x, roi_list, scale):
        """ROI guided refinement: ROI guided Zoom-in & ROI guided Dropblock"""
        n, c, x2_h, x2_w = x.size()
        roi_3, roi_4, roi_5 = roi_list
        roi_all = torch.cat([roi_3, roi_4, roi_5], 0)
        x2_ret = []
        crop_info_all = []
        if self.training:
            for i in range(n):
                roi_all_i = roi_all[roi_all[:, 0] == i] / scale
                xx1_resize, yy1_resize, = torch.min(roi_all_i[:, 1:3], 0)[0]
                xx2_resize, yy2_resize = torch.max(roi_all_i[:, 3:5], 0)[0]
                roi_3_i = roi_3[roi_3[:, 0] == i] / scale
                roi_4_i = roi_4[roi_4[:, 0] == i] / scale
                # alway drop the roi with highest score
                mask_un = torch.ones(c, x2_h, x2_w).cuda()
                pro_rand = random.random()
                if pro_rand < 0.3:
                    ind_rand = random.randint(0, roi_3_i.size(0) - 1)
                    xx1_drop, yy1_drop = roi_3_i[ind_rand, 1:3]
                    xx2_drop, yy2_drop = roi_3_i[ind_rand, 3:5]
                    mask_un[:, yy1_drop.long():yy2_drop.long(), xx1_drop.long():xx2_drop.long()] = 0
                elif pro_rand < 0.6:
                    ind_rand = random.randint(0, roi_4_i.size(0) - 1)
                    xx1_drop, yy1_drop = roi_4_i[ind_rand, 1:3]
                    xx2_drop, yy2_drop = roi_4_i[ind_rand, 3:5]
                    mask_un[:, yy1_drop.long():yy2_drop.long(), xx1_drop.long():xx2_drop.long()] = 0
                x2_drop = x[i] * mask_un
                x2_crop = x2_drop[:, yy1_resize.long():yy2_resize.long(),
                            xx1_resize.long():xx2_resize.long()].contiguous().unsqueeze(0)
                # normalize
                scale_rate = c*(yy2_resize-yy1_resize)*(xx2_resize-xx1_resize) / torch.sum(mask_un[:, yy1_resize.long():yy2_resize.long(),
                            xx1_resize.long():xx2_resize.long()])
                x2_crop = x2_crop * scale_rate  

                x2_crop_resize = F.interpolate(x2_crop, (x2_h, x2_w), mode='bilinear', align_corners=False)
                x2_ret.append(x2_crop_resize)

                crop_info = [xx1_resize, xx2_resize, yy1_resize, yy2_resize]
                crop_info_all.append(crop_info)
        else:
            for i in range(n):
                roi_all_i = roi_all[roi_all[:, 0] == i] / scale
                xx1_resize, yy1_resize, = torch.min(roi_all_i[:, 1:3], 0)[0]
                xx2_resize, yy2_resize = torch.max(roi_all_i[:, 3:5], 0)[0]
                x2_crop = x[i, :, yy1_resize.long():yy2_resize.long(),
                            xx1_resize.long():xx2_resize.long()].contiguous().unsqueeze(0)
                x2_crop_resize = F.interpolate(x2_crop, (x2_h, x2_w), mode='bilinear', align_corners=False)
                x2_ret.append(x2_crop_resize)

                crop_info = [xx1_resize, xx2_resize, yy1_resize, yy2_resize]
                crop_info_all.append(crop_info)
        return torch.cat(x2_ret, 0), crop_info_all

    def generate_spatial_mask(self, x3, x4, x5):
        # Can think of a way to generate spatial mask
        x3 = x3.mean(1, keepdim=True)
        x4 = x4.mean(1, keepdim=True)
        x5 = x5.mean(1, keepdim=True)
        
        return x3, x4, x5

    def forward(self, x2, x3, x4, x5):
        x3_mask, x4_mask, x5_mask = self.generate_spatial_mask(x3, x4, x5)

        roi_3 = self.get_att_roi(x3_mask, 2 ** 3, 64, self.input_img_H, self.input_img_W, iou_thred=0.05, topk=5)
        roi_4 = self.get_att_roi(x4_mask, 2 ** 4, 128, self.input_img_H, self.input_img_W, iou_thred=0.05, topk=3)
        roi_5 = self.get_att_roi(x5_mask, 2 ** 5, 256, self.input_img_H, self.input_img_W, iou_thred=0.05, topk=1)
        roi_list = [roi_3, roi_4, roi_5]

        x2_crop_resize, _ = self.get_roi_crop_feat(x2, roi_list, 2 ** 3)

        return roi_list, x2_crop_resize


class APCNNCropMine(nn.Module):
    """
    https://github.com/PRIS-CV/AP-CNN_Pytorch-master
    """
    def __init__(self, input_img_H, input_img_W):
        super(APCNNCropMine, self). __init__()
        self.input_img_H = input_img_H
        self.input_img_W = input_img_W
    
    def get_att_roi(
        self, att_mask, feature_stride, anchor_size, img_h, img_w,
        iou_thred=0.2, topk=1, feature_map_level=None):
        """generation multi-leve ROIs upon spatial attention masks with NMS method"""
        with torch.no_grad():
            roi_ret_nms = []
            n, c, h, w = att_mask.size()
            att_corner_unmask = torch.zeros_like(att_mask, device=att_mask.device)
            att_corner_unmask[:, :, int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)] = 1
            att_mask = att_mask * att_corner_unmask
            feat_anchor = self.generate_anchors_single_pyramid([anchor_size], [1], [h, w], feature_stride, 1)
            feat_anchor = feat_anchor.to(att_mask.device)
            feat_new_cls = att_mask.clone()
            for i in range(n):
                boxes = feat_anchor.clone().float()
                scores = feat_new_cls[i].view(-1)
                score_thred_index = scores > scores.mean()
                boxes = boxes[score_thred_index, :]
                scores = scores[score_thred_index]
                
                nms_index = nms(boxes, scores, iou_thred)[:topk]
                boxes_nms = boxes[nms_index, :]
                scores = scores[nms_index]
                
                if len(boxes_nms.size()) == 1:
                    boxes_nms = boxes_nms.unsqueeze(0)
                boxes_nms[:, 0] = torch.clamp(boxes_nms[:, 0], min=0)
                boxes_nms[:, 1] = torch.clamp(boxes_nms[:, 1], min=0)
                boxes_nms[:, 2] = torch.clamp(boxes_nms[:, 2], max=img_w - 1)
                boxes_nms[:, 3] = torch.clamp(boxes_nms[:, 3], max=img_h - 1)
                roi_ret_nms.append(torch.cat([
                    torch.FloatTensor([i] * boxes_nms.size(0), device=att_mask.device).unsqueeze(1),
                    boxes_nms, scores.unsqueeze(1),
                    torch.tensor([feature_map_level] * boxes_nms.size(0), device=att_mask.device).unsqueeze(1)
                    ], 1))

            return torch.cat(roi_ret_nms, 0)

    def generate_anchors_single_pyramid(self, scales, ratios, shape, feature_stride, anchor_stride):
        """
        scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
        shape: [height, width] spatial shape of the feature map over which
                to generate anchors.
        feature_stride: Stride of the feature map relative to the image in pixels.
        anchor_stride: Stride of anchors on the feature map. For example, if the
            value is 2 then generate anchors for every other feature map pixel.
        """
        # Get all combinations of scales and ratios
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()

        # Enumerate heights and widths from scales and ratios
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)

        # Enumerate shifts in feature space
        shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        box_centers = np.stack(
            [box_centers_x, box_centers_y], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_widths, box_heights], axis=2).reshape([-1, 2])

        # Convert to corner coordinates (x1, y1, x2, y2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
        return torch.from_numpy(boxes)

    def get_roi_crop_feat(self, x, roi_list, scale):
        """ROI guided refinement: ROI guided Zoom-in & ROI guided Dropblock"""
        n, c, x2_h, x2_w = x.size()
        roi_3, roi_4, roi_5 = roi_list
        roi_all = torch.cat([roi_3, roi_4, roi_5], 0)
        x2_ret = []
        crop_info_all = []
        if self.training:
            for i in range(n):
                roi_all_i = roi_all[roi_all[:, 0] == i] / scale
                xx1_resize, yy1_resize, = torch.min(roi_all_i[:, 1:3], 0)[0]
                xx2_resize, yy2_resize = torch.max(roi_all_i[:, 3:5], 0)[0]
                roi_3_i = roi_3[roi_3[:, 0] == i] / scale
                roi_4_i = roi_4[roi_4[:, 0] == i] / scale
                # alway drop the roi with highest score
                mask_un = torch.ones(c, x2_h, x2_w, device=x.device)
                pro_rand = random.random()
                if pro_rand < 0.3:
                    ind_rand = random.randint(0, roi_3_i.size(0) - 1)
                    xx1_drop, yy1_drop = roi_3_i[ind_rand, 1:3]
                    xx2_drop, yy2_drop = roi_3_i[ind_rand, 3:5]
                    mask_un[:, yy1_drop.long():yy2_drop.long(), xx1_drop.long():xx2_drop.long()] = 0
                elif pro_rand < 0.6:
                    ind_rand = random.randint(0, roi_4_i.size(0) - 1)
                    xx1_drop, yy1_drop = roi_4_i[ind_rand, 1:3]
                    xx2_drop, yy2_drop = roi_4_i[ind_rand, 3:5]
                    mask_un[:, yy1_drop.long():yy2_drop.long(), xx1_drop.long():xx2_drop.long()] = 0
                x2_drop = x[i] * mask_un
                x2_crop = x2_drop[:, yy1_resize.long():yy2_resize.long(),
                            xx1_resize.long():xx2_resize.long()].contiguous().unsqueeze(0)
                # normalize
                scale_rate = c*(yy2_resize-yy1_resize)*(xx2_resize-xx1_resize) / torch.sum(mask_un[:, yy1_resize.long():yy2_resize.long(),
                            xx1_resize.long():xx2_resize.long()])
                x2_crop = x2_crop * scale_rate  

                x2_crop_resize = F.interpolate(x2_crop, (x2_h, x2_w), mode='bilinear', align_corners=False)
                x2_ret.append(x2_crop_resize)

                crop_info = [xx1_resize, xx2_resize, yy1_resize, yy2_resize]
                crop_info_all.append(crop_info)
        else:
            for i in range(n):
                roi_all_i = roi_all[roi_all[:, 0] == i] / scale
                xx1_resize, yy1_resize, = torch.min(roi_all_i[:, 1:3], 0)[0]
                xx2_resize, yy2_resize = torch.max(roi_all_i[:, 3:5], 0)[0]
                x2_crop = x[i, :, yy1_resize.long():yy2_resize.long(),
                            xx1_resize.long():xx2_resize.long()].contiguous().unsqueeze(0)
                x2_crop_resize = F.interpolate(x2_crop, (x2_h, x2_w), mode='bilinear', align_corners=False)
                x2_ret.append(x2_crop_resize)

                crop_info = [xx1_resize, xx2_resize, yy1_resize, yy2_resize]
                crop_info_all.append(crop_info)
        return torch.cat(x2_ret, 0), crop_info_all

    def generate_spatial_mask(self, x3, x4, x5):
        # Can think of a way to generate spatial mask
        x3 = x3.mean(1, keepdim=True)
        x4 = x4.mean(1, keepdim=True)
        x5 = x5.mean(1, keepdim=True)
        
        return x3, x4, x5

    def generate_crop(self, roi, x, scale, crop_count, min_size=2):
        roi = torch.cat(roi, 0)
        roi[:, 1:5] = roi[:, 1:5] / scale
        crop_all = []
        for i in range(len(x)):
            roi_tmp = roi[roi[:, 0] == i]
            
            # Use nms to get top bbox
            # index = nms(roi_tmp[:, 1:5], roi_tmp[:, -2], iou_threshold=0.05)[:crop_count]
            index = nms(roi_tmp[:, 1:5], roi_tmp[:, -2], iou_threshold=0.05)
            if len(index) != crop_count:
                logging.error("RoI not enough...")
            roi_tmp = roi_tmp[index]
            
            # Use scores to get bbox
            # roi_tmp = roi_tmp[roi_tmp[:, -2].argsort()[-crop_count:]]
            
            crop_img = []
            crop_count_ = 0
            for roi_tmp_ in roi_tmp:
                img_index, x1, y1, x2, y2, scores, x_lvl = roi_tmp_
                x1, y1 = math.floor(x1.item()), math.floor(y1.item()) 
                x2, y2 = math.ceil(x2.item()), math.ceil(y2.item())
                if (x2 - x1) >= min_size or (y2 - y1) >= min_size:
                    crop_tmp = x[i:i+1, y1:y2, x1:, x2:]
                
                    # Resize crop
                    # crop_tmp = F.interpolate(crop_tmp, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

                    # Get emb
                    crop_tmp = nn.AdaptiveAvgPool2d(1)(crop_tmp).squeeze()

                    crop_img.append(crop_tmp)
                    crop_count_ += 1

                if crop_count_ == crop_count:
                    break

            crop_all.append(torch.stack(crop_img, 0))
        
        # crop_all = torch.stack(crop_all, 0)
        
        return crop_all

    def forward(self, x1, x2, x3, x4, x5):
        x3_mask, x4_mask, x5_mask = self.generate_spatial_mask(x3, x4, x5)
        
        # changed the anchor_size if input resolution changes
        roi_3 = self.get_att_roi(x3_mask, 2 ** 3, 64//2, self.input_img_H, self.input_img_W, iou_thred=0.05, topk=5, feature_map_level=3)
        roi_4 = self.get_att_roi(x4_mask, 2 ** 4, 128//2, self.input_img_H, self.input_img_W, iou_thred=0.05, topk=5, feature_map_level=4)
        roi_5 = self.get_att_roi(x5_mask, 2 ** 5, 256//2, self.input_img_H, self.input_img_W, iou_thred=0.05, topk=5, feature_map_level=5)
        roi_list = [roi_3, roi_4, roi_5]

        x2_crop_resize, _ = self.get_roi_crop_feat(x2, roi_list, 2 ** 3)

        # can crop from original image or feature maps
        # crop = self.generate_crop(roi_list, x3, scale=2**3, crop_count=4, min_size=8)
        # crop = self.generate_crop(roi_list, x4, scale=2**4, crop_count=4, min_size=4)
        crop = self.generate_crop(roi_list, x5, scale=2**5, crop_count=4, min_size=2)
        
        return x2_crop_resize, crop


def getROIS(resolution=33, gridSize=3, minSize=1):
    """
    https://github.com/ArdhenduBehera/cap/blob/main/train_CAP.py
    """
    coordsList = []
    step = resolution / gridSize # width/height of one grid square
	
	#go through all combinations of coordinates
    for column1 in range(0, gridSize + 1):
        for column2 in range(0, gridSize + 1):
            for row1 in range(0, gridSize + 1):
                for row2 in range(0, gridSize + 1):
					
					#get coordinates using grid layout
                    x0 = int(column1 * step)
                    x1 = int(column2 * step)
                    y0 = int(row1 * step)
                    y1 = int(row2 * step)
                    
                    if x1 > x0 and y1 > y0 and ((x1 - x0) >= (step * minSize) or (y1 - y0) >= (step * minSize)): #ensure ROI is valid size
                        if not (x0==y0==0 and x1==y1==resolution): #ignore full image
                            coordsList.append([x0, y0, x1, y1]) #add bounding box to list
    coordsArray = np.array(coordsList)	 #format coordinates as numpy array						
    
    return coordsArray


def RoiPooling(x, rois_coord, pool_size=None):
    if pool_size is None:
        pool = nn.AdaptiveAvgPool2d(1)
    rois = []
    for roi_coord in rois_coord:
        x0, y0, x1, y1 = roi_coord
        
        roi = x[:, :, y0:y1, x0:x1]
        if pool_size is None:
            roi = pool(roi)
        else:
            roi = F.interpolate(roi, pool_size, mode='bilinear', align_corners=False)
        rois.append(roi)
    rois = torch.stack(rois, 1)
    
    return rois    


def AOLM(x, fms, fm1=None):
    # https://github.com/ZF4444/MMAL-Net/blob/master/utils/AOLM.py
    A = torch.sum(fms, dim=1, keepdim=True)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    M = (A > a).float()

    if fm1 is not None:
        A1 = torch.sum(fm1, dim=1, keepdim=True)
        a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
        M1 = (A1 > a1).float()


    coordinates = []
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(14, 14)
        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas))

        if fm1 is None:
            intersection = (component_labels==(max_idx+1))
        else:
            intersection = ((component_labels==(max_idx+1)).astype(int) + (M1[i][0].cpu().numpy()==1).astype(int)) ==2
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, 14//2, 14//2]
            print('there is one img no intersection')
        else:
            bbox = prop[0].bbox

        x_lefttop = bbox[0] * 32 - 1
        y_lefttop = bbox[1] * 32 - 1
        x_rightlow = bbox[2] * 32 - 1
        y_rightlow = bbox[3] * 32 - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)

    new_imgs = torch.zeros([x.size(0), 3, 448, 448]).to(x.device)  # [N, 3, 448, 448]
    for i in range(x.size(0)):
        [x0, y0, x1, y1] = coordinates[i]
        new_imgs[i:i + 1] = F.interpolate(
            x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(448, 448),
            mode='bilinear', align_corners=True)  # [N, 3, 224, 224]
    return new_imgs


def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    """
    https://github.com/GuYuc/WS-DAN.PyTorch
    """
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


def gradcam_crop(input_tensor, mask_all):
    B, C, H, W = input_tensor.size()
    crop_imgs = []
    for i in range(B):
        ## create crop imgs
        mask = mask_all[i]
        # mask = (mask-mask.min())/(mask.max()-mask.min())
        threshold = torch.tensor(np.random.uniform(0.4, 0.6), device=mask.device)
        # threshold = 0.5
        itemindex = torch.where(mask >= mask.max()*threshold)
        padding_h = int(0.1*H)
        padding_w = int(0.1*W)
        height_min = itemindex[0].min()
        height_min = max(0,height_min-padding_h)
        height_max = itemindex[0].max() + padding_h
        width_min = itemindex[1].min()
        width_min = max(0,width_min-padding_w)
        width_max = itemindex[1].max() + padding_w
        out_img = input_tensor[i][:,height_min:height_max,width_min:width_max].unsqueeze(0)
        out_img = torch.nn.functional.interpolate(out_img,size=(W,H),mode='bilinear',align_corners=True)
        out_img = out_img.squeeze(0)
        crop_imgs.append(out_img)
    crop_imgs = torch.stack(crop_imgs)
    return crop_imgs


class FocalModulation(nn.Module):
    """
    https://github.com/microsoft/FocalNet/blob/633d9e4b80b4d329fe06770277fc385fd6d4a45c/classification/focalnet.py#L38
    """
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0., use_postln_in_modulation=False, normalize_modulator=False):
        """
        Referring below, focal_level and focal_window are normally set as 3
        https://github.com/microsoft/FocalNet/tree/633d9e4b80b4d329fe06770277fc385fd6d4a45c/configs
        """
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2*dim + (self.focal_level+1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()
                
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    # Can set dilation rate
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, 
                    groups=dim, dilation=1, padding=kernel_size//2, bias=False,),
                    nn.GELU(),
                    )
                )              
            self.kernel_sizes.append(kernel_size)          
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1]

        # pre linear projection
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level+1), 1)
        
        # context aggreation
        ctx_all = 0 
        for l in range(self.focal_level):         
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*self.gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global*self.gates[:,self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level+1)

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q*self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
        
        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0

        flops += N * self.dim * (self.dim * 2 + (self.focal_level+1))

        # focal convolution
        for k in range(self.focal_level):
            flops += N * (self.kernel_sizes[k]**2+1) * self.dim

        # global gating
        flops += N * 1 * self.dim 

        #  self.linear
        flops += N * self.dim * (self.dim + 1)

        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


if __name__ == '__main__':
    x1 = torch.rand((8, 64, 56, 56), dtype=torch.float32)
    x2 = torch.rand((8, 256, 56, 56), dtype=torch.float32)
    x3 = torch.rand((8, 512, 28, 28), dtype=torch.float32)
    x4 = torch.rand((8, 1024, 14, 14), dtype=torch.float32)
    x5 = torch.rand((8, 2048, 7, 7), dtype=torch.float32)
    APCNNCropMine(224, 224)(x1, x2, x3, x4, x5)
