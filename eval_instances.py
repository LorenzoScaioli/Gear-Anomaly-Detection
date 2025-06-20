from argparse import ArgumentParser
import torch
from models.evaluator import *
import models.basic_model as basic # import CDEvaluator

import cv2
import numpy as np
import os

print(torch.cuda.is_available())

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

def filter_components(mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)
    kept_ids = []
    for label in range(1, num_labels):  # do not consider label 0 (background)
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_mask[labels == label] = 1
            kept_ids.append(label)
    return filtered_mask, labels, kept_ids

def evaluate_instance_segmentation(pred_mask, gt_mask, iou_threshold=0.5, min_area_pred=10, min_area_gt=10):
    # Label the instances in the predicted and ground truth masks (the connected comeponents)
    _, gt_labeled, gt_ids = filter_components(gt_mask, min_area_gt)
    _, pred_labeled, pred_ids = filter_components(pred_mask, min_area_pred)

    # num_gt, gt_labeled = cv2.connectedComponents(gt_mask.astype(np.uint8))
    # num_pred, pred_labeled = cv2.connectedComponents(pred_mask.astype(np.uint8))

    gt_matched = set()
    pred_matched = set()
    iou_matches = []

    # Iterate over all predicted instances
    for pred_id in pred_ids:  #range(1, num_pred):  # 0 is background
        pred_instance = (pred_labeled == pred_id)

        best_iou = 0
        best_gt_id = -1

        # Compare with every ground truth instance
        for gt_id in gt_ids: #range(1, num_gt):
            if gt_id in gt_matched:
                continue  # Already matched

            gt_instance = (gt_labeled == gt_id)
            iou = compute_iou(pred_instance, gt_instance)

            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id

        # If we have a valid match
        if best_gt_id != -1:
            gt_matched.add(best_gt_id)
            pred_matched.add(pred_id)
            iou_matches.append((pred_id, best_gt_id, best_iou))

    TP = len(iou_matches)
    # FP = (num_pred - 1) - TP
    FP = len(pred_ids) - TP
    # FN = (num_gt - 1) - TP
    FN = len(gt_ids) - TP

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Matches': iou_matches  # tuples' list (pred_id, gt_id, iou)
    }

def draw_overlay(original_img, pred_mask, gt_mask, alpha=0.5, min_area_pred=10, min_area_gt=10):
    gt_mask_filtered, _, _ = filter_components(gt_mask, min_area_gt)
    pred_mask_filtered, _, _ = filter_components(pred_mask, min_area_pred)

    original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    if len(original_img_bgr.shape) == 2 or original_img_bgr.shape[2] == 1:
        original_img_bgr = cv2.cvtColor(original_img_bgr, cv2.COLOR_GRAY2BGR)

    overlay = original_img_bgr.copy()
    # Red for predictions
    overlay[pred_mask_filtered == 1] = (0, 0, 255)
    # Green for GT
    overlay[gt_mask_filtered == 1] = (0, 255, 0)
    # Where pred and gt overlap → yellow
    overlay[np.logical_and(pred_mask_filtered == 1, gt_mask_filtered == 1)] = (0, 255, 255)

    blended = cv2.addWeighted(overlay, alpha, original_img_bgr, 1 - alpha, 0)
    return blended

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--training_name', type=str, help='name of the training')
    parser.add_argument('--pred_masks', type=str, default='predict', help='path for folder with predicted masks')
    parser.add_argument('--gt_masks', type=str, default='label', help='path for folder with groundtruth masks')
    parser.add_argument('--original_imgs', type=str, default='B', help='path for folder with original images')
    parser.add_argument('--blended_mask', type=str, default='0', help='path for folder where to save the blended masks')

    args = parser.parse_args()
    
    training_name = args.training_name
    input_folder_path = f"vis/{training_name}"
    output_folder_path = f"output_analysis/{training_name}"
    
    pred_masks_path = os.path.join(output_folder_path, args.pred_masks)
    gt_masks_path = args.gt_masks
    original_imgs_path = args.original_imgs
    min_area_pred = 300
    min_area_gt = 100

    if args.blended_mask != '0':
        blended_mask_path = os.path.join(output_folder_path, args.blended_mask)
        os.makedirs(blended_mask_path, exist_ok=True)

    TP = 0
    FP = 0
    FN = 0

    for filename in os.listdir(pred_masks_path):
        pred_mask = cv2.imread(os.path.join(pred_masks_path, filename), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(os.path.join(gt_masks_path, filename), cv2.IMREAD_GRAYSCALE)
        original_img = cv2.imread(os.path.join(original_imgs_path, filename), cv2.IMREAD_GRAYSCALE)
        
        # If needed, convert to boolean
        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)

        result = evaluate_instance_segmentation(pred_mask, gt_mask, iou_threshold=0.05, min_area_pred=min_area_pred, min_area_gt=min_area_gt)
        
        TP += result["TP"]
        FP += result["FP"]
        FN += result["FN"]
        if result["FN"] > 0:
            print(f"File: {filename}, TP: {result['TP']}, FP: {result['FP']}, FN: {result['FN']}, Precision: {result['Precision']:.4f}, Recall: {result['Recall']:.4f}, F1: {result['F1']:.4f}")

        if args.blended_mask != '0':
            save_path=os.path.join(blended_mask_path, filename)
            blended=draw_overlay(original_img, pred_mask, gt_mask, alpha=0.5, min_area_pred=min_area_pred, min_area_gt=min_area_gt)
            cv2.imwrite(save_path, blended)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

if __name__ == '__main__':
    main()