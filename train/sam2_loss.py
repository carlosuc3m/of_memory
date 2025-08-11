# vgg19_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional, Sequence, Union, List, Tuple
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor

import random

class SAM2Loss(nn.Module):
    """
    Extracts intermediate feature maps from a pretrained VGG‑19 network,
    corresponding to layers conv1_2, conv2_2, conv3_2, conv4_2, conv5_2.
    Input images are expected in [0,1] RGB; this module multiplies by 255
    and subtracts ImageNet means to match the original TF implementation.
    """
    def __init__(self, device: torch.device = torch.device('cpu'), loss_weights=[1, 0.1], eps=1e-6):
        super().__init__()

        def load_encoder():
            # Example stub — replace with your actual model
            from sam2.build_sam import build_sam2

            sam_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", ckpt_path="/home/carlos/Downloads/sam2.1_hiera_large.pt")
            return SAM2ImagePredictor(sam_model.to(device).eval())
        # Load pretrained VGG‑19 up to the last conv layer we need
        self.sam2_predictor = load_encoder()
        self.sam2_predictor.model.image_encoder.cpu()
        self.sam2_predictor.model.use_high_res_features_in_sam = False
        self.sam2_predictor.model.sam_mask_decoder.use_high_res_features = False
        # Freeze parameters
        for p in self.sam2_predictor.model.parameters():
            p.requires_grad = False
        self.eps = eps
        self.ww = loss_weights

        self.prompts = None

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """This function is very similar to predict(...), however it is used for batched mode, when the model is expected to generate predictions on multiple images.
        It returns a tuple of lists of masks, ious, and low_res_masks_logits.
        """
        num_images = len(self.sam2_predictor._features["image_embed"])
        for img_idx in range(num_images):
            # Transform input prompts
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input, unnorm_coords, labels, unnorm_box = self.sam2_predictor._prep_prompts(
                point_coords,
                point_labels,
                box,
                None,
                normalize_coords,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
        return masks, iou_predictions, low_res_masks
    

    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.sam2_predictor.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self.sam2_predictor._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.sam2_predictor.model.sam_mask_decoder(
            image_embeddings=self.sam2_predictor._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.sam2_predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = self.sam2_predictor._transforms.postprocess_masks(
            low_res_masks, self.sam2_predictor._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def forward(self, gt_encoding, predicted_encoding) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] RGB in [0,1]
        Returns:
            Dict mapping each layer name to its [B,C,H_i,W_i] activations.
        """
        # Scale to [0,255] and subtract mean
        self.sam2_predictor._features = {}
        self.sam2_predictor._features["image_embed"] = gt_encoding
        self.sam2_predictor._features["high_res_feats"] = []
        self.sam2_predictor._orig_hw = [[1024, 1024]] * gt_encoding.shape[0]

        if self.prompts == None:
            self.prompts = self.create_random_prompts()
        
        prompts = self.prompts

        t_masks, t_ious, _ = self.predict_batch(
            point_coords_batch=prompts["point_coords_batch"] if "point_coords_batch" in prompts.keys() else None,
            point_labels_batch=prompts["labels"],
            box_batch=prompts["box_batch"] if "box_batch" in prompts.keys() else None,
            multimask_output=True,
            return_logits=True,
            )
        self.sam2_predictor._features = {}
        self.sam2_predictor._features["image_embed"] = predicted_encoding
        self.sam2_predictor._features["high_res_feats"] = []

        s_masks, s_ious, _ = self.predict_batch(
            point_coords_batch=prompts["point_coords_batch"] if "point_coords_batch" in prompts.keys() else None,
            point_labels_batch=prompts["labels"],
            box_batch=prompts["box_batch"] if "box_batch" in prompts.keys() else None,
            multimask_output=True,
            return_logits=True,
            )
        
        T = 1.0
        t_logits_T = t_masks / T
        s_logits_T = s_masks / T

        t_probs_T = torch.sigmoid(t_logits_T)           # soft targets
        """
        s_logprobs_T = F.logsigmoid(s_logits_T)         # student log‑probs

        kd_loss = F.kl_div(
            input    = s_logprobs_T,
            target   = t_probs_T.detach(),
            reduction= "batchmean",
        ) * (T*T)

        kd_loss = F.binary_cross_entropy_with_logits(
            input  = s_logits_T,
            target = t_probs_T
        ) * (T*T)
        """

        # use the softened teacher probabilities as targets
        bce_loss = F.binary_cross_entropy_with_logits(
            input = s_masks, 
            target= t_probs_T.detach(), 
        )
        iou_loss = F.mse_loss(
            input = s_ious, 
            target= t_ious.detach()
        )

        """
        p = torch.sigmoid(s_masks)
        q = (t_probs_T.detach() >= 0.5).float()
        inter = (p * q).sum(dim=[2,3])
        union = p.sum(dim=[2,3]) + q.sum(dim=[2,3])
        dice_loss = 1 - ((2 * inter + self.eps) / (union + self.eps)).mean()
        """

        #return self.ww[0] * kd_loss, self.ww[1] * bce_loss, self.ww[2] * iou_loss, self.ww[3] * dice_loss
        return self.ww[0] * bce_loss, self.ww[1] * iou_loss


    def create_random_prompts(self):
        prompts = {}
        if random.randint(0, 1) == 1:
            box, label = self.create_random_box_prompt()
            prompts["box_batch"] = box
            prompts["labels"] = label
        else:
            points, labels = self.create_random_point_prompt()
            prompts["point_coords_batch"] = points
            prompts["labels"] = labels
        return prompts
    

    def create_random_point_prompt(self):
        """
        Point prompts are (x, y)
        """
        n_batch = self.sam2_predictor._features["image_embed"].shape[0]
        n_prompts = random.randint(0, 1)
        if n_prompts == 0:
            n_prompts = random.randint(2, 5)
        prompts = np.zeros((n_batch, n_prompts, 2), dtype="uint32")
        labels = np.ones((n_batch, n_prompts), dtype="uint32")
        for i in range(n_prompts):
            for b in range(n_batch):
                prompts[b, i] = create_random_point_prompt(1024, 1024)
        return prompts, labels

    def create_random_box_prompt(self):
        """
        Box prompts are (x0, y0, x1, y1)
        """
        n_batch = self.sam2_predictor._features["image_embed"].shape[0]
        prompts = np.zeros((n_batch, 4), dtype="uint32")
        labels = np.ones((n_batch, 1), dtype="uint32")
        for i in range(n_batch):
            prompts[i] = create_random_box_prompt()
        return prompts, labels
    

def create_random_point_prompt(max_w, max_h):
    return np.array([random.randint(0, max_w), random.randint(0, max_h)])

def create_random_box_prompt():
        side_1 = random.randint(40, 650)
        ratio = random.uniform(1, 5)
        if random.randint(0, 1) == 0:
            side_2 = min(1000, int(side_1 * ratio))
        else:
            side_2 = max(40, int(side_1 / ratio))

        pos_1 = random.randint(0, 1024 - side_1)
        pos_2 = random.randint(0, 1024 - side_2)
        return np.array([pos_1, pos_2, pos_1 + side_1, pos_2 + side_2])


if __name__ == '__main__':
    loss = SAM2Loss()
    enc1 = torch.from_numpy(np.random.random_sample((2, 256, 64, 64)).astype("float32")).cpu()
    enc2 = torch.from_numpy(np.random.random_sample((2, 256, 64, 64)).astype("float32")).cpu()
    loss(enc1, enc2)