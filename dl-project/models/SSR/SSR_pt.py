"""
SSR_pt.py: PyTorch-only version (충청도 사투리 주석 포함)
원본: SSR.py (MMCV/mmdet3d 기반)
이 버전은 외부 의존성 없이 PyTorch만 사용함
형님, 이 파일은 연구/실험용이니 참고해서 쓰셔유!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TokenFuser 등은 tokenlearner_pt.py에서 import해서 써야 함
# from .tokenlearner_pt import TokenFuser

class SSRPT(nn.Module):
    """SSR: PyTorch-only 버전. 기존 구조 최대한 유지, 외부 의존성 제거."""
    def __init__(self, embed_dims=256, fut_ts=6, fut_mode=6, valid_fut_ts=6):
        super().__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = valid_fut_ts
        # 필요한 레이어/임베딩 등 직접 정의해서 쓰셔유
        # self.tokenfuser = TokenFuser(...)

    def extract_img_feat(self, img, img_metas=None, len_queue=None):
        # 형님, 여기는 이미지 피처 뽑는 함수여유. 실제 논리 구현해서 쓰셔유!
        pass

    def extract_feat(self, img, img_metas=None, len_queue=None):
        # 형님, 여기는 extract_feat 함수여유. 실제 논리 구현해서 쓰셔유!
        pass

    def forward_pts_train(self, pts_feats, gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d, img_metas, gt_bboxes_ignore=None, map_gt_bboxes_ignore=None, prev_bev=None, next_bev=None, ego_his_trajs=None, ego_fut_trajs=None, ego_fut_masks=None, ego_fut_cmd=None, ego_lcf_feat=None, gt_attr_labels=None):
        # 형님, 여기는 포인트 클라우드 학습용 forward 함수여유. 실제 논리 구현해서 쓰셔유!
        pass

    def forward_dummy(self, img):
        # 형님, 여기는 더미용 forward 함수여유. 실제 논리 구현해서 쓰셔유!
        pass

    def forward(self, return_loss=True, **kwargs):
        # 형님, 여기는 forward 함수여유. 실제 논리 구현해서 쓰셔유!
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self, points=None, img_metas=None, gt_bboxes_3d=None, gt_labels_3d=None, map_gt_bboxes_3d=None, map_gt_labels_3d=None, gt_labels=None, gt_bboxes=None, img=None, proposals=None, gt_bboxes_ignore=None, map_gt_bboxes_ignore=None, img_depth=None, img_mask=None, ego_his_trajs=None, ego_fut_trajs=None, ego_fut_masks=None, ego_fut_cmd=None, ego_lcf_feat=None, gt_attr_labels=None):
        # 형님, 여기는 학습용 forward 함수여유. 실제 논리 구현해서 쓰셔유!
        pass

    def forward_test(self, img_metas, gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d, img=None, ego_his_trajs=None, ego_fut_trajs=None, ego_fut_cmd=None, ego_lcf_feat=None, gt_attr_labels=None, **kwargs):
        # 형님, 여기는 테스트용 forward 함수여유. 실제 논리 구현해서 쓰셔유!
        pass

    def simple_test(self, img_metas, gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d, img=None, prev_bev=None, points=None, fut_valid_flag=None, rescale=False, ego_his_trajs=None, ego_fut_trajs=None, ego_fut_cmd=None, ego_lcf_feat=None, gt_attr_labels=None, **kwargs):
        # 형님, 여기는 simple_test 함수여유. 실제 논리 구현해서 쓰셔유!
        pass

    def simple_test_pts(self, x, img_metas, gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d, prev_bev=None, fut_valid_flag=None, rescale=False, start=None, ego_his_trajs=None, ego_fut_trajs=None, ego_fut_cmd=None, ego_lcf_feat=None, gt_attr_labels=None):
        # 형님, 여기는 simple_test_pts 함수여유. 실제 논리 구현해서 쓰셔유!
        pass

    def map_pred2result(self, bboxes, scores, labels, pts, attrs=None):
        # 형님, 여기는 map_pred2result 함수여유. 실제 논리 구현해서 쓰셔유!
        pass

    def compute_planner_metric_stp3(self, pred_ego_fut_trajs, gt_ego_fut_trajs, gt_agent_boxes, gt_agent_feats, gt_map_boxes, gt_map_labels, fut_valid_flag):
        # 형님, 여기는 compute_planner_metric_stp3 함수여유. 실제 논리 구현해서 쓰셔유!
        pass

    def set_epoch(self, epoch):
        # 형님, 여기는 set_epoch 함수여유. 실제 논리 구현해서 쓰셔유!
        pass 