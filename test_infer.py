#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import torch_points_kernels as tp

import numpy as np
import torch

def ball_query_cpu(radius, max_knn, x, y, mode="partial_dense", batch_x=None, batch_y=None):
    # CPU fallback for ball_query: for each query point, find up to max_knn neighbors within radius
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    M = y_np.shape[0]
    neighs = np.zeros((M, max_knn), dtype=np.int64)
    r2 = radius * radius
    for i in range(M):
        diff = x_np - y_np[i]
        dist2 = np.sum(diff * diff, axis=1)
        within = np.where(dist2 <= r2)[0]
        if within.size > 0:
            sorted_idx = within[np.argsort(dist2[within])]
            k = min(sorted_idx.size, max_knn)
            chosen = sorted_idx[:k]
            if k < max_knn:
                chosen = np.concatenate([chosen, np.zeros(max_knn - k, dtype=np.int64)])
        else:
            chosen = np.zeros(max_knn, dtype=np.int64)
        neighs[i] = chosen
    idx_t = torch.from_numpy(neighs).long()
    return (idx_t,)

# override GPU-only ball_query with CPU fallback
tp.ball_query = ball_query_cpu

from util import config, transform
from util.common_util import check_makedirs
from util.voxelize import voxelize

def get_parser():
    p = argparse.ArgumentParser(description='Inference for Stratified Transformer')
    p.add_argument('--config',    required=True, help='path to config YAML')
    p.add_argument('--model_path',required=True, help='.pth checkpoint')
    p.add_argument('--input_pcd', required=True, help='input scene PCD')
    p.add_argument('--output_ply',required=True, help='where to write colored PLY')
    p.add_argument('--gpu',       type=int, default=0, help='which GPU to use')
    p.add_argument('opts',        nargs=argparse.REMAINDER,
                      help='override config options, e.g. DATA.data_name=market')
    return p

def get_logger():
    fmt = "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    return logging.getLogger()

def input_normalize(coord, feat):
    # shift to zero; assume feat already in [0,1]
    coord_min = coord.min(0)
    coord = coord - coord_min
    return coord, feat

def build_model_from_cfg(args):
    if args.arch == 'stratified_transformer':
        from model.stratified_transformer import Stratified
        args.patch_size   = args.grid_size * args.patch_size
        args.window_size  = [args.patch_size * args.window_size * (2**i)
                              for i in range(args.num_layers)]
        args.grid_sizes   = [args.patch_size * (2**i)
                              for i in range(args.num_layers)]
        args.quant_sizes  = [args.quant_size * (2**i)
                              for i in range(args.num_layers)]
        model = Stratified(
            args.downsample_scale, args.depths, args.channels,
            args.num_heads, args.window_size, args.up_k,
            args.grid_sizes, args.quant_sizes,
            rel_query=args.rel_query, rel_key=args.rel_key,
            rel_value=args.rel_value, drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz, num_classes=args.classes,
            ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size,
            sigma=1.0, num_layers=args.num_layers,
            stem_transformer=args.stem_transformer
        )
    else:
        raise RuntimeError(f"Unsupported arch {args.arch}")
    return model

def load_checkpoint(model, path, logger):
    ckpt = torch.load(path, map_location='cpu')
    sd   = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    new  = { k.replace('module.', ''):v for k,v in sd.items() }
    # drop relative-position tables to avoid shape mismatches
    filtered_sd = {k: v for k, v in new.items() if "relative_pos" not in k}
    # ← replace strict=True with strict=False so relative-pos tables get skipped
    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    logger.warning(f"Partially loaded checkpoint '{path}'")
    logger.warning(f"  Missing keys:    {missing}")
    logger.warning(f"  Unexpected keys: {unexpected}")
    return model

def main():
    args = get_parser().parse_args()
    cfg  = config.load_cfg_from_cfg_file(args.config)
    if args.opts:
        # support KEY=VALUE syntax in overrides
        override_list = []
        for o in args.opts:
            if '=' in o:
                k, v = o.split('=', 1)
                override_list += [k, v]
            else:
                override_list.append(o)
        cfg = config.merge_cfg_from_list(cfg, override_list)

    # flatten all cfg sections into args
    for key, val in cfg.items():
        if hasattr(val, 'items'):
            for subk, subv in val.items():
                setattr(args, subk, subv)
        else:
            setattr(args, key, val)

    logger = get_logger()
    logger.info(f"Config and opts: {args}")

    # 1. build & load model
    model = build_model_from_cfg(args).cuda(args.gpu)
    model = load_checkpoint(model, args.model_path, logger)
    model.eval()

    # 2. read PCD
    pcd = o3d.io.read_point_cloud(args.input_pcd)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pcd.has_colors():
        feat = np.asarray(pcd.colors, dtype=np.float32)
    else:
        feat = np.zeros_like(pts, dtype=np.float32)
    N   = pts.shape[0]

    # 3. normalize & voxel‐partition
    coord, feat = input_normalize(pts, feat)
    if args.voxel_size:
        coord_min = coord.min(0)
        coord = coord - coord_min
        idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
        chunks = []
        for i in range(count.max()):
            base = np.insert(count,0,0).cumsum()[:-1]
            idx_part = idx_sort[ base + (i % count) ]
            chunks.append(idx_part)
    else:
        chunks = [ np.arange(N) ]

    # 4. chunk‐wise inference
    pred_scores = np.zeros((N, args.classes), dtype=np.float32)
    for cidx, idx_part in enumerate(chunks):
        logger.info(f"Processing chunk {cidx+1}/{len(chunks)} (pts {idx_part.size})")

        c_pts  = coord[idx_part]
        c_fea  = feat[idx_part]
        # normalize again per‐chunk
        c_pts, c_fea = input_normalize(c_pts, c_fea)

        # build batch offsets
        offsets = torch.IntTensor([c_pts.shape[0]]).cuda(args.gpu)
        batch   = torch.zeros(c_pts.shape[0], dtype=torch.long).cuda(args.gpu)

        # query neighbors
        sigma  = 1.0
        radius = 2.5 * args.grid_size * sigma
        xy     = torch.from_numpy(c_pts).cuda(args.gpu)
        neigh  = tp.ball_query(radius, args.max_num_neighbors,
                               xy, xy, mode="partial_dense",
                               batch_x=batch, batch_y=batch)[0]
        neigh  = neigh.cuda(args.gpu)

        # prepare feature tensor
        f_t = torch.from_numpy(c_fea).cuda(args.gpu)
        if args.concat_xyz:
            f_t = torch.cat([f_t, xy], dim=1)

        # forward
        with torch.no_grad():
            logits = model(f_t, xy, offsets, batch, neigh)
            probs  = F.softmax(logits, -1).cpu().numpy()

        pred_scores[idx_part] += probs

    # 5. average & argmax
    pred_scores /= len(chunks)
    labels     = pred_scores.argmax(axis=1)

    # 6. color & save
    colors = np.zeros((N,3), dtype=np.float32)
    # assume class 1 = target
    colors[labels == 1] = [1,0,0]
    colors[labels != 1] = [0,0,1]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    check_makedirs(os.path.dirname(args.output_ply))
    o3d.io.write_point_cloud(args.output_ply, pcd)
    logger.info(f"Saved result to {args.output_ply}")

if __name__ == '__main__':
    main()

'''
python test_infer.py \
  --config /home/s2671222/Stratified-Transformer/config/market/market_stratified_transformer.yaml \
  --model_path /home/s2671222/Stratified-Transformer/runs/market_stratified_transformer3/model/model_best.pth \
  --input_pcd /home/s2671222/Stratified-Transformer/realworld_scene/realworld_scene_1.pcd \
  --output_ply /home/s2671222/Stratified-Transformer/result/real_world/segmentation/my_scene_mask/scene_mask111.ply \
  DATA.data_name=market \
  TRAIN.voxel_size=0.02 \
  TRAIN.max_num_neighbors=32 \
  TRAIN.grid_size=0.02

'''