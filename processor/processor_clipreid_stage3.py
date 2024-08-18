import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics_stage3 import R1_mAP_eval
# from utils.metrics_stage3_ranklist import R1_mAP_eval
# from utils.metrics import R1_mAP_eval
# from utils.metrics_ranklist import R1_mAP_eval
from torch.cuda import amp

import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from scipy import spatial
import numpy as np
from datasets.make_dataloader_hard_sample import make_dataloader


def euclidean_dist(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # dist_mat.addmm_(1, -2, qf, gf.t())
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def sort_similarity_matrix(similarity_matrix):

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    # sorted_indices_dict = {row_idx: indices.tolist() for row_idx, indices in enumerate(sorted_indices)}
    sorted_indices_dict = {row_idx: indices[indices != row_idx].tolist() for row_idx, indices in
                           enumerate(sorted_indices)}
    return sorted_indices_dict


def compute_similarity_matrix(features):
    normalized_features = features / torch.norm(features, dim=1, keepdim=True)
    normalized_features = normalized_features.float()
    similarity_matrix = torch.matmul(normalized_features, normalized_features.t())
    return similarity_matrix


def do_train_stage3(cfg,
                    model_clip,
                    train_loader_stage2,
                    val_loader,
                    optimizer,
                    scheduler,
                    loss_fn,
                    num_query, local_rank):
    log_period = cfg.SOLVER.STAGE3.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE3.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE3.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE3.MAX_EPOCHS

    logger = logging.getLogger("transreid.train.stage3")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model_clip.to(local_rank)
        # fusion_model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model_clip)
            num_classes = model.module.num_classes
        else:
            num_classes = model_clip.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter_use_logits = AverageMeter()
    fusion_model = model_clip(get_FusionModel=True)
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING,
                            model=fusion_model)
    scaler = amp.GradScaler()
    # xent = SupConLoss(device)

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    batch = cfg.SOLVER.STAGE3.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes - batch * (num_classes // batch)
    if left != 0:
        i_ter = i_ter + 1

    ########################
    # TGHEMS IMPLEMENTATION
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i + 1 != i_ter:
                l_list = torch.arange(i * batch, (i + 1) * batch)
            else:
                l_list = torch.arange(i * batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model_clip(label=l_list, get_text=True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()
    dist_mat = compute_similarity_matrix(text_features)
    similarity_dict = sort_similarity_matrix(dist_mat)

    num_hard_samplers = cfg.SOLVER.STAGE3.NUM_HARD_SAMPLER
    if num_hard_samplers == 0:
        train_loader = train_loader_stage2
    else:
        train_loader = make_dataloader(cfg, similarity_dict=similarity_dict, num_hard_samples=num_hard_samplers)
    # TGHEMS IMPLEMENTATION
    ########################
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        acc_meter_use_logits.reset()

        model_clip.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            # print(vid)
            optimizer.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            with amp.autocast(enabled=True):
                score, feat, image_features = model_clip(x=img, label=target, Feature_Fusion=True, cam_label=target_cam,
                                                         view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam=target_cam, use_clip_loss=False, use_other_loss=True)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            # lr=scheduler._get_lr(epoch)
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_loader),
                            loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        scheduler.step(epoch)

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model_clip.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model_clip.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model_clip.eval()
                    # fusion_model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else:
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else:
                                target_view = None
                            feat = model_clip(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model_clip.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else:
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else:
                            target_view = None
                        feat = model_clip(img, cam_label=camids, view_label=target_view, Feature_Fusion=False)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


#
def do_inference(cfg,
                 model,
                 fusion_model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
        num_classes = model.num_classes

    model.eval()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING,
                            model=fusion_model)

    evaluator.reset()

    img_path_list = []
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view, Feature_Fusion=False)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
