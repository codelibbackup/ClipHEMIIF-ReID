# encoding: utf-8

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .centroid_triplet_loss import CentroidLoss


def make_loss(cfg, num_classes):  # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            ctl_loss = CentroidLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            ctl_loss = CentroidLoss(cfg.SOLVER.MARGIN)
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, i2tscore=None, use_clip_loss=True, use_other_loss=True):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if use_other_loss:
                        if isinstance(score, list):
                            ID_LOSS = [xent(scor, target) for scor in score[0:]]
                            ID_LOSS = sum(ID_LOSS)
                        else:
                            ID_LOSS = xent(score, target)

                        if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                            CTL_LOSS = [ctl_loss(feats, target)[0] for feats in feat[0:]]
                            CTL_LOSS = sum(CTL_LOSS)
                        else:
                            TRI_LOSS = triplet(feat, target)[0]
                            CTL_LOSS = ctl_loss(feat, target)[0]

                        loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + cfg.MODEL.CTL_LOSS_WEIGHT * CTL_LOSS
                    if use_clip_loss and use_other_loss:
                        if i2tscore != None:
                            I2TLOSS = xent(i2tscore, target)
                            loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                    elif use_clip_loss and not use_other_loss:
                        if i2tscore != None:
                            I2TLOSS = xent(i2tscore, target)
                            loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS
                    # print('ID_LOSS: {:.3f} TRI_LOSS: {:.3f} CTL_LOSS: {:.3f} CLIP_Loss:{:.3f}'.format(cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS, cfg.MODEL.TRIPLET_LOSS_WEIGHT *TRI_LOSS,cfg.MODEL.CTL_LOSS_WEIGHT * CTL_LOSS,cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS))
                    return loss
                else:
                    if use_other_loss:
                        if isinstance(score, list):
                            ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                            ID_LOSS = sum(ID_LOSS)
                        else:
                            ID_LOSS = F.cross_entropy(score, target)

                        if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                            CTL_LOSS = [ctl_loss(feats, target)[0] for feats in feat[0:]]
                            CTL_LOSS = sum(CTL_LOSS)
                        else:
                            TRI_LOSS = triplet(feat, target)[0]
                            CTL_LOSS = ctl_loss(feat, target)[0]

                        loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS +cfg.MODEL.CTL_LOSS_WEIGHT * CTL_LOSS

                    if use_clip_loss and use_other_loss:
                        if i2tscore != None:
                            I2TLOSS = xent(i2tscore, target)
                            loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                    elif use_clip_loss and not use_other_loss:
                        if i2tscore != None:
                            I2TLOSS = xent(i2tscore, target)
                            loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS
                    # print('====================================')
                    # print('ID_LOSS: {:.3f} TRI_LOSS: {:.3f} CTL_LOSS: {:.3f} I2T_LOSS:{:.3f}  Loss:{:.3f}'.format(cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS, cfg.MODEL.TRIPLET_LOSS_WEIGHT *TRI_LOSS, cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS,cfg.MODEL.CTL_LOSS_WEIGHT * CTL_LOSS,loss))
                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
