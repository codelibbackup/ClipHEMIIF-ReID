import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, qf=None, gf=None, model=None, num_fussion=150):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    gf_temp = gf.clone()
    for q_idx in range(num_q):
        gf_temp[:] = gf
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        g_pids_num_fusion = g_pids[indices[q_idx]][:num_fussion]
        g_camids_num_fusion = g_camids[indices[q_idx]][:num_fussion]
        indices_num_fusion = indices[q_idx][:num_fussion]
        gallery_feature_need_fusion = gf_temp[indices[q_idx]][:num_fussion]
        feature_need_fusion = torch.cat((gallery_feature_need_fusion, qf[q_idx].reshape(1, 1024)), dim=0)
        feature_need_fusion = model(feature_need_fusion.to("cuda"))
        feature_need_fusion = torch.nn.functional.normalize(feature_need_fusion, dim=1, p=2)
        new_query_feature = feature_need_fusion[-1, :].reshape(1, 1024)
        new_gallery_featuer = feature_need_fusion[:-1, :]
        distmat_new = euclidean_distance(new_query_feature.cpu().detach(), new_gallery_featuer.cpu().detach())
        indices_new = np.argsort(distmat_new, axis=1)
        matches = (g_pids_num_fusion[indices_new] == q_pids[q_idx, np.newaxis]).astype(np.int32)
        order = indices_new
        remove = (g_pids_num_fusion[order] == q_pid) & (g_camids_num_fusion[order] == q_camid)
        keep = np.invert(remove)
        ###########################################

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        if len(orig_cmc) < max_rank:
            padded_cmc = np.pad(orig_cmc, (0, max_rank - len(orig_cmc)), 'constant', constant_values=(0,))
        else:
            padded_cmc = orig_cmc
        cmc = padded_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:15])
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    mAP = np.mean(all_AP)
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q


    return all_cmc, mAP



class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, model=None):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.model = model

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, qf=qf, gf=gf, model=self.model)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf
