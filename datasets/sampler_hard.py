import select
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np


class RandomIdentitySampler_Hard(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, similarity_dict, num_hard_samples):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.similarity_dict = similarity_dict
        self.num_hard_samples = num_hard_samples
        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            random_select_single_pid = random.choice(avai_pids)
            selected_pids_hard = []
            num_hard = 0
            for i in range(len(avai_pids)):
                if self.similarity_dict[random_select_single_pid][i] in avai_pids:
                    selected_pids_hard.append(self.similarity_dict[random_select_single_pid][i])
                    num_hard += 1
                    if num_hard >= self.num_hard_samples - 1:
                        break
            remaining_pids = list(set(avai_pids).difference(set(selected_pids_hard)))
            remaining_pids = list(set(remaining_pids).difference(set([random_select_single_pid])))
            if self.num_pids_per_batch > len(selected_pids_hard):
                selected_pids_random = random.sample(remaining_pids, self.num_pids_per_batch - len(selected_pids_hard) - 1)
                selected_pids = [random_select_single_pid] + selected_pids_hard + selected_pids_random
            else:
                selected_pids = [random_select_single_pid] + selected_pids_hard
            for selected_pid in selected_pids:
                batch_idxs = batch_idxs_dict[selected_pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[selected_pid]) == 0:
                    avai_pids.remove(selected_pid)
        return iter(final_idxs)

    def __len__(self):
        return self.length
