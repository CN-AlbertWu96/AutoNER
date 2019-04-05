"""
.. module:: dataset
    :synopsis: dataset for sequence labeling
 
.. moduleauthor:: Liyuan Liu, Jingbo Shang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import sys
import pickle
from tqdm import tqdm
import random
import numpy as np

from torch.utils.data import Dataset

class RawDataset(object):
    """    
    Raw Dataset for Sequence Labeling

    Parameters
    ----------
    dataset : ``list``, required.
        The encoded dataset (outputs of preprocess scripts).
    w_pad : ``int``, required.
        The pad index for the word-level inputs.
    c_pad : ``int``, required.
        The pad index for the character-level inputs.
    token_per_batch: ``int``, required.
        Batch size.
    """
    def __init__(self, 
                dataset: list, 
                w_pad: int, 
                c_pad: int, 
                token_per_batch: int):
        super(RawDataset, self).__init__()
        self.dataset = dataset
        self.w_pad = w_pad
        self.c_pad = c_pad
        self.token_per_batch = token_per_batch

        self.construct_index()

    def get_tqdm(self, device):
        """
        construct dataset reader and the corresponding tqdm.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        """
        return tqdm(self.reader(device), mininterval=2, total=self.index_length, leave=False, file=sys.stdout)

    def construct_index(self):
        """
        construct index for the dataset.

        Parameters
        ----------
        dataset: ``list``, required.
            the encoded dataset (outputs of preprocess scripts).        
        """
        self.index_length =len(self.dataset)

    def reader(self, device):
        """
        construct dataset reader.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        Returns
        -------
        reader: ``iterator``.
            A lazy iterable object        
        """
        cur_idx = 0
        while cur_idx < self.index_length:

            batch = self.dataset[cur_idx]

            word_t = torch.LongTensor([batch[0]]).to(device)
            char_t = torch.LongTensor([batch[1]]).to(device)
            chunk_mask = torch.ByteTensor([batch[2]]).to(device)
            chunk_index = torch.LongTensor(batch[3]).to(device)
            chunk_surface = batch[4]
            cur_idx += 1

            yield word_t, char_t, chunk_mask, chunk_index, chunk_surface

class NERDataset(object):
    """
    Evaluation Dataset for Sequence Labeling

    Parameters
    ----------
    dataset : ``list``, required.
        The encoded dataset (outputs of preprocess scripts).
    w_pad : ``int``, required.
        The pad index for the word-level inputs.
    c_pad : ``int``, required.
        The pad index for the character-level inputs.
    token_per_batch: ``int``, required.
        Batch size.
    """
    def __init__(self, 
                dataset: list, 
                w_pad: int, 
                c_pad: int, 
                token_per_batch: int):
        super(NERDataset, self).__init__()
        self.dataset = dataset
        self.w_pad = w_pad
        self.c_pad = c_pad
        self.token_per_batch = token_per_batch

        self.construct_index()

    def shuffle(self):
        """
        shuffle dataset
        """
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device):
        """
        construct dataset reader and the corresponding tqdm.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        """
        return tqdm(self.reader(device), mininterval=2, total=self.index_length, leave=False, file=sys.stdout)

    def construct_index(self):
        """
        construct index for the dataset.    
        """
        dataset_size = len(self.dataset)
        self.index_list = list()
        start_index = 0
        while start_index < dataset_size:
            self.index_list.append(start_index)
            cur_seq_length = len(self.dataset[start_index][0]) - 1
            cur_batch_size = max(int(self.token_per_batch / cur_seq_length), 1)
            start_index = start_index + cur_batch_size
        self.index_length =len(self.index_list)
        self.index_list.append(dataset_size)
        self.shuffle_list = list(range(self.index_length-1, -1, -1))

    def reader(self, device):
        """
        construct dataset reader.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        Returns
        -------
        reader: ``iterator``.
            A lazy iterable object        
        """
        cur_idx = 0
        while cur_idx < self.index_length:
            batch_idx = self.shuffle_list[cur_idx]
            batch = self.dataset[self.index_list[batch_idx]: self.index_list[batch_idx + 1]]
            cur_seq_length = len(batch[0][0])
            word_t = torch.LongTensor([tup[0] + [self.w_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            char_t = torch.LongTensor([tup[1] + [self.c_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            chunk_mask = torch.ByteTensor([tup[2] + [0] * (cur_seq_length - len(tup[2])) for tup in batch]).to(device)
            chunk_label = torch.FloatTensor([label for tup in batch for label in tup[3]]).to(device)
            type_mask = torch.ByteTensor([mask for tup in batch for mask in tup[4]]).to(device)
            label_list = [label for tup in batch for label in tup[5]]
            type_label = torch.FloatTensor(label_list[0:-1]).to(device)
            cur_idx += 1
            yield word_t, char_t, chunk_mask, chunk_label, type_mask, type_label
        self.shuffle()
            
class TrainDataset(object):
    """
    Training Dataset for Sequence Labeling

    Parameters
    ----------
    dataset_name : ``str``, required.
        The name of dataset (outputs of preprocess scripts).
    w_pad : ``int``, required.
        The pad index for the word-level inputs.
    c_pad : ``int``, required.
        The pad index for the character-level inputs.
    token_per_batch: ``int``, required.
        Batch size.
    sample_ratio: ``float``, optional (default = 1.0)
        The ratio for sampling.
    """
    def __init__(self, 
                dataset_name: str, 
                w_pad: int, 
                c_pad: int, 
                token_per_batch: int, 
                sample_ratio: float = 1.0):
        
        super(TrainDataset, self).__init__()
        self.sample_ratio = sample_ratio

        self.dataset_name = dataset_name

        self.w_pad = w_pad
        self.c_pad = c_pad
        self.token_per_batch = token_per_batch

        self.total_batch_num = -1

        self.open_file()

    def get_tqdm(self, device):
        """
        construct dataset reader and the corresponding tqdm.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        """
        return tqdm(self.reader(device), mininterval=2, total=self.total_batch_num, leave=False, file=sys.stdout).__iter__()

    def reader(self, device):
        """
        construct dataset reader.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        Returns
        -------
        reader: ``iterator``.
            A lazy iterable object        
        """
        cur_idx = 0

        while cur_idx < self.index_length:

            batch_idx = self.shuffle_list[cur_idx]
            batch = self.dataset[self.index_list[batch_idx]: self.index_list[batch_idx + 1]]

            cur_seq_length = len(batch[0][0])
            # for tup in batch:
            #     print(cur_seq_length - len(tup[0]), end=" ")
            word_t = torch.LongTensor([tup[0] + [self.w_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            char_t = torch.LongTensor([tup[1] + [self.c_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            chunk_mask = torch.ByteTensor([tup[2] + [0] * (cur_seq_length - len(tup[2])) for tup in batch]).to(device)
            chunk_label = torch.FloatTensor([label for tup in batch for label in tup[3]]).to(device)
            type_mask = torch.ByteTensor([mask for tup in batch for mask in tup[4]]).to(device)
            label_list = [label for tup in batch for label in tup[5]]
            type_label = torch.FloatTensor(label_list[0:-1]).to(device)

            cur_idx += 1

            yield word_t, char_t, chunk_mask, chunk_label, type_mask, type_label

        random.shuffle(self.shuffle_list)

    def open_file(self):
        """
        Open the dataset by name.      
        """
        self.dataset = pickle.load(open(self.dataset_name, 'rb'))

        self.dataset = list(filter(lambda t: random.uniform(0, 1) <= self.sample_ratio, self.dataset))

        dataset_size = len(self.dataset)
        print("dataset_size", dataset_size)
        self.index_list = list()
        start_index = 0
        while start_index < dataset_size:
            self.index_list.append(start_index)
            cur_seq_length = len(self.dataset[start_index][0]) - 1
            cur_batch_size = max(int(self.token_per_batch / cur_seq_length), 1)
            start_index = start_index + cur_batch_size
        self.index_length =len(self.index_list)
        self.index_list.append(dataset_size)
        
        self.shuffle_list = list(range(self.index_length-1, -1, -1))

        self.total_batch_num = self.index_length

class ActiveTrainDataset(object):
    def __init__(self, 
            dataset_name: str, 
            w_pad: int, 
            c_pad: int, 
            token_per_batch: int,
            random_seed: int,
            seed_sample_ratio: float = 0.1,
            sample_ratio: float = 1.0):
        super(ActiveTrainDataset, self).__init__()
        random.seed(random_seed)
        self.sample_ratio = sample_ratio

        self.dataset_name = dataset_name

        self.w_pad = w_pad
        self.c_pad = c_pad
        self.token_per_batch = token_per_batch

        self.active_batch_num = 0
        self.reserved_batch_num = 0

        self.seed_sample_ratio = seed_sample_ratio

        self.active_sample_index = []
        self.reserved_sample_index = []

        self.dataset = None
        self.dataset_size = 0
        self.open_file()
        self.activate()

    def activate(self, to_activate=None):
        print("***********************")
        print("Datasize:", self.dataset_size, "Active:", len(self.active_sample_index))
        if to_activate is None:
            random.shuffle(self.reserved_sample_index)
            num_seed = int(self.dataset_size * self.seed_sample_ratio)
            self.active_sample_index.extend(self.reserved_sample_index[:num_seed])
            self.reserved_sample_index = self.reserved_sample_index[num_seed:]
            print("To activate", num_seed)
        else:
            self.active_sample_index.extend(to_activate)
            to_activate_set = set(to_activate)
            self.reserved_sample_index = list(filter(lambda idx: idx not in to_activate_set, self.reserved_sample_index))
            print("To activate: (from hr):", len(to_activate))
        print("Active Now:", len(self.active_sample_index))
        self.active_dataset, self.active_sample_index,\
            self.active_index_list, self.active_batch_num, self.active_shuffle_list\
            = self._get_index_list(self.active_sample_index)
        self.reserved_dataset, self.reserved_sample_index,\
            self.reserved_index_list, self.reserved_batch_num, self.reserved_shuffle_list\
            = self._get_index_list(self.reserved_sample_index)

        assert len(self.active_sample_index) + len(self.reserved_sample_index) == len(self.dataset), "Error when activate dataset"

    def get_tqdm_active(self, device):
        return tqdm(self.reader_active(device), mininterval=2, total=self.active_batch_num, leave=False, file=sys.stdout).__iter__()

    def get_tqdm_reserved(self, device):
        return tqdm(self.reader_reserved(device), mininterval=2, total=self.reserved_batch_num, leave=False, file=sys.stdout).__iter__()

    def reader_active(self, device):
        for cur_idx in range(self.active_batch_num):
            
            batch_idx = self.active_shuffle_list[cur_idx]
            batch = self.active_dataset[self.active_index_list[batch_idx]: self.active_index_list[batch_idx + 1]]
            sample_index = self.active_sample_index[self.active_index_list[batch_idx]: self.active_index_list[batch_idx + 1]]
            sample_index = np.array(sample_index)
           
            cur_seq_length = len(batch[0][0])
            # print("========================================================================")
            # for tup in batch:
            #     print(cur_seq_length - len(tup[0]), end=" ")
            word_t = torch.LongTensor([tup[0] + [self.w_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            char_t = torch.LongTensor([tup[1] + [self.c_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            chunk_mask = torch.ByteTensor([tup[2] + [0] * (cur_seq_length - len(tup[2])) for tup in batch]).to(device)
            chunk_label = torch.FloatTensor([label for tup in batch for label in tup[3]]).to(device)
            type_mask = torch.ByteTensor([mask for tup in batch for mask in tup[4]]).to(device)
            label_list = [label for tup in batch for label in tup[5]]
            type_label = torch.FloatTensor(label_list[0:-1]).to(device)

            yield word_t, char_t, chunk_mask, chunk_label, type_mask, type_label, sample_index

        random.shuffle(self.active_shuffle_list)
    
    def reader_reserved(self, device):
        for cur_idx in range(self.reserved_batch_num):

            batch_idx = self.reserved_shuffle_list[cur_idx]
            batch = self.reserved_dataset[self.reserved_index_list[batch_idx]: self.reserved_index_list[batch_idx + 1]]
            sample_index = self.reserved_sample_index[self.reserved_index_list[batch_idx]: self.reserved_index_list[batch_idx + 1]]
            sample_index = np.array(sample_index)

            cur_seq_length = len(batch[0][0])
            word_t = torch.LongTensor([tup[0] + [self.w_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            char_t = torch.LongTensor([tup[1] + [self.c_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            chunk_mask = torch.ByteTensor([tup[2] + [0] * (cur_seq_length - len(tup[2])) for tup in batch]).to(device)
            chunk_label = torch.FloatTensor([label for tup in batch for label in tup[3]]).to(device)
            type_mask = torch.ByteTensor([mask for tup in batch for mask in tup[4]]).to(device)
            label_list = [label for tup in batch for label in tup[5]]
            type_label = torch.FloatTensor(label_list[0:-1]).to(device)

            yield word_t, char_t, chunk_mask, chunk_label, type_mask, type_label, sample_index

        random.shuffle(self.reserved_shuffle_list)

    def open_file(self):
        """
        Open the dataset by name.      
        """
        self.dataset = pickle.load(open(self.dataset_name, 'rb'))

        # Initialize the golden seed
        self.dataset = list(filter(lambda t: random.uniform(0, 1) <= self.sample_ratio, self.dataset))

        self.dataset_size = len(self.dataset)
        print("dataset:", self.dataset_name, "size:", self.dataset_size)

        self.reserved_sample_index = list(range(self.dataset_size))

    def _get_index_list(self, sample_index):
        n_samples = len(sample_index)
        if n_samples == 0:
            return [], [], [], 0, []
        sort_dataset = [(self.dataset[idx], idx) for idx in sample_index]
        sort_dataset.sort(key=lambda t: len(t[0][0]), reverse=True)
        new_dataset, new_sample_index = list(zip(*sort_dataset))
        new_sample_index = list(new_sample_index)
        
        assert n_samples == len(new_sample_index), "Error when get index list"

        index_list = list()
        start_index = 0
        while start_index < n_samples:
            index_list.append(start_index)
            cur_seq_length = len(new_dataset[start_index][0]) - 1
            cur_batch_size = max(int(self.token_per_batch / cur_seq_length), 1)
            start_index = start_index + cur_batch_size
        index_length = len(index_list)
        index_list.append(n_samples)

        shuffle_list = list(range(index_length-1, -1, -1))

        return new_dataset, new_sample_index, index_list, index_length, shuffle_list


class DS_GOLD_MIXED_Dataset(object):
    """
    Training Dataset for Sequence Labeling

    Parameters
    ----------
    dataset_name : ``str``, required.
        The name of dataset (outputs of preprocess scripts).
    w_pad : ``int``, required.
        The pad index for the word-level inputs.
    c_pad : ``int``, required.
        The pad index for the character-level inputs.
    token_per_batch: ``int``, required.
        Batch size.
    sample_ratio: ``float``, optional (default = 1.0)
        The ratio for sampling.
    """    
    def __init__(self, 
                dataset_name: str, 
                w_pad: int, 
                c_pad: int, 
                token_per_batch: int, 
                sample_ratio: float = 1.0):
        
        super(DS_GOLD_MIXED_Dataset, self).__init__()
        self.sample_ratio = sample_ratio

        self.dataset_name = dataset_name

        self.w_pad = w_pad
        self.c_pad = c_pad
        self.token_per_batch = token_per_batch

        self.total_batch_num = -1

        self.open_file()

    def get_tqdm(self, device):
        """
        construct dataset reader and the corresponding tqdm.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        """
        return tqdm(self.reader(device), mininterval=2, total=self.total_batch_num, leave=False, file=sys.stdout).__iter__()

    def reader(self, device):
        """
        construct dataset reader.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        Returns
        -------
        reader: ``iterator``.
            A lazy iterable object        
        """
        cur_idx = 0

        while cur_idx < self.index_length:

            batch_idx = self.shuffle_list[cur_idx]
            batch = self.dataset[self.index_list[batch_idx]: self.index_list[batch_idx + 1]]

            cur_seq_length = len(batch[0][0])
            word_t = torch.LongTensor([tup[0] + [self.w_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            char_t = torch.LongTensor([tup[1] + [self.c_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            chunk_mask = torch.ByteTensor([tup[2] + [0] * (cur_seq_length - len(tup[2])) for tup in batch]).to(device)
            chunk_label = torch.FloatTensor([label for tup in batch for label in tup[3]]).to(device)
            type_mask = torch.ByteTensor([mask for tup in batch for mask in tup[4]]).to(device)
            label_list = [label for tup in batch for label in tup[5]]
            type_label = torch.FloatTensor(label_list[0:-1]).to(device)

            cur_idx += 1

            yield word_t, char_t, chunk_mask, chunk_label, type_mask, type_label

        random.shuffle(self.shuffle_list)

    def open_file(self):
        """
        Open the dataset by name.      
        """
        self.dataset = pickle.load(open(self.dataset_name, 'rb'))
        self.dataset = list(filter(lambda t: t[6] or random.uniform(0, 1) <= self.sample_ratio, self.dataset))

        dataset_size = len(self.dataset)
        print(dataset_size)
        self.index_list = list()
        start_index = 0
        while start_index < dataset_size:
            self.index_list.append(start_index)
            cur_seq_length = len(self.dataset[start_index][0]) - 1
            cur_batch_size = max(int(self.token_per_batch / cur_seq_length), 1)
            start_index = start_index + cur_batch_size
        self.index_length =len(self.index_list)
        self.index_list.append(dataset_size)
        
        self.shuffle_list = list(range(self.index_length-1, -1, -1))

        self.total_batch_num = self.index_length
