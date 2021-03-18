# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Haihui Gao (gao.haihui@huairuo.ai)
#
# Object for training data.

import os
import random
import torch

from torch.utils.data import Dataset
import json
from tqdm import tqdm
from MRC.dataset.util import read_squad_examples, convert_examples_to_features


from transformers import BertTokenizer





class DataFormat(Dataset):
  """
  BIO data structure for training NER model.
  """
  def __init__(self, root):
    """
    BIO data structure for training NER model.

    Args:
      root(str): Train data root path.
      mode(str): Train mode or test mode.
      max_length(int): Character max length.
      label_dic(dict): Label and index map.
      vocab(dict): All character and index map.
      label_mode(str): Label mode, contain BIOSE or BIO.
    """
    super(DataFormat, self).__init__()
    self.root = root
    self.data = self.read_data(root)
    self.input_ids = torch.LongTensor([temp.input_ids for temp in self.data])
    self.input_mask = torch.LongTensor([temp.input_mask for temp in self.data])
    self.segment_ids = torch.LongTensor([temp.segment_ids for temp in self.data])
    self.start_position = torch.LongTensor([temp.start_position for temp in self.data])
    self.end_position = torch.LongTensor([temp.end_position for temp in self.data])

  def __len__(self):
    """
    Gets data length.

    Returns:
      len(int): Entire data length.
    """
    return len(self.data)

  def __getitem__(self, idx):
    """
    Gets data by idx.

    Args:
      idx(int): Character id.

    Returns:
      input_ids(int): Character.
      masks(int): Mask label.
      tags(int): Label index.
    """
    return self.input_ids[idx], \
           self.input_mask[idx],\
           self.segment_ids[idx],\
           self.start_position[idx],\
           self.end_position[idx]

  def read_data(self, path):
    """
    Reads data from path.

    Args:
      path(str): Data path.

    Returns:
      result(list): List of train data has been formated.
    """
    result = []
    tokenizer = BertTokenizer.from_pretrained(r'D:\self\Graduation\data\bert\vocab.txt', do_lower_case=True)
    # 生成训练数据， train.data

    examples = read_squad_examples(path)
    features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
                                            max_seq_length=512, max_query_length=60)

    return features



