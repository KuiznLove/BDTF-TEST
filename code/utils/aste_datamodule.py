import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
# from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict, defaultdict
import os
from . import load_json

# import random
# from torch.utils.data.sampler import Sampler

from prepare_vocab import VocabHelp

polarity_map = {
    'NEG': 0,
    'NEU': 1,
    'POS': 2
}

polarity_map_reversed = {
    0: 'NEG',
    1: 'NEU',
    2: 'POS'
}


class Example:
    def __init__(self, data, max_length=-1):
        self.data = data
        self.max_length = max_length
        self.data['tokens'] = eval(str(self.data['tokens']))

    def __getitem__(self, key):
        return self.data[key]

    def table_label(self, length, ty, id_len):
        label = [[-1 for _ in range(length)] for _ in range(length)]
        id_len = id_len.item()-9

        for i in range(1, id_len - 1):
            for j in range(1, id_len - 1):
                label[i][j] = 0
        for t_start, t_end, o_start, o_end, pol in self['pairs']:
            if ty == 'S':
                label[t_start + 1][o_start + 1] = 1
            elif ty == 'E':
                label[t_end][o_end] = 1
        return label

    def seq_label(self, length, ty, id_len):
        label = [-1 for _ in range(length)]
        id_len = id_len.item()-7

        for i in range(1, id_len - 1):
            label[i] = 0
        for t_start, t_end, o_start, o_end, pol in self['pairs']:
            if ty == 'S':
                for j in range(t_start+1, t_end+1):
                    label[j] = 2
                label[t_start + 1] = 1
            elif ty == 'E':
                for j in range(o_start+1, o_end+1):
                    label[j] = 2
                label[o_start + 1] = 1
        return label

    def mask_label(self, length, id_len):
        label = [0 for _ in range(length)]
        # id_len = id_len.item()
        for i in self['mask_position']:
                label[i+1] = 1
        return label

    def set_pairs(self, pairs):
        self.data['pairs'] = pairs


class DataCollatorForASTE:
    # def __init__(self, tokenizer, max_seq_length, post_vocab, deprel_vocab, postag_vocab, synpost_vocab):
    def __init__(self, tokenizer, max_seq_length, post_vocab, deprel_vocab): # 删掉了两个关系矩阵
    # def __init__(self, tokenizer, max_seq_length, post_vocab, deprel_vocab, postag_vocab): # 加上了词性关系矩阵
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        #------------------------------------------------------
        self.post_vocab = post_vocab
        self.deprel_vocab = deprel_vocab
        #------------------------------------------------------
        # self.postag_vocab = postag_vocab
        # self.synpost_vocab = synpost_vocab
        #------------------------------------------------------

    def __call__(self, examples):

        batch = self.tokenizer_function(examples)

        length = batch['input_ids'].size(1)

        batch['example_ids'] = [example['ID'] for example in examples]
        batch['table_labels_S'] = torch.tensor(
            [examples[i].table_label(length, 'S', (batch['input_ids'][i] > 0).sum()) for i in range(len(examples))],
            dtype=torch.long)
        batch['table_labels_E'] = torch.tensor(
            [examples[i].table_label(length, 'E', (batch['input_ids'][i] > 0).sum()) for i in range(len(examples))],
            dtype=torch.long)
        # --------------------------------------------------------
        # 注释掉术语提示掩码部分mask_position
        # batch['mask_position'] = torch.tensor(
        #     [examples[i].mask_label(length, (batch['input_ids'][i] > 0).sum()) for i in range(len(examples))],
        #     dtype=torch.long)
        # --------------------------------------------------------

        al = [example['pairs'] for example in examples]
        pairs_ret = []
        for pairs in al:
            pairs_chg = []
            for p in pairs:
                pairs_chg.append([p[0], p[1], p[2], p[3], polarity_map[p[4]] + 1])
            pairs_ret.append(pairs_chg)
        batch['pairs_true'] = pairs_ret

        return {
            'ids': batch['example_ids'],
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],

            'table_labels_S': batch['table_labels_S'],
            'table_labels_E': batch['table_labels_E'],
            # 'mask_position': batch['mask_position'],

            'pairs_true': batch['pairs_true'],
            #--------------------------------------
            'word_pair_position': batch['word_pair_position'],
            'word_pair_deprel': batch['word_pair_deprel'],
            #--------------------------------------
            'word_pair_pos': batch['word_pair_pos'],
            # 'word_pair_synpost': batch['word_pair_synpost'],
            #--------------------------------------
            #--------------------------------------
            # 'contrast_mask': batch['contrast_mask'],
            #--------------------------------------
            'aspect_mask': batch['aspect_mask'],
            'opinion_mask': batch['opinion_mask'],
            #--------------------------------------
        }

    def tokenizer_function(self, examples):
        # text = [example['sentence'] for example in examples]
        text = [example['sentence'] for example in examples]
        kwargs = {
            'text': text,
            'return_tensors': 'pt'
        }

        if self.max_seq_length in (-1, 'longest'):
            kwargs['padding'] = True
        else:
            kwargs['padding'] = 'max_length'
            kwargs['max_length'] = self.max_seq_length
            kwargs['truncation'] = True

        batch_encodings = self.tokenizer(**kwargs)
        word_pair_positions = []
        word_pair_deprels = []
        word_pair_poss = []
        word_pair_synposts = []
        # ------------------------------------------------------
        # 新增对比学习掩码
        contrast_masks = []
        seq_length = len(batch_encodings['input_ids'][0])
        # ------------------------------------------------------
        # ------------------------------------------------------
        # 新增方面术语掩码
        aspect_masks = []
        opinion_masks =[]
        # ------------------------------------------------------
        for example in examples:
            txt = example['sentence']
            txt0 = txt[:-72]
            tokens = txt0.strip().split()
            # tokens = txt.strip().split()
            head = example['head']
            postag = example['postag']
            deprel = example['deprel']
            pairs = example['pairs']
            # ------------------------------------------------------
            # 提取方面术语掩码
            # aspect_mask = self.prepare_aspect_mask(seq_length, pairs)
            aspect_mask, opinion_mask = self.prepare_aspect_mask(seq_length, pairs)
            aspect_masks.append(aspect_mask)
            opinion_masks.append(opinion_mask)
            # ------------------------------------------------------
            token_range = []
            token_start = 1
            for i, w, in enumerate(tokens):
                token_end = token_start + len(self.tokenizer.encode(w, add_special_tokens=False))
                token_range.append([token_start, token_end - 1])
                token_start = token_end
            '''1. generate position index of the word pair'''
            '''word_pair_position：为词对生成依赖距离索引。'''
            word_pair_position = torch.zeros(seq_length, seq_length).long()
            for i in range(len(tokens)):
                start, end = token_range[i][0], token_range[i][1]
                for j in range(len(tokens)):
                    s, e = token_range[j][0], token_range[j][1]
                    for row in range(start, end + 1):
                        for col in range(s, e + 1):
                            word_pair_position[row][col] = self.post_vocab.stoi.get(abs(row - col),
                                                                                    self.post_vocab.unk_index)

            """2. generate deprel index of the word pair"""
            '''word_pair_deprel：为词对生成依赖关系标签。'''
            word_pair_deprel = torch.zeros(seq_length, seq_length).long()
            for i in range(len(tokens)):
                start = token_range[i][0]
                end = token_range[i][1]
                for j in range(start, end + 1):
                    s, e = token_range[head[i] - 1] if head[i] != 0 else (0, 0)
                    for k in range(s, e + 1):
                        word_pair_deprel[j][k] = self.deprel_vocab.stoi.get(deprel[i])
                        word_pair_deprel[k][j] = self.deprel_vocab.stoi.get(deprel[i])
                        word_pair_deprel[j][j] = self.deprel_vocab.stoi.get('self')
            # ---------------------------------------------------------------------
            """3. generate POS tag index of the word pair"""
            '''word_pair_pos：为词对生成词性标注索引。'''
            # word_pair_pos = torch.zeros(seq_length, seq_length).long()
            # for i in range(len(tokens)):
            #     start, end = token_range[i][0], token_range[i][1]
            #     for j in range(len(tokens)):
            #         s, e = token_range[j][0], token_range[j][1]
            #         for row in range(start, end + 1):
            #             for col in range(s, e + 1):
            #                 word_pair_pos[row][col] = self.postag_vocab.stoi.get(
            #                     tuple(sorted([postag[i], postag[j]])))
            #---------------------------------------------------------------------
            #词性增强邻接矩阵
            # word_pair_pos = (word_pair_deprel != 0).int()
            word_pair_pos = (word_pair_deprel != 0).int().clone().detach()
            # 找postag中为'NN'和'JJ'的索引
            pos_list = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']
            nn_jj_index = []
            for i in range(len(postag)):
                if postag[i] in pos_list:
                    nn_jj_index.append(i)
            nn_jj_bert_index = []

            for i in nn_jj_index:
                start = token_range[i][0]
                end = token_range[i][1]
                for j in range(start, end + 1):
                    nn_jj_bert_index.append(j)

            for i in range(seq_length):
                for j in range(i, seq_length):
                    if word_pair_pos[i][j] == 1:
                        if i in nn_jj_bert_index or j in nn_jj_bert_index:
                            word_pair_pos[i][j] = word_pair_pos[i][j] + 1
                            # word_pair_pos[i][j] = word_pair_pos[i][j] + 0
                            word_pair_pos[j][i] = word_pair_pos[i][j]
            # --------------------------------------------------------------------
            # """4. generate synpost index of the word pair"""
            # '''word_pair_synpost：通过句法位置生成词对的语法层级关系。'''
            # word_pair_synpost = torch.zeros(104, 104).long()
            # tmp = [[0] * len(tokens) for _ in range(len(tokens))]
            # for i in range(len(tokens)):
            #     j = head[i]
            #     if j == 0:
            #         continue
            #     tmp[i][j - 1] = 1
            #     tmp[j - 1][i] = 1
            #
            # tmp_dict = defaultdict(list)
            # for i in range(len(tokens)):
            #     for j in range(len(tokens)):
            #         if tmp[i][j] == 1:
            #             tmp_dict[i].append(j)
            #
            # word_level_degree = [[4] * len(tokens) for _ in range(len(tokens))]
            #
            # for i in range(len(tokens)):
            #     node_set = set()
            #     word_level_degree[i][i] = 0
            #     node_set.add(i)
            #     for j in tmp_dict[i]:
            #         if j not in node_set:
            #             word_level_degree[i][j] = 1
            #             node_set.add(j)
            #         for k in tmp_dict[j]:
            #             if k not in node_set:
            #                 word_level_degree[i][k] = 2
            #                 node_set.add(k)
            #                 for g in tmp_dict[k]:
            #                     if g not in node_set:
            #                         word_level_degree[i][g] = 3
            #                         node_set.add(g)
            #
            # for i in range(len(tokens)):
            #     start, end = token_range[i][0], token_range[i][1]
            #     for j in range(len(tokens)):
            #         s, e = token_range[j][0], token_range[j][1]
            #         for row in range(start, end + 1):
            #             for col in range(s, e + 1):
            #                 word_pair_synpost[row][col] = self.synpost_vocab.stoi.get(word_level_degree[i][j],
            #                                                                           self.synpost_vocab.unk_index)
            # ---------------------------------------------------------------------
            word_pair_positions.append(word_pair_position)
            word_pair_deprels.append(word_pair_deprel)
            # -------------------------------------------
            word_pair_poss.append(word_pair_pos)
            # word_pair_synposts.append(word_pair_synpost)
            # -------------------------------------------
        word_pair_position = torch.stack(word_pair_positions)
        word_pair_deprel = torch.stack(word_pair_deprels)
        # ------------------------------------------------------
        word_pair_pos = torch.stack(word_pair_poss)
        # word_pair_synpost = torch.stack(word_pair_synposts)
        # ------------------------------------------------------
        # ------------------------------------------------------
        aspect_mask = torch.stack(aspect_masks)  # 新增：堆叠方面掩码
        opinion_mask = torch.stack(opinion_masks)  # 新增：堆叠观点掩码
        # ------------------------------------------------------
        batch_encodings = dict(batch_encodings)

        batch_encodings['word_pair_position'] = word_pair_position
        batch_encodings['word_pair_deprel'] = word_pair_deprel
        # ------------------------------------------------------
        batch_encodings['word_pair_pos'] = word_pair_pos
        # batch_encodings['word_pair_synpost'] = word_pair_synpost
        # ------------------------------------------------------
        # ------------------------------------------------------
        batch_encodings['aspect_mask'] = aspect_mask  # 新增：方面掩码
        batch_encodings['opinion_mask'] = opinion_mask  # 新增：观点掩码
        # ------------------------------------------------------
        # 添加sentence，需要修改ptgcn_model.py的forward参数
        batch_encodings['sentence'] = [example['sentence'] for example in examples]
        return batch_encodings

    def prepare_aspect_mask(self, seq_len, pairs):
        """
        根据pairs信息生成aspect位置掩码
        参数：
        seq_len : int - bert分词序列长度
        pairs : list - 格式[[start1, end1, start2, end2, polarity], ...]
        返回：
        torch.Tensor - 形状为(len(tokens),)的二进制掩码
        """
        aspect_mask = torch.zeros(seq_len, dtype=torch.bool)
        opinion_mask = torch.zeros(seq_len, dtype=torch.bool)
        for pair in pairs:
            # 解析方面词位置信息
            aspect_start = pair[0]
            aspect_end = pair[1]
            # ---------------------------------
            # 新增观点词位置信息
            opinion_start = pair[2]
            opinion_end = pair[3]
            # ---------------------------------
            # 设置掩码（注意Python的切片是左闭右开）
            aspect_mask[aspect_start:aspect_end] = 1
            opinion_mask[opinion_start:opinion_end] = 1
        return aspect_mask, opinion_mask

class ASTEDataModule(pl.LightningDataModule):
    def __init__(self,
                 model_name_or_path: str = '',
                 max_seq_length: int = -1,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 data_dir: str = '',
                 num_workers: int = 4,
                 cuda_ids: int = -1,
                 ):

        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length if max_seq_length > 0 else 'longest'
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.cuda_ids = cuda_ids

        self.table_num_labels = 6  # 4
        #-------------------------------------------------------------------------------
        self.post_vocab = VocabHelp.load_vocab(self.data_dir + '/vocab_post.vocab')
        self.deprel_vocab = VocabHelp.load_vocab(self.data_dir + '/vocab_deprel.vocab')
        # ---------------------------------------------------------------------------------
        self.postag_vocab = VocabHelp.load_vocab(self.data_dir + '/vocab_postag.vocab')
        # self.synpost_vocab = VocabHelp.load_vocab(self.data_dir + '/vocab_synpost.vocab')
        # ---------------------------------------------------------------------------------
        self.post_size = len(self.post_vocab)
        self.deprel_size = len(self.deprel_vocab)
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        self.postag_size = len(self.postag_vocab)
        # self.synpost_size = len(self.synpost_vocab)
        #-------------------------------------------------------------------------------

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

    def load_dataset(self):
        train_file_name = os.path.join(self.data_dir, 'train.json')
        dev_file_name = os.path.join(self.data_dir, 'dev.json')
        test_file_name = os.path.join(self.data_dir, 'test.json')

        if not os.path.exists(dev_file_name):
            dev_file_name = test_file_name

        train_examples = [Example(data, self.max_seq_length) for data in load_json(train_file_name)]
        dev_examples = [Example(data, self.max_seq_length) for data in load_json(dev_file_name)]
        test_examples = [Example(data, self.max_seq_length) for data in load_json(test_file_name)]

        self.raw_datasets = {
            'train': train_examples,
            'dev': dev_examples,
            'test': test_examples
        }

    def get_dataloader(self, mode, batch_size, shuffle):
        dataloader = DataLoader(
            dataset=self.raw_datasets[mode],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=DataCollatorForASTE(tokenizer=self.tokenizer,
                                           max_seq_length=self.max_seq_length,
                                           post_vocab=self.post_vocab,
                                           deprel_vocab=self.deprel_vocab,
                                           # ---------------------------------
                                           # postag_vocab=self.postag_vocab,
                                           # synpost_vocab=self.synpost_vocab
                                           # ---------------------------------
                                           ),
            pin_memory=True,
            prefetch_factor=16
        )
        return dataloader


    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.eval_batch_size, shuffle=False)