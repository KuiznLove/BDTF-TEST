import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from .table import TableEncoder
from .matching_layer import MatchingLayer
from torch.nn import functional as F
from .TypeEnhancedGCN import TypeEnhancedGCN, TypeEnhancedGCNDeepseek
from .attention import SelfAttention, AspectAwareAttention, AspectAwareAttention2
from .fusion import InteractiveAttention, InteractiveAttentionLayer
from .bilstm import DualEncoderWithBiLSTM


class BDTFModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        # ----------------------------------------------------
        # 新增Bi-LSTM
        # self.bilstm = DualEncoderWithBiLSTM()
        # self.encode_fusion = nn.Linear(config.hidden_size*2, config.hidden_size)
        # ----------------------------------------------------
        self.table_encoder = TableEncoder(config)
        self.inference = InferenceLayer(config)
        self.matching = MatchingLayer(config)
        self.init_weights()
        # ----------------------------------------------------
        # 新增方面感知注意力层
        # self.aspect_aware_attention = AspectAwareAttention(config.hidden_size, num_heads=4)

        self.self_attention_out = SelfAttention(config.hidden_size, 256)
        self.aspect_aware_out = AspectAwareAttention(config.hidden_size, 256)
        self.opinion_aware_out = AspectAwareAttention(config.hidden_size, 256)

        # self.GCN = BatchGCNLayer(config.hidden_size, config.hidden_size)
        # self.dense = nn.Sequential(nn.Linear(config.hidden_size*4, config.hidden_size), nn.ReLU())

        self.Wo = nn.Linear(config.hidden_size, 256)
        # ----------------------------------------------------
        # ----------------------------------------------------
        # 新增依赖类型增强的图卷积网络
        # self.TypeEnhancedGCN = TypeEnhancedGCN(config.hidden_size, dep_type_dim=64, out_dim=192, num_layers=1)
        self.TypeEnhancedGCN = TypeEnhancedGCNDeepseek(config.hidden_size, 768)
        # self.TypeEnhancedGCN = EnhancedTypeGCN(config.hidden_size, 768)
        # ----------------------------------------------------
        # self.fusion = InteractiveAttention(hidden_dim=768, num_heads=12, dropout=0.3) #14lap, 16res
        self.fusion = InteractiveAttentionLayer()

    def forward(self, input_ids, attention_mask, ids, 
                # start_label_masks, end_label_masks,
                t_start_labels=None, t_end_labels=None,
                o_start_labels=None, o_end_labels=None,
                table_labels_S=None,   table_labels_E=None,
                polarity_labels=None, pairs_true=None,
                word_pair_position=None, word_pair_deprel=None,
                # word_pair_pos=None, word_pair_synpost=None,
                word_pair_pos=None,
                contrast_mask=None, sentence=None,
                aspect_mask=None, opinion_mask=None):

        seq = self.bert(input_ids, attention_mask)[0]
        # lstm_out = self.bilstm(seq)
        # seq = torch.cat((seq, lstm_out), dim=-1)
        # seq = self.encode_fusion(seq)
        # ----------------------------------------------------
        # 新增方面感知注意力层
        # seq_aspect_aware_attention = self.aspect_aware_attention(seq, aspect_mask)
        # seq_aspect_aware_attention = self.GCN(seq, seq_aspect_aware_attention)
        # seq = torch.cat((seq, seq_aspect_aware_attention), dim=-1)
        # seq = self.dense(seq)

        # seq = seq + seq_aspect_aware_attention
        # ----------------------------------------------------
        # ----------------------------------------------------
        # 自注意力+方面+观点
        seq_self_attention = self.self_attention_out(seq)
        seq_aspect_aware = self.aspect_aware_out(seq, aspect_mask)
        seq_opinion_aware = self.opinion_aware_out(seq, opinion_mask)
        seq_sem = torch.cat((seq_self_attention, seq_aspect_aware, seq_opinion_aware), dim=-1)
        # ----------------------------------------------------
        # ----------------------------------------------------
        # 新增依赖类型增强的图卷积网络
        # seq_deprel = self.TypeEnhancedGCN(seq, word_pair_deprel)
        seq_deprel = self.TypeEnhancedGCN(seq, word_pair_deprel, word_pair_pos, aspect_mask)
        # seq = self.Wo(seq) + seq_deprel
        seq_syn = seq + seq_deprel #没有方面感知

        seq = self.fusion(seq_sem, seq_syn)
        # ----------------------------------------------------
        table = self.table_encoder(seq, attention_mask)

        output = self.inference(table, attention_mask, table_labels_S, table_labels_E)

        output['ids'] = ids

        output = self.matching(output, table, pairs_true, seq)
        return output

    def process_input(self, input_ids, attention_mask, aspect_mask, opinion_mask, table_labels_S, table_labels_E):
        batch_size, seq_len = input_ids.shape
        input_ids = input_ids[:, :-11]
        attention_mask = attention_mask[:, 11:]
        aspect_mask = aspect_mask[:, :-11]
        opinion_mask = opinion_mask[:, :-11]
        table_labels_S = table_labels_S[:, :-11]
        table_labels_E = table_labels_E[:, :-11]
        input_ids = input_ids * attention_mask
        for i in range(batch_size):
            index = attention_mask[i].sum()
            input_ids[i][index - 1] = 102
        return input_ids, attention_mask, aspect_mask, opinion_mask, table_labels_S, table_labels_E

class InferenceLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.cls_linear_S = nn.Linear(768,1)
        self.cls_linear_E = nn.Linear(768,1)

    def span_pruning(self, pred, z, attention_mask):
        mask_length = attention_mask.sum(dim=1)-2
        length = ((attention_mask.sum(dim=1)-2)*z).long()
        length[length<5] = 5
        max_length = mask_length**2
        for i in range(length.shape[0]):
            if length[i] > max_length[i]:
                length[i] = max_length[i]
        batch_size = attention_mask.shape[0]
        pred_sort,_ = pred.view(batch_size, -1).sort(descending=True)
        batchs = torch.arange(batch_size).to('cuda')
        topkth = pred_sort[batchs, length-1].unsqueeze(1)        
        return pred >= (topkth.view(batch_size,1,1))

    def forward(self, table, attention_mask, table_labels_S, table_labels_E):      
        outputs = {}
        
        logits_S = torch.squeeze(self.cls_linear_S(table), 3)
        logits_E = torch.squeeze(self.cls_linear_E(table), 3)
        

        loss_func = nn.BCEWithLogitsLoss(weight=(table_labels_S>=0))
        
        outputs['table_loss_S'] = loss_func(logits_S, table_labels_S.float())
        outputs['table_loss_E'] = loss_func(logits_E, table_labels_E.float())

        S_pred = torch.sigmoid(logits_S) * (table_labels_S>=0)
        E_pred = torch.sigmoid(logits_E) * (table_labels_S>=0)

        if self.config.span_pruning != 0:
            table_predict_S = self.span_pruning(S_pred, self.config.span_pruning, attention_mask)
            table_predict_E = self.span_pruning(E_pred, self.config.span_pruning, attention_mask) 
        else:
            table_predict_S = S_pred>0.5
            table_predict_E = E_pred>0.5
        outputs['table_predict_S'] = table_predict_S
        outputs['table_predict_E'] = table_predict_E
        outputs['table_labels_S'] = table_labels_S
        outputs['table_labels_E'] = table_labels_E
        return outputs