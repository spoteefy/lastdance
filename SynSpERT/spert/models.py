from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from spert import sampling
from spert import util
import sys

import math
from typing import Optional, List
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

# CÁC THÔNG SỐ
# h: vector của câu đầu vào sinh ra từ BERT/SciBERT

# Nếu có một lượng E token trong một câu và mỗi token được chuyển đổi 
# thành một vector có kích thước S (= emb_size = 768), thì kích thước tensor h = BatchSize x E x S.

# x: Các định danh số (ID) của các mã hóa nhận được sau khi tokenization cho toàn bộ lô; kích thước = BatchSize x E. 
# Mỗi slot là một tokenID (số nguyên).

# token: Định danh số (ID) của token mong muốn, ví dụ, 'CLS'
# Ta thấy tensor 'h' chứa nhúng BERT/SciBERT của mỗi token trong câu, do đó
# nhảy đến chỉ mục (index) của h[token]
# h.shape= (10, 40, 768) => 10 câu mỗi câu có độ dài 40 token. [(Sentence embedding)+]
# x.shape= (10, 40) => 10 câu mỗi câu có độ dài 40 token. [(Sentence embedding)+]

def get_token(h: torch.tensor, x: torch.tensor, token: int):  # token: Định danh số (ID) của token mong muốn, ví dụ, 'CLS'
    """ Lấy token embedding cụ thể (ví dụ: [CLS]) """

    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)  # Sẽ có các cột emb_size. Số hàng = số token
                                    # -1 có nghĩa là kích thước thực sự sẽ được suy luận.
                                    # Do đó, h.shape=(10*40, 768)
                                    
    flat = x.contiguous().view(-1)  # Làm phẳng encoding, hay, 
                                    # flat = vector của 10*40 token ids

    # Lấy contextualized embedding của token nhận được

    token_h = token_h[flat == token, :]     # Kích thước token_h = (10, 768) nếu mỗi câu
                                            # chỉ chứa token một lần.
                                            # Nếu token xuất hiện K lần trong tất cả các câu,
                                            # thì kích thước token_h = (K, 768)
    
    return token_h

class PrepareForMultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the heads.
        head_shape = x.shape[:-1]

        # Linear transform
        x = self.linear(x)

        # Split last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, heads, d_model]`
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):

        super().__init__()

        # Number of features per head
        self.d_k = d_model // heads
        # Number of heads
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # Softmax for attention along the time dimension of `key`
        self.softmax = nn.Softmax(dim=1)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # Scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)

        # We store attentions so that it can be used for logging, or other computations if needed
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):

        # Calculate $Q K^\top$ or $S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}$
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):

        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        # Same mask applied to all heads.
        mask = mask.unsqueeze(-1)

        # resulting mask has shape `[seq_len_q, seq_len_k, batch_size, heads]`
        return mask

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[seq_len, batch_size, heads, d_k]`.
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Compute attention scores $Q K^\top$.
        # This gives a tensor of shape `[seq_len, seq_len, batch_size, heads]`.
        scores = self.get_scores(query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = self.softmax(scores)

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        # Save attentions for any other calculations 
        self.attn = attn.detach()

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        return self.output(x)

class Highway(nn.Module):
    r"""Highway Layers
    Args:
        - num_highway_layers(int): number of highway layers.
        - input_size(int): size of highway input.
    """

    def __init__(self, num_highway_layers, input_size):
        super(Highway, self).__init__()
        self.num_highway_layers = num_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        for layer in range(self.num_highway_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            # Compute percentage of non linear information to be allowed for each element in x
            non_linear = F.relu(self.non_linear[layer](x))
            # Compute non linear information
            linear = self.linear[layer](x)
            # Compute linear information
            x = gate * non_linear + (1 - gate) * linear
            # Combine non linear and linear information according to gate
            x = self.dropout(x)
        return x

class FusionGate(nn.Module):

    def __init__(self, num_fusion_layers, input_size):
        super(FusionGate, self).__init__()
        self.num_fusion_layers = num_fusion_layers
        self.f_gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_fusion_layers)])
        self.f_dropout = nn.Dropout(0.1)

    def forward(self, x, y):
        for layer in range(self.num_fusion_layers):
            gate = torch.sigmoid(self.f_gate[layer](y))
            u = gate * y + (1 - gate) * x
            # Combine non linear and linear information according to gate
            x = self.f_dropout(u)
        return x

class SpERT(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    VERSION = '1.1'

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100,

                 # Cấu hình
                 use_pos: bool = False, # Không sử dụng thẻ POS trong mô hình, FALSE
                 use_entity_clf: str = "none"
                 ):

        super(SpERT, self).__init__(config)
        self._use_pos = True
        self._pos_embedding = 25  # Kích thước nhúng #POS (part-of-speech) phải phù hợp với kích thước
                                  # BERT embedding size, xác định bởi config.hidden_size
        self._use_entity_clf = use_entity_clf
        
        # Cú pháp cho mô hình BERT
        self.bert = BertModel(config)

        # Các tầng trong mô hình
        # entc_in_dim cho biết kích thước biểu diễn cho entity_classifier
        if (self._use_pos == False):
            entc_in_dim = (config.hidden_size) * 2 + size_embedding  # CLS + ENT + SIZE = 2H + SIZE
        else:
            # CLS + ENT + SIZE + POS =  H*2 + SIZE + POS
            entc_in_dim = (config.hidden_size) * 2  + size_embedding   + self._pos_embedding
           
        self.entity_classifier = nn.Linear( entc_in_dim, entity_types )

        # relc_in_dim cho biết kích thước biểu diễn cho rel_classifier
        # (ENT + SIZE)*2 + SPAN = 3H + SIZE

        # relc_in_dim =  (config.hidden_size ) * 4 + size_embedding * 2 
        # if (self._use_entity_clf != "none"): # Tăng cường biểu diễn cho cặp entity trong rel_classifier
        #     relc_in_dim +=  entity_types * 2 
        # if (self._use_pos): # Tăng cường biểu diễn khi sử dụng thẻ POS-tagging
        #     relc_in_dim +=  self._pos_embedding * 4
        
        relc_in_dim = 2148
        
        self.rel_classifier = nn.Linear(relc_in_dim, relation_types)
   
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.pos_embeddings = nn.Embedding(52, self._pos_embedding, padding_idx=0)
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

        embed_dim = 793
        hidden_dim = config.hidden_size
        proj_dim = 512
        #dropout_rate = 0.1

        self.projection_entity = nn.Sequential(
            nn.Linear(1586, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            #nn.Dropout(dropout_rate),
            )

        self.projection_context = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            #nn.Dropout(dropout_rate),
            )
        self.projection_full = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            #nn.Dropout(dropout_rate),
            )
        

        self.projection_repr = nn.Sequential(
            nn.Linear(1636, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            #nn.Dropout(dropout_rate),
            )

        self.highwaynet = Highway(num_highway_layers=2,
                                    input_size = 2379)

        self.fusion_gate = FusionGate(num_fusion_layers=2,
                                    input_size = 512)

        self.multihead_attn = MultiHeadAttention(heads= 8, d_model= 512)
        
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # Đóng băng tham số BERT (huấn luyện đóng băng)
            for param in self.bert.parameters():
                param.requires_grad = False
    

    def _run_entity_classifier (self, x: torch.tensor):
        y = self.entity_classifier(x)
        return y
    
    def _run_rel_classifier (self, x: torch.tensor):
        y = self.rel_classifier(x)
        return y
        
    def _classify_entities(self, encodings, h, entity_masks, size_embeddings, 
                           pos=None, hlarge=None):
        
        # max-pooling entity candidate spans
        if (hlarge == None):
            hlarge = h

        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)  #torch.Size([10, 105, 40, 1])

        entity_spans_pool = m + hlarge.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1) #torch.Size([10, 105, 40, 768])
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]
        
        # Lấy token 'CLS' làm đại diện ngữ cảnh ứng viên
        entity_ctx = get_token(h, encodings, self._cls_token)
        
        # Tạo các biểu diễn ứng viên bao gồm ngữ cảnh, max-pool span và size embedding
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)

        # Phân loại thực thể ứng viên
        entity_clf = self._run_entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool
    
    def _classify_relations(self, entity_spans, size_embeddings, 
                            relations, rel_masks, h, chunk_start,
                            entity_clf = None, hlarge1=None):
        batch_size = relations.shape[0]

        # Tạo chunks nếu cần thiết
        # max-pooling relation candidate spans
        if (hlarge1 == None):
            hlarge1 = h

        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            hlarge1 = hlarge1[:, :relations.shape[1], :]

        # Lấy biểu diễn cặp thực thể ứng viên
        entity_pairs = util.batch_index(entity_spans, relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # Lấy size embeddings tương ứng
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # Ngữ cảnh quan hệ (ngữ cảnh giữa hai thực thể ứng viên)
        # Đánh dấu non-entity candidate tokens
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx =  m + hlarge1

        rel_ctx = rel_ctx.max(dim=2)[0]
        full_ctx = hlarge1.max(dim=2)[0] # ngữ cảnh toàn cục
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0 # ngữ cảnh cục bộ giữa entity_pair

        rel_ctx_pro = self.projection_context(rel_ctx)
        full_ctx_pro = self.projection_context(full_ctx)
        entity_pairs_pro = self.projection_entity(entity_pairs)
        
        # Tạo các biểu diễn ứng viên mối quan hệ bao gồm ngữ cảnh, max-pooled cặp ứng viên thực thể  
        # và các size embedding tương ứng
        
        rel_repr = torch.cat([entity_pairs, size_pair_embeddings], dim=2)
        #rel_repr = self.projection_repr(rel_repr)
        #rel_repr2 = self.multihead_attn(query = entity_pairs_pro, key = rel_ctx_pro, value = rel_ctx_pro)
        rel_repr3 = self.multihead_attn(query = entity_pairs_pro, key = full_ctx_pro, value = full_ctx_pro)
        #rel_ctx = self.fusion_gate(x = rel_repr2, y = rel_repr3)
        rel_repr = torch.cat([rel_repr3, rel_repr], dim=2)
        
        rel_repr = self.dropout(rel_repr)
        chunk_rel_logits = self._run_rel_classifier(rel_repr)
        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
    # Lọc các span dựa trên việc phân loại các thực thể và tạo ra các mối quan hệ giữa chúng
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # Lấy kiểu thực thể (bao gồm none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # Lấy span đã phân loại là thực thể
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # Tạo quan hệ và che giấu (mask)
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # khả năng 1: không có nhiều hơn 2 spans được phân loại là thực thể 
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # khả năng 2: có nhiều hơn 2 spans được phân loại là thực thể
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # ngăn xếp (stack)
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)

##############################################################
""" 
- SynSpERTConfig là một lớp con của BertConfig, được sử dụng để tùy chỉnh cấu hình cho mô hình SynSpERT.
- SynSpERT là một lớp con của SpERT, và nó kế thừa từ SynSpERTConfig. 
Lớp này mở rộng SpERT bằng cách thêm một số tính năng mới như sử dụng POS tags, cùng với việc tăng cường biểu diễn thực thể (use_entity_clf). 
Các tham số khác cũng được truyền vào hàm khởi tạo của SpERT. 
"""

class SynSpERTConfig(BertConfig):
    def __init__(self, **kwargs):
        super(SynSpERTConfig, self).__init__(**kwargs)

class SynSpERT(SpERT):
    config_class = SynSpERTConfig
    VERSION = '1.0'

    def __init__(self, config: SynSpERTConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100, 
                 use_pos: bool = True,  
                 use_entity_clf: str = "none"                 
                 ):

        super(SynSpERT, self).__init__(config, cls_token, relation_types, entity_types,
                                       size_embedding, prop_drop, freeze_transformer, max_pairs,
                                       use_pos,
                                       use_entity_clf)

        self.config = config

        self.init_weights()

 
    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor, 
                       # Extra params follow
                       dephead: torch.tensor, deplabel: torch.tensor, pos: torch.tensor ):
        # Lấy contextualized token embeddings từ lớp transformer gần nhất
        context_masks = context_masks.float()
        # Sử dụng BERT để tính toán biểu diễn của văn bản
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        
        # Tính toán độ dài thực tế của mỗi câu trong batch 
        # thông qua số lượng token không phải padding
        seq_len = torch.count_nonzero(context_masks, dim=1)
        
        # Lấy thông tin về kích thước batch và số lượng token trong mỗi câu.
        batch_size = encodings.shape[0]
        token_len = h.shape[1]

        hlarge1 = None
        # Thêm pos embeddings vào các token
        if (self._use_pos):
           pos_em = self.pos_embeddings(pos).to(self.rel_classifier.weight.device)
           hlarge1 = h
           hlarge1 = torch.cat((hlarge1, pos_em), -1)

        # Phân loại entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, 
                                        entity_masks, size_embeddings, pos, hlarge1)

        # Phân loại relations
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        hlarge2 = None
        if (self._use_pos):
           hlarge2 = hlarge1.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # Tính logits của quan hệ
        # Chia thành cac chunks để giảm bộ nhớ sử dụng
        for i in range(0, relations.shape[1], self._max_pairs):
            # Phân loại quan hệ cho ứng viên
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i, 
                                                        entity_clf, hlarge2)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits

        return entity_clf, rel_clf

    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                      entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor, 
                      dephead: torch.tensor, deplabel: torch.tensor, pos: torch.tensor):
        # Lấy contextualized token embeddings từ lớp transformer gần nhất
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']

        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]
        
        seq_len = torch.count_nonzero(context_masks, dim=1)      

        hlarge1 = None
        if (self._use_pos):
           pos_em = self.pos_embeddings(pos).to(self.rel_classifier.weight.device)
           hlarge1 = h
           hlarge1 = torch.cat((hlarge1, pos_em), -1)
          
    
        # Phân loại entities
        size_embeddings = self.size_embeddings(entity_sizes)  # nhúng kích thước thực thể ứng viên
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, 
                                entity_masks, size_embeddings, pos, hlarge1)

        # Bỏ qua các ứng viên thực thể không tạo thành một thực thể thực sự cho mối 
        # quan hệ (dựa trên bộ phân loại)
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, 
                                                                    ctx_size)

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        hlarge2 = None
        if (self._use_pos):
            hlarge2 = hlarge1.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)

        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # Tính toán logits cho quan hệ
        # Xử lý thành các chunks để giảm bộ nhớ sử dụng
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, 
                                                        h_large, i, 
                                                        entity_clf, hlarge2)
            # áp dụng sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # che giấu (mask)

        # áp dụng softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations

# Model access

_MODELS = {
    'spert': SpERT,
    'syn_spert': SynSpERT,
}

def get_model(name):
    return _MODELS[name]
