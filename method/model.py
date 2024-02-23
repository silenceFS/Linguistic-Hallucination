import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from method.model_components import BertAttention, LinearLayer, BertSelfAttention, TrainablePositionalEncoding
from method.model_components import clip_nce, frame_nce
from method.model_components import PositionalEmbedding
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")



class MS_SL_Net(nn.Module):
    def __init__(self, config):
        super(MS_SL_Net, self).__init__()
        self.config = config

        self.hallucination_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.query_input_size, dropout=config.input_drop)

        self.hallucination_encoder = BertAttention(edict(hidden_size=config.query_input_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))
        self.hallucination_fc = LinearLayer(config.query_input_size, 30522, layer_norm=True,
                                            dropout=config.input_drop, relu=True)

        self.hallucination_positional_encoding = PositionalEmbedding(config.query_input_size, dropout=config.drop, maxlen=config.max_desc_l)
        self.hallucination_tgt_to_emb = nn.Embedding(30522,config.query_input_size,padding_idx = 0)

        decoder_layer = nn.TransformerDecoderLayer(config.query_input_size, config.n_heads, dropout = config.drop,batch_first=True)
        self.hallucination_decoder = nn.TransformerDecoder(decoder_layer, config.num_layers, nn.LayerNorm(config.query_input_size))

        self.generate_query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l * 2,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)

        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l * (config.support_set_number + 1),
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)

        self.generate_clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)
        self.generate_frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)
        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)
        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)
        #
        self.generate_query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        #
        self.generate_query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))
        self.query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))
        self.generate_clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.generate_clip_encoder = copy.deepcopy(self.query_encoder)
        self.generate_frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                             dropout=config.input_drop, relu=True)
        self.generate_frame_encoder = copy.deepcopy(self.query_encoder)
        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.clip_encoder = copy.deepcopy(self.query_encoder)
        self.frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                             dropout=config.input_drop, relu=True)
        self.frame_encoder = copy.deepcopy(self.query_encoder)
        self.generate_modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)
        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)
        self.pool_layers = nn.ModuleList([nn.Identity()]
                                         + [nn.AvgPool1d(i, stride=1) for i in range(2, config.map_size + 1)]
                                         )
        self.generate_mapping_linear = nn.ModuleList([nn.Linear(config.hidden_size, out_features=config.hidden_size, bias=False)
                                             for i in range(2)])
        self.mapping_linear = nn.ModuleList([nn.Linear(config.hidden_size, out_features=config.hidden_size, bias=False)
                                             for i in range(2)])
        self.clip_nce_criterion = clip_nce(reduction='mean')
        self.video_nce_criterion = frame_nce(reduction='mean')
        self.huber_loss = nn.SmoothL1Loss(reduction='mean')
        self.consistency_loss = nn.MSELoss(reduction='mean')
        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def generate_state_dict(self):
        state_dict = []
        state_dict.append(self.hallucination_pos_embed.state_dict())
        state_dict.append(self.hallucination_encoder.state_dict())
        state_dict.append(self.generate_query_pos_embed.state_dict())
        state_dict.append(self.generate_clip_pos_embed.state_dict())
        state_dict.append(self.generate_frame_pos_embed.state_dict())
        state_dict.append(self.generate_query_input_proj.state_dict())
        state_dict.append(self.generate_query_encoder.state_dict())
        state_dict.append(self.generate_clip_input_proj.state_dict())
        state_dict.append(self.generate_clip_encoder.state_dict())
        state_dict.append(self.generate_frame_input_proj.state_dict())
        state_dict.append(self.generate_frame_encoder.state_dict())
        state_dict.append(self.generate_modular_vector_mapping.state_dict())
        state_dict.append(self.generate_mapping_linear.state_dict())
        state_dict.append(self.hallucination_decoder.state_dict())
        state_dict.append(self.hallucination_fc.state_dict())
        return state_dict
    
    def load_generate_state_dict(self, state_dict):
        self.hallucination_pos_embed.load_state_dict(state_dict[0])
        self.hallucination_encoder.load_state_dict(state_dict[1])
        self.generate_query_pos_embed.load_state_dict(state_dict[2])
        self.generate_clip_pos_embed.load_state_dict(state_dict[3])
        self.generate_frame_pos_embed.load_state_dict(state_dict[4])
        self.generate_query_input_proj.load_state_dict(state_dict[5])
        self.generate_query_encoder.load_state_dict(state_dict[6])
        self.generate_clip_input_proj.load_state_dict(state_dict[7])
        self.generate_clip_encoder.load_state_dict(state_dict[8])
        self.generate_frame_input_proj.load_state_dict(state_dict[9])
        self.generate_frame_encoder.load_state_dict(state_dict[10])
        self.generate_modular_vector_mapping.load_state_dict(state_dict[11])
        self.generate_mapping_linear.load_state_dict(state_dict[12])
        if len(state_dict) > 13:
            self.hallucination_decoder.load_state_dict(state_dict[13])
            self.hallucination_fc.load_state_dict(state_dict[14])
        

    def forward(self, clip_video_feat, frame_video_feat, frame_video_mask, query_feat, query_mask, query_labels, support_query_feat, support_query_mask, 
                support_mask, stage,support_set_tokens,support_set_tokens_mask,support_set_tokens_padding_mask):

        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)
        encoded_frame_feat, vid_proposal_feat = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask)
        cat_query_feat = torch.cat((query_feat, support_query_feat.view(support_query_feat.size(0),-1,support_query_feat.size(3))), dim=1)
        cat_query_mask = torch.cat((query_mask, support_query_mask.view(support_query_mask.size(0),-1)), dim=1)
        cat_query = self.encode_query(cat_query_feat, cat_query_mask)
        clip_scale_scores, frame_scale_scores, clip_scale_scores_, frame_scale_scores_ \
            = self.get_pred_from_raw_query(
            cat_query, query_labels, vid_proposal_feat, encoded_frame_feat, frame_video_mask, cross=False,
            return_query_feats=True)
        clip_nce_loss = 0.02 * self.clip_nce_criterion(query_labels, label_dict, clip_scale_scores_)
        clip_trip_loss = self.get_clip_triplet_loss(clip_scale_scores, query_labels)
        frame_nce_loss = 0.04 * self.video_nce_criterion(frame_scale_scores_)
        frame_trip_loss = self.get_frame_trip_loss(frame_scale_scores)
        loss = clip_nce_loss + clip_trip_loss + frame_nce_loss + frame_trip_loss
        if stage == 'support':
            return {"loss_overall": loss, 'clip_nce_loss': clip_nce_loss,
                'clip_trip_loss': clip_trip_loss,
                'frame_nce_loss': frame_nce_loss, 'frame_trip_loss': frame_trip_loss
                } 

        generate_query_feat = self.hallucination_encoder(self.hallucination_pos_embed(query_feat), query_mask.unsqueeze(1) if query_mask is not None else query_mask)
        generate_query_mask = torch.ones(query_mask.size(0), query_mask.size(1)).to(query_mask.device)
        rselect_indexes = support_set_tokens_mask[:,0].type(torch.bool) # batch_size 
        select_indexes = rselect_indexes.nonzero().squeeze()
        target_tokens = support_set_tokens[select_indexes,0].long()
        input_generate_query_feat = generate_query_feat[select_indexes]

        input_support_query_feat = support_query_feat[select_indexes,0]
        input_support_tokens_padding_mask = support_set_tokens_padding_mask[select_indexes,0]
        hallucination_loss = self.hallucination_loss_visual(input_generate_query_feat,target_tokens,input_support_tokens_padding_mask,False)
        generate_query_feat = torch.cat((query_feat, generate_query_feat), dim=1)
        generate_query_mask = torch.cat((query_mask, generate_query_mask), dim=1)
        generate_query = self.encode_query_generate(generate_query_feat, generate_query_mask)
        generate_encoded_frame_feat, generate_vid_proposal_feat = self.encode_context_generate(
        clip_video_feat, frame_video_feat, frame_video_mask)
        generate_clip_scale_scores, generate_frame_scale_scores, generate_clip_scale_scores_, generate_frame_scale_scores_ \
            = self.get_pred_from_raw_query_generate(
            generate_query, query_labels, generate_vid_proposal_feat, generate_encoded_frame_feat, frame_video_mask, cross=False,
            return_query_feats=True)
        generate_clip_nce_loss = 0.02 * self.clip_nce_criterion(query_labels, label_dict, generate_clip_scale_scores_)
        generate_clip_trip_loss = self.get_clip_triplet_loss(generate_clip_scale_scores, query_labels)
        generate_frame_nce_loss = 0.04 * self.video_nce_criterion(generate_frame_scale_scores_)
        generate_frame_trip_loss = self.get_frame_trip_loss(generate_frame_scale_scores)
        generate_loss = generate_clip_nce_loss + generate_clip_trip_loss + generate_frame_nce_loss + generate_frame_trip_loss
        query_consistency_loss = self.consistency_loss(generate_query, cat_query)
        video_consistency_loss = self.consistency_loss(generate_vid_proposal_feat, vid_proposal_feat) + self.consistency_loss(generate_encoded_frame_feat, encoded_frame_feat)

        similarity_loss = self.huber_loss(generate_clip_scale_scores, clip_scale_scores) + self.huber_loss(generate_frame_scale_scores, frame_scale_scores)
        return {"loss_overall": loss, 'clip_nce_loss': clip_nce_loss,
                'clip_trip_loss': clip_trip_loss,
                'frame_nce_loss': frame_nce_loss, 'frame_trip_loss': frame_trip_loss,
                "generate_loss_overall": generate_loss, 'generate_clip_nce_loss': generate_clip_nce_loss,
                'generate_clip_trip_loss': generate_clip_trip_loss,
                'generate_frame_nce_loss': generate_frame_nce_loss, 'generate_frame_trip_loss': generate_frame_trip_loss,
                'query_consistency_loss': query_consistency_loss, 'video_consistency_loss': video_consistency_loss,
                'similarity_loss': similarity_loss,'hallucination_loss':hallucination_loss
                }

    def get_hallucination_loss_fn(self,input_vocab,target_tokens):
        temp_loss_fn =  nn.CrossEntropyLoss(ignore_index = 0)
        return temp_loss_fn(input_vocab,target_tokens)
    def hallucination_loss_visual(self,memories,tgt,tgt_padding_mask,need_visual):
        tgt_input = tgt[:, :-1]  # N  L-1
        tgt_out = tgt[:, 1:]  # N L-1
        tgt_input_padding_mask = tgt_padding_mask[:, :-1]
        tgt_mask = generate_square_subsequent_mask(tgt_input.shape[1]).to(tgt_input.device)
        tgt_emb = self.hallucination_positional_encoding(self.hallucination_tgt_to_emb(tgt_input)) # N x L x D
        #import ipdb;ipdb.set_trace()
        outs = self.hallucination_decoder(
            tgt_emb, memories,
            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_input_padding_mask,
        )
        if type(outs) is tuple:
            outs, self.attn_weights = outs
        logits = self.hallucination_fc(outs)
        loss = self.get_hallucination_loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            tgt_out.reshape(-1)
        )
        
        if not need_visual:
            return loss
        
        batch_size = memories.shape[0]
        start_id = 101
        end_id = 102
        # predict
        ys = torch.ones(batch_size, 1).fill_(start_id).type(torch.long).to(tgt.device)  # N, 1
        end_flag = [0] * batch_size
        for i in range(self.config.max_desc_l - 1):
            tgt_emb = self.hallucination_positional_encoding(self.hallucination_tgt_to_emb(ys))
            tgt_mask = generate_square_subsequent_mask(tgt_emb.shape[1]).to(torch.bool).to(tgt_emb.device)
            outs = self.hallucination_decoder(
                tgt_emb, memories,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                )
            logits = self.hallucination_fc(outs[:, -1])
            _, next_word = torch.max(logits, dim=1)  # N
            ys = torch.cat([ys, next_word.unsqueeze(1).type(torch.long)], dim=1)  # N, t
            for k, flag in enumerate((next_word == end_id).tolist()):
                if flag is True:
                    end_flag[k] = 1
            if sum(end_flag) >= batch_size:
                break
        # to string
        result = []
        for idx_cap in ys.tolist(): # N x L
            end_count = -1
            for i, idx in enumerate(idx_cap):
                if idx == end_id:
                    end_count = i
                    break
            idx_cap = idx_cap[1:end_count]
            token_cap = tokenizer.convert_ids_to_tokens(idx_cap)
            result.append(tokenizer.convert_tokens_to_string(token_cap))
        return loss

    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        video_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 1
        return video_query

    def encode_query_generate(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.generate_query_input_proj, self.generate_query_encoder,
                                          self.generate_query_pos_embed)  # (N, Lq, D)
        video_query = self.get_modularized_queries_generate(encoded_query, query_mask)  # (N, D) * 1
        return video_query

    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None):
        encoded_clip_feat = self.encode_input(clip_video_feat, None, self.clip_input_proj, self.clip_encoder,
                                               self.clip_pos_embed)
        encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj,
                                                self.frame_encoder,
                                                self.frame_pos_embed)
        vid_proposal_feat_map = self.encode_feat_map(encoded_clip_feat)
        return encoded_frame_feat, \
               vid_proposal_feat_map

    def encode_context_generate(self, clip_video_feat, frame_video_feat, video_mask=None):


        encoded_clip_feat = self.encode_input(clip_video_feat, None, self.generate_clip_input_proj, self.generate_clip_encoder,
                                               self.generate_clip_pos_embed)

        encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.generate_frame_input_proj,
                                                self.generate_frame_encoder,
                                                self.generate_frame_pos_embed)

        vid_proposal_feat_map = self.encode_feat_map(encoded_clip_feat)


        return encoded_frame_feat, \
               vid_proposal_feat_map


    def encode_feat_map(self, x_feat):

        pool_in = x_feat.permute(0, 2, 1)

        proposal_feat_map = []
        for idx, pool in enumerate(self.pool_layers):
            x = pool(pool_in).permute(0, 2, 1)
            proposal_feat_map.append(x)
        proposal_feat_map = torch.cat(proposal_feat_map, dim=1)


        return proposal_feat_map


    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        return encoder_layer(feat, mask)  # (N, L, D_hidden)

    def get_modularized_queries(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()

    def get_modularized_queries_generate(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.generate_modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()

    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                  dim=1)  # (N, N) diagonal positions are positive pairs
        return query_context_scores, indices



    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):

        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        query_context_scores, _ = torch.max(query_context_scores, dim=1)

        return query_context_scores

    def key_clip_guided_attention(self, frame_feat, proposal_feat, feat_mask, max_index, query_labels):
        selected_max_index = max_index[[i for i in range(max_index.shape[0])], query_labels]

        expand_frame_feat = frame_feat[query_labels]

        expand_proposal_feat = proposal_feat[query_labels]

        key = self.mapping_linear[0](expand_frame_feat)
        query = expand_proposal_feat[[i for i in range(key.shape[0])], selected_max_index, :].unsqueeze(-1)
        value = self.mapping_linear[1](expand_frame_feat)

        if feat_mask is not None:
            expand_feat_mask = feat_mask[query_labels]
            scores = torch.bmm(key, query).squeeze()
            masked_scores = scores.masked_fill(expand_feat_mask.eq(0), -1e9).unsqueeze(1)
            masked_scores = nn.Softmax(dim=-1)(masked_scores)
            attention_feat = torch.bmm(masked_scores, value).squeeze()
        else:
            scores = nn.Softmax(dim=-1)(torch.bmm(key, query).transpose(1, 2))
            attention_feat = torch.bmm(scores, value).squeeze()

        return attention_feat

    def key_clip_guided_attention_generate(self, frame_feat, proposal_feat, feat_mask, max_index, query_labels):
        selected_max_index = max_index[[i for i in range(max_index.shape[0])], query_labels]

        expand_frame_feat = frame_feat[query_labels]

        expand_proposal_feat = proposal_feat[query_labels]

        key = self.generate_mapping_linear[0](expand_frame_feat)
        query = expand_proposal_feat[[i for i in range(key.shape[0])], selected_max_index, :].unsqueeze(-1)
        value = self.generate_mapping_linear[1](expand_frame_feat)

        if feat_mask is not None:
            expand_feat_mask = feat_mask[query_labels]
            scores = torch.bmm(key, query).squeeze()
            masked_scores = scores.masked_fill(expand_feat_mask.eq(0), -1e9).unsqueeze(1)
            masked_scores = nn.Softmax(dim=-1)(masked_scores)
            attention_feat = torch.bmm(masked_scores, value).squeeze()
        else:
            scores = nn.Softmax(dim=-1)(torch.bmm(key, query).transpose(1, 2))
            attention_feat = torch.bmm(scores, value).squeeze()

        return attention_feat

    def key_clip_guided_attention_in_inference(self, frame_feat, proposal_feat, feat_mask, max_index):
        key = self.mapping_linear[0](frame_feat)
        value = self.mapping_linear[1](frame_feat)
        num_vid = frame_feat.shape[0]

        index = torch.arange(num_vid).unsqueeze(1)
        query = proposal_feat[index, max_index.t()]
        if feat_mask is not None:
            scores = torch.bmm(key, query.transpose(2, 1))
            masked_scores = scores.masked_fill(feat_mask.unsqueeze(-1).eq(0), -1e9)
            masked_scores = nn.Softmax(dim=1)(masked_scores)
            attention_feat = torch.bmm(masked_scores.transpose(1, 2), value)
        else:
            scores = torch.bmm(key, query.transpose(2, 1))
            scores = nn.Softmax(dim=1)(scores)
            attention_feat = torch.bmm(scores.transpose(1, 2), value)

        return attention_feat

    def key_clip_guided_attention_in_inference_generate(self, frame_feat, proposal_feat, feat_mask, max_index):
        key = self.generate_mapping_linear[0](frame_feat)
        value = self.generate_mapping_linear[1](frame_feat)
        num_vid = frame_feat.shape[0]

        index = torch.arange(num_vid).unsqueeze(1)
        query = proposal_feat[index, max_index.t()]
        if feat_mask is not None:
            scores = torch.bmm(key, query.transpose(2, 1))
            masked_scores = scores.masked_fill(feat_mask.unsqueeze(-1).eq(0), -1e9)
            masked_scores = nn.Softmax(dim=1)(masked_scores)
            attention_feat = torch.bmm(masked_scores.transpose(1, 2), value)
        else:
            scores = torch.bmm(key, query.transpose(2, 1))
            scores = nn.Softmax(dim=1)(scores)
            attention_feat = torch.bmm(scores.transpose(1, 2), value)

        return attention_feat

    def get_pred_from_raw_query(self, video_query, query_labels=None,
                                video_proposal_feat=None,
                                video_feat=None,
                                video_feat_mask=None,
                                cross=False,
                                return_query_feats=False):


        # get clip-level retrieval scores

        clip_scale_scores, key_clip_indices = self.get_clip_scale_scores(
            video_query, video_proposal_feat)

        if return_query_feats:
            frame_scale_feat = self.key_clip_guided_attention(video_feat, video_proposal_feat, video_feat_mask,
                                                          key_clip_indices, query_labels)
            frame_scale_scores = torch.matmul(F.normalize(video_query, dim=-1),
                                              F.normalize(frame_scale_feat, dim=-1).t())
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            frame_scale_scores_ = torch.matmul(video_query, frame_scale_feat.t())

            return clip_scale_scores, frame_scale_scores, clip_scale_scores_,frame_scale_scores_
        else:
            frame_scale_feat = self.key_clip_guided_attention_in_inference(video_feat, video_proposal_feat, video_feat_mask,
                                                                       key_clip_indices).to(video_query.device)
            frame_scales_cores_ = torch.mul(F.normalize(frame_scale_feat, dim=-1),
                                            F.normalize(video_query, dim=-1).unsqueeze(0))
            frame_scale_scores = torch.sum(frame_scales_cores_, dim=-1).transpose(1, 0)

            return clip_scale_scores, frame_scale_scores

    def get_pred_from_raw_query_generate(self, video_query, query_labels=None,
                                video_proposal_feat=None,
                                video_feat=None,
                                video_feat_mask=None,
                                cross=False,
                                return_query_feats=False):


        # get clip-level retrieval scores

        clip_scale_scores, key_clip_indices = self.get_clip_scale_scores(
            video_query, video_proposal_feat)

        if return_query_feats:
            frame_scale_feat = self.key_clip_guided_attention_generate(video_feat, video_proposal_feat, video_feat_mask,
                                                          key_clip_indices, query_labels)
            frame_scale_scores = torch.matmul(F.normalize(video_query, dim=-1),
                                              F.normalize(frame_scale_feat, dim=-1).t())
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            frame_scale_scores_ = torch.matmul(video_query, frame_scale_feat.t())

            return clip_scale_scores, frame_scale_scores, clip_scale_scores_,frame_scale_scores_
        else:
            frame_scale_feat = self.key_clip_guided_attention_in_inference_generate(video_feat, video_proposal_feat, video_feat_mask,
                                                                       key_clip_indices).to(video_query.device)
            frame_scales_cores_ = torch.mul(F.normalize(frame_scale_feat, dim=-1),
                                            F.normalize(video_query, dim=-1).unsqueeze(0))
            frame_scale_scores = torch.sum(frame_scales_cores_, dim=-1).transpose(1, 0)

            return clip_scale_scores, frame_scale_scores

    def get_clip_triplet_loss(self, query_context_scores, labels):
        v2t_scores = query_context_scores.t()
        t2v_scores = query_context_scores
        labels = np.array(labels)

        # cal_v2t_loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])


            neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)
            if self.config.use_hard_negative:
                sample_neg_pair_scores = neg_pair_scores[0]
            else:
                v2t_sample_max_idx = neg_pair_scores.shape[0]
                sample_neg_pair_scores = neg_pair_scores[
                    torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]



            v2t_loss += (self.config.margin + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        # cal_t2v_loss
        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
        t2v_pos_scores = t2v_scores[text_indices, labels]
        mask_score = copy.deepcopy(t2v_scores.data)
        mask_score[text_indices, labels] = 999
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)
        t2v_sample_max_idx = min(1 + self.config.hard_pool_size,
                                 t2v_scores.shape[1]) if self.config.use_hard_negative else t2v_scores.shape[1]
        sample_indices = sorted_scores_indices[
            text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]

        t2v_neg_scores = t2v_scores[text_indices, sample_indices]

        t2v_loss = (self.config.margin + t2v_neg_scores - t2v_pos_scores).clamp(min=0)

        return t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)

    def get_frame_trip_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """

        bsz = len(query_context_scores)

        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx + loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """

        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)

        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)

        sample_min_idx = 1  # skip the masked positive

        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) if self.config.use_hard_negative else bsz

        # sample_max_idx = 2

        # (N, )
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                       size=(bsz,)).to(scores.device)]

        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)



def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask