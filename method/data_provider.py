import json
import torch
import torch.utils.data as data
import numpy as np
import re
import h5py
import random
import os
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import tqdm
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from transformers import AutoTokenizer


class CapPreprocessor:
    def __init__(self, tokenizer_type = 'bert-base-uncased', device=torch.device('cuda')):
        self.tokenizer_type = tokenizer_type
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        self.pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.start_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
        self.end_id = self.tokenizer.convert_tokens_to_ids("[SEP]")

    def __call__(self, captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Turn raw text captions to tensor by Hugging Face tokenizer
        text -> id -> batching -> masking
        :param captions: list of raw caption strings.
        :return: batched text tensor and mask tensor (True for valid position).
        """
        # 1-pass
        batch_size = len(captions)
        tokens = []
        for i in range(batch_size):
            tokens.append(self.tokenizer.encode(captions[i], return_tensors="pt").squeeze())
        # 2-pass
        text_len = [len(i) for i in tokens]
        max_len = max(text_len)
        text_ts = torch.ones([batch_size, max_len], dtype=torch.long)* self.pad_id
        for i in range(batch_size):
            text_ts[i, :len(tokens[i])] = tokens[i]
        text_mask_ts = (text_ts == self.pad_id)
        return text_ts, text_mask_ts



def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    return vid_id

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def average_to_fixed_length(visual_input, map_size):
    visual_input = torch.from_numpy(visual_input)
    num_sample_clips = map_size
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips

    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))

    new_visual_input = []

    for i in range(num_sample_clips):

        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0).numpy()


    return new_visual_input

def uniform_feature_sampling(features, max_len):
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_features = []
    for i in range(max_len):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_features.append(np.mean(features[s_idx:e_idx], axis=0))
        else:
            new_features.append(features[s_idx])
    new_features = np.asarray(new_features)
    return new_features


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)



def collate_train(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    clip_video_features, frame_video_features, captions, idxs, cap_ids, video_ids, support_captions,cap_tokens,all_support_cap_tokens,all_support_cap_tokens_padding_mask = zip(*data)

    #videos
    clip_videos = torch.cat(clip_video_features, dim=0).float()
    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    #captions
    feat_dim = captions[0][0].shape[-1]

    merge_captions = []
    all_lengths = []
    labels = []

    support_set = []
    support_numbers = []

    for index, caps in enumerate(captions):
        labels.extend(index for i in range(len(caps)))
        all_lengths.extend(len(cap) for cap in caps)
        merge_captions.extend(cap for cap in caps)
        for idx, cap in enumerate(caps):
            support_set.append(support_captions[index][idx])
            support_numbers.append(len(support_captions[index][idx]))
    tokens_length = []
    merge_tokens = []
    merge_support_set_tokens = []
    merge_support_set_tokens_padding_mask = []
    for index,single_video_cap_tokens in enumerate(cap_tokens):
        tokens_length.extend(len(single_token) for single_token in single_video_cap_tokens) # 
        merge_tokens.extend(single_token for single_token in single_video_cap_tokens)
        for idx, single_cap_token in enumerate(single_video_cap_tokens): # single_cap_token
            merge_support_set_tokens.append(all_support_cap_tokens[index][idx]) # 
            merge_support_set_tokens_padding_mask.append(all_support_cap_tokens_padding_mask[index][idx])
    target = torch.zeros(len(all_lengths), max(max(tokens_length),max(all_lengths)), feat_dim)
    words_mask = torch.zeros(len(all_lengths), max(max(tokens_length),max(all_lengths)))

    rs_tokens = torch.zeros(len(tokens_length),max(max(tokens_length),max(all_lengths)))
    rs_support_tokens = torch.zeros(len(tokens_length),max(support_numbers),max(max(tokens_length),max(all_lengths)))
    tokens_mask = torch.zeros(len(tokens_length),max(support_numbers))
    for index,single_tokens in enumerate(merge_tokens):
        rs_tokens[index,:len(single_tokens)]  = single_tokens
        for idx in range(len(merge_support_set_tokens[index])):
            temp_support_set_tokens = merge_support_set_tokens[index][idx]
            rs_support_tokens[index,idx,:len(temp_support_set_tokens)] = temp_support_set_tokens
            tokens_mask[index,idx] = 1
    support_tokens_padding_mask = (rs_support_tokens == 0) # N x ns x L
    support_mask = torch.zeros(len(all_lengths), max(support_numbers))
    support_target = torch.zeros(len(all_lengths), max(support_numbers), max(max(tokens_length),max(all_lengths)), feat_dim)
    support_words_mask = torch.zeros(len(all_lengths), max(support_numbers), max(max(tokens_length),max(all_lengths)))

    for index, cap in enumerate(merge_captions):
        end = all_lengths[index]
        target[index, :end, :] = cap[:end, :]
        words_mask[index, :end] = 1.0

        support_end = len(support_set[index])
        for idx in range(support_end):
            tmp_cap = support_set[index][idx]
            support_mask[index, idx] = 1
            support_target[index, idx, :len(tmp_cap), :] = tmp_cap
            support_words_mask[index, idx, :len(tmp_cap)] = 1

    return dict(clip_video_features=clip_videos,
                frame_video_features=frame_videos,
                videos_mask=videos_mask,
                text_feat=target,
                text_mask=words_mask,
                text_labels=labels,
                support_text_feat=support_target,
                support_text_mask=support_words_mask,
                support_mask=support_mask,
                text_tokens = rs_tokens,
                support_tokens = rs_support_tokens,
                support_tokens_mask = tokens_mask,
                support_tokens_padding_mask = support_tokens_padding_mask
                )


def collate_frame_val(data):
    clip_video_features, frame_video_features, idxs, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    # videos
    clip_videos = torch.cat(clip_video_features, dim=0).float()

    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    return clip_videos, frame_videos, videos_mask, idxs, video_ids


def collate_text_val(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, idxs, cap_ids, support_captions,cap_tokens,support_cap_tokens,support_cap_tokens_mask = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        support_set_numbers = []
        support_lengths = []
        tokens_length = [len(temp_caption_tokens) for temp_caption_tokens in cap_tokens]
        support_tokens_num = [len(temp_support_set_tokens) for temp_support_set_tokens in support_cap_tokens]
        support_tokens_length = []
        for batch_idx,single_caption_support_set_tokens in enumerate(support_cap_tokens):
            support_tokens_length.extend([len(temp_support_set_token) for temp_support_set_token in single_caption_support_set_tokens])
        
        for i, cap in enumerate(captions):
            support_set_numbers.append(len(support_captions[i]))
            support_lengths.extend([len(tmp) for tmp in support_captions[i]])
        max_length = max(max(max(support_tokens_length),max(lengths)),max(support_lengths))
        
        target = torch.zeros(len(captions), max_length, captions[0].shape[-1])
        words_mask = torch.zeros(len(captions), max_length)

        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
        support_target = torch.zeros(len(captions), max(support_set_numbers), max_length, captions[0].shape[-1])
        support_words_mask = torch.zeros(len(captions), max(support_set_numbers), max_length)
        support_mask = torch.zeros(len(captions), max(support_set_numbers))
        
        rs_tokens = torch.zeros(len(tokens_length),max(tokens_length))
        for batch_idx,single_caption_token in enumerate(cap_tokens):
            rs_tokens[batch_idx,:len(single_caption_token)] = single_caption_token

        rs_support_tokens = torch.zeros(len(tokens_length),max(support_tokens_num),max_length)# batch_size x number of support set x length
        rs_support_mask = torch.zeros(len(tokens_length),max(support_tokens_num))   
        for batch_idx,single_caption_support_set_tokens in enumerate(support_cap_tokens):
            for support_set_idx,single_support_set_token in enumerate(single_caption_support_set_tokens):
                rs_support_tokens[batch_idx,support_set_idx,:len(single_support_set_token)] = single_support_set_token
                rs_support_mask[batch_idx,support_set_idx] = 1
        support_tokens_padding_mask = (rs_support_tokens == 0) 
        for i, captions in enumerate(support_captions):
            for j, cap in enumerate(captions):
                end = len(cap)
                support_target[i, j, :end, :] = cap
                support_words_mask[i, j, :end] = 1
                support_mask[i, j] = 1
        
    else:
        target = None
        lengths = None
        words_mask = None


    return target, words_mask, idxs, cap_ids, support_target, support_words_mask, support_mask,rs_tokens,rs_support_tokens,rs_support_mask,support_tokens_padding_mask



class Dataset4MS_SL(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, visual_feat, text_feat_path, opt, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        self.video2frames = video2frames
        self.support_set_number = opt.support_set_number
        merge_caption = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                merge_caption.append(caption)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                if video_id in self.vid_caps:
                    self.vid_caps[video_id].append(cap_id)
                else:
                    self.vid_caps[video_id] = []
                    self.vid_caps[video_id].append(cap_id)

        tokens_get_processer = CapPreprocessor()
        tokens,tokens_mask = tokens_get_processer(merge_caption)
        
        self.tokens_ls = tokens
        self.tokens_mask_ls = tokens_mask
        self.tokens_dict = {}
        self.tokens_mask_padding_dict = {}
        for idx,cap_id in enumerate(self.cap_ids):
            self.tokens_dict[cap_id] = tokens[idx]
            self.tokens_mask_padding_dict[cap_id] = tokens_mask[idx]
        print(self.cap_ids[3],merge_caption[3],self.tokens_mask_ls[3])

        self.visual_feat = visual_feat
        self.text_feat_path = text_feat_path
        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
        self.max_desc_len = opt.max_desc_l
        self.open_file = False
        self.length = len(self.vid_caps)

    def __getitem__(self, index):

        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')

            self.open_file = True
        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]

        # video
        frame_list = self.video2frames[video_id]

        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))

        clip_video_feature = average_to_fixed_length(np.array(frame_vecs), self.map_size)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)

        frame_video_feature = uniform_feature_sampling(np.array(frame_vecs), self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        # text
        cap_tensors = []
        cap_tokens = []

        for cap_id in cap_ids:
            cap_feat = self.text_feat[cap_id][...]
            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
            cap_tensors.append(cap_tensor)
            cap_tokens.append(self.tokens_dict[cap_id][:self.max_desc_len])
        all_support_cap_tensors = []
        all_support_cap_tokens = []
        all_support_cap_tokens_padding_mask = []
        for cap_id in cap_ids:
            support_cap_tensors = []
            support_cap_tokens = []
            support_cap_tokens_padding_mask = []
            tmp_cap_ids = cap_ids.copy()
            tmp_cap_ids.remove(cap_id)
            if len(tmp_cap_ids) > self.support_set_number:
                support_cap_ids = random.sample(tmp_cap_ids, self.support_set_number)
            else:
                support_cap_ids = tmp_cap_ids
            for tmp_cap_id in support_cap_ids:
                cap_feat = self.text_feat[tmp_cap_id][...]
                cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
                support_cap_tensors.append(cap_tensor)
                support_cap_tokens.append(self.tokens_dict[tmp_cap_id][:self.max_desc_len])
                support_cap_tokens_padding_mask.append(self.tokens_mask_padding_dict[tmp_cap_id][:self.max_desc_len])
            all_support_cap_tensors.append(support_cap_tensors)
            all_support_cap_tokens.append(support_cap_tokens)
            all_support_cap_tokens_padding_mask.append(support_cap_tokens_padding_mask)
        return clip_video_feature, frame_video_feature, cap_tensors, index, cap_ids, video_id, all_support_cap_tensors,cap_tokens,all_support_cap_tokens,all_support_cap_tokens_padding_mask
    
    def __len__(self):
        return self.length

class VisDataSet4MS_SL(data.Dataset):

    def __init__(self, visual_feat, video2frames, opt, video_ids=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = video2frames.keys()
        self.length = len(self.video_ids)
        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
    def __getitem__(self, index):
        video_id = self.video_ids[index]
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        clip_video_feature = average_to_fixed_length(np.array(frame_vecs), self.map_size)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)

        frame_video_feature = uniform_feature_sampling(np.array(frame_vecs), self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        return clip_video_feature, frame_video_feature, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet4MS_SL(data.Dataset):
    """
    Load captions
    """

    def __init__(self, cap_file, text_feat_path, opt):
        # Captions
        self.captions = {}
        self.cap_ids = []

        self.video_ids = []
        self.vid_caps = {}
        self.support_set_number = opt.support_set_number
        merge_caption = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                merge_caption.append(caption)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                if video_id in self.vid_caps:
                    self.vid_caps[video_id].append(cap_id)
                else:
                    self.vid_caps[video_id] = []
                    self.vid_caps[video_id].append(cap_id)
        self.text_feat_path = text_feat_path
        self.max_desc_len = opt.max_desc_l
        self.open_file = False
        self.length = len(self.cap_ids)
        tokens_get_processer = CapPreprocessor()
        tokens,tokens_mask = tokens_get_processer(merge_caption)
        
        self.tokens_ls = tokens
        self.tokens_mask_ls = tokens_mask
        self.tokens_dict = {}
        self.tokens_mask_dict = {}
        for idx,cap_id in enumerate(self.cap_ids):
            self.tokens_dict[cap_id] = tokens[idx]
            self.tokens_mask_dict[cap_id] = tokens_mask[idx]
        print(self.cap_ids[133],merge_caption[133],self.tokens_mask_ls[133])
        


    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')

            self.open_file = True


        cap_feat = self.text_feat[cap_id][...]

        cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
        cap_tokens = self.tokens_dict[cap_id][:self.max_desc_len]
        # support text
        video_id = cap_id.split('#')[0]
        cap_ids = self.vid_caps[video_id]
        cap_ids.remove(cap_id)
        if len(cap_ids) > self.support_set_number:
            support_cap_ids = random.sample(cap_ids, self.support_set_number)
        else:
            support_cap_ids = cap_ids
        support_cap_tensors = []
        support_cap_tokens = []
        support_cap_tokens_padding_mask = []
        for tmp_cap_id in support_cap_ids:
            support_cap_feat = self.text_feat[tmp_cap_id][...]
            support_cap_tensor = torch.from_numpy(l2_normalize_np_array(support_cap_feat))[:self.max_desc_len]
            support_cap_tensors.append(support_cap_tensor)
            support_cap_tokens.append(self.tokens_dict[tmp_cap_id][:self.max_desc_len])
            support_cap_tokens_padding_mask.append(self.tokens_mask_dict[tmp_cap_id][:self.max_desc_len])#1 x nS x L
        return cap_tensor, index, cap_id, support_cap_tensors,cap_tokens,support_cap_tokens,support_cap_tokens_padding_mask

    def __len__(self):
        return self.length

if __name__ == '__main__':
    pass


