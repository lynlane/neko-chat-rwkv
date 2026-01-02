########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import torch.nn.functional as F

import numpy as np
import torch
import lightning as L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning_utilities.core.rank_zero import rank_zero_info
from .binidx import MMapIndexedDataset
from rwkvt.args_type import TrainingArgs
from rwkv.utils import PIPELINE
from rwkvt.dataset.SFTdataset import sft_dataset
import time
import jsonlines
from .mask import generate_mask, create_mask,mask_fn_dict
pipeline = PIPELINE('rwkv6', "rwkv_vocab_v20230424")


# 用法

class MyDataset(Dataset):
    def __init__(self, args, processor=None):

        self.args = args
        self.processor = processor
        self.data_type = args.data_type

        if args.data_type == "sft":
            self.data = sft_dataset(args)
        elif args.data_type == "jsonl":
            with jsonlines.open(args.data_file) as file:
                self.data = list(file)
            if args.epoch_steps < len(self.data) :
                self.data = self.data[:args.epoch_steps]
        elif args.data_type == "binidx":
            self.data = MMapIndexedDataset(args.data_file)
            self.data = self.data.head(args.epoch_steps)
            self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
            rank_zero_info(f"Data has {self.data_size} tokens.")

        print(f"Trimmed to {len(self.data)} samples for epoch_steps {args.epoch_steps}.")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        args = self.args

        if args.data_type == "sft":

            inputs, labels, attn_mask = self.data[0][idx], self.data[1][idx], self.data[2][idx]
            labels= torch.roll(labels, shifts=-1, dims=-1)

            return inputs, labels, attn_mask
        elif args.data_type == "jsonl":
            ctx_len = args.ctx_len
            req_len = ctx_len + 1
            # 处理instruction和output字段，组合成对话格式
            item = self.data[idx]
            # 检查字段名称，处理不同的数据集格式
            if 'instruction' in item and 'output' in item:
                # NekoQA-10K数据集格式
                instruction = item['instruction']
                output = item['output']
                # 格式化为指令-响应对话
                ctx = f"User: {instruction}\nAssistant: {output}"
            elif 'text' in item:
                # 原始text字段格式
                ctx = item['text']
            else:
                # 其他格式，组合所有字段
                ctx = " ".join([str(v) for v in item.values()])
            
            token = torch.tensor(pipeline.encode(ctx))
            token_len = len(token)
            min_len = min(token_len, req_len)
            if req_len < token_len :
                token = token[:req_len]
                pad_len = 0
            else:
                pad_len = req_len - token_len
        
            dix = F.pad(token, (0, pad_len), value=0)
            label = F.pad(token, (0, pad_len), value=-100)
            x = dix[:-1]
            y = label[1:]

        else:
            ctx_len = args.ctx_len
            req_len = ctx_len + 1
            data = self.data


            if args.data_type == "binidx":
                if args.dataload == 'pad':
                    dix, min_len = data.pad(idx=idx, length=req_len)
                elif args.dataload == 'only':
                    dix = data.only(idx=idx, length=req_len).astype(int)

            x = torch.tensor(dix[:-1], dtype=torch.long)
            dix[min_len:] = -100
            y = torch.tensor(dix[1:], dtype=torch.long)

        mask_fn = mask_fn_dict.get(args.loss_mask)

        if mask_fn!=None:
            t1 = pipeline.encode('User:')
            t2 = pipeline.encode('Assistant:')
            y = mask_fn(dix, t1, t2, min_len)
        return x, y
    

import lightning as L
from torch.utils.data import DataLoader

class MyDataModule(L.LightningDataModule):
    def __init__(self, args, processor=None):
        super().__init__()
        self.args = args
        self.processor = processor

    def setup(self, stage=None):
        self.train_dataset = MyDataset(self.args, self.processor)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.micro_bsz,
            shuffle=True,    # Lightning 自动替换成 DistributedSampler
            num_workers=self.args.num_workers,
            pin_memory=True
        )