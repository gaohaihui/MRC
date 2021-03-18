import os
import args
import torch
import random
import pickle
from tqdm import tqdm
from torch import nn, optim

import evaluate
from optimizer import BertAdam
from model_dir.modeling import BertForQuestionAnswering, BertConfig
from dataset.data_format import DataFormat
from torch.utils.data import DataLoader

# 随机种子
random.seed(args.seed)
torch.manual_seed(args.seed)

def train():
    # 加载预训练bert
    model = BertForQuestionAnswering.from_pretrained(r"D:\self\Graduation\data\bert")
    device = args.device
    model.to(device)

    # 准备 optimizer
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=0.1, t_total=args.num_train_optimization_steps)

    # 准备数据
    source_data_path = r"data/train.json"
    train_dataset = DataFormat(source_data_path)
    train_size = int(0.9 * len(train_dataset))
    dev_size = len(train_dataset) - train_size
    train_dataset, dev_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, dev_size])
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=32, num_workers=4)
    # dev data build
    dev_loader = DataLoader(
        dev_dataset, shuffle=False, batch_size=16, num_workers=2)
    best_loss = 100000.0
    model.train()
    for i in range(args.num_train_epochs):
        for step , batch in enumerate(train_loader):
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            input_ids, input_mask, segment_ids, start_positions, end_positions = \
              input_ids.to(device), input_mask.to(device), segment_ids.to(device), start_positions.to(device), end_positions.to(device)

            # 计算loss
            loss, _, _ = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, start_positions=start_positions, end_positions=end_positions)
            loss = loss / args.gradient_accumulation_steps
            print("Steps:", step, "Loss",loss.data)
            loss.backward()

            # 更新梯度
            if (step+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 验证
        eval_loss = evaluate.evaluate(model, dev_loader)
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), './model_dir/' + "best_model")
            model.train()

if __name__ == "__main__":
    train()
