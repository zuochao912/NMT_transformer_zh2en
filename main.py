import torch
import torch.nn as nn
import torch.optim as optim
from transformer_utils import Encoder,Decoder
from transformer_model import Seq2Seq
from transformer_pipeline import train,evaluate,prepare_data

import math
import time

import sys
#先中译英吧
# prepare the dataloader

GPUidx=eval(sys.argv[1])
if torch.cuda.is_available():
    print("config model on GPU:{}".format(GPUidx))
else:
    print("config model on CPU")

device = torch.device('cuda:{}'.format(GPUidx) if torch.cuda.is_available() else 'cpu')
iterators,fileds,_=prepare_data(device)

SRC,TRG=fileds
train_iterator, valid_iterator, test_iterator=iterators

print("-----data is done!-----")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
print("Input_dim:{},Output_dim:{}".format(INPUT_DIM,OUTPUT_DIM))

#Initialize the whole model
print("-----initializing model!-----")
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3  #原本这里是3，但是Transformer原文是6
DEC_LAYERS = 3  #原本这里是3，但是Transformer原文是6
ENC_HEADS = 8   #multihead部分
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
MAX_length=100

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device,
              MAX_length)
# ENC_LAYERS指的是，有几个identical 的encoder layer(multi head self attention + LN+ residual)
#HID_DIM: token的 embedding的维度,也是多头注意力中，WQ,WK,WV矩阵的维度(HID_DIM,HID_DIM),WO矩阵也是(HID_DIM,HID_DIM)
#ENC_PF_DIM：Encoder layer中，全连接层的hidden_layer的维度；
#ENC_HEADS：指的是，多头注意力的头数；

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device,
              MAX_length)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
print(f'The model has {count_parameters(model):,} trainable parameters')

model.apply(initialize_weights)

# optimizer parameter

LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

#train process
N_EPOCHS = 30
CLIP = 1

best_valid_loss = float('inf')

print("----training-----!")

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP,MAX_length)  #这里改成test了
    valid_loss = evaluate(model, valid_iterator, criterion,MAX_length)

    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

#test process
print("-----testing the model!-----")

model.load_state_dict(torch.load('tut-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)