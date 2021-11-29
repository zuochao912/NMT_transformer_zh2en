import torch
import torch.nn as nn

from transformer_utils import Encoder,Decoder
from transformer_model import Seq2Seq
from transformer_pipeline import prepare_data,calculate_bleu,initialize_weights

# prepare the dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
iterators,fileds,datas=prepare_data(device)

SRC,TRG=fileds
train_iterator, valid_iterator, test_iterator=iterators
train_data, val_data, test_data=datas

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
              device)


dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.apply(initialize_weights)


criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
model.load_state_dict(torch.load('tut-model.pt',map_location=torch.device('cpu')))
#model.load_state_dict(torch.load('tut-model.pt'))

bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)
print(f'BLEU score = {bleu_score*100:.2f}')