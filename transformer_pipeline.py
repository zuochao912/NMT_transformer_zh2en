import torch
import torch.nn as nn
import os
import spacy
import random
import numpy as np
from torchtext.datasets.translation import TranslationDataset
from torchtext.data.metrics import bleu_score
from torchtext.data import Field, BucketIterator

spacy_zh = spacy.load('zh_core_web_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_zh(text):
    """
    Tokenizes zh/en text from a string into a list of strings
    """
    return [tok.text for tok in spacy_zh.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes zh/en text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

class TextListData(TranslationDataset):
    """TextData"""

    urls = None
    name = 'IWSLT'
    dirname = None

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train', validation='valid', test='test', **kwargs):

        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(TextListData, cls).splits(exts, fields, path, root, train, validation, test, **kwargs)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def prepare_data(device):
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    SRC = Field(tokenize = tokenize_zh, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True, 
                batch_first = True)

    TRG = Field(tokenize = tokenize_en, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True, 
                batch_first = True)

    #path=r"C:\语料\IWSLT\test"

    train_data, val_data, test_data= TextListData.splits(exts=('.zh','.en'),fields=(SRC, TRG),path=r"../IWSLT")
    #中翻英啊！

    SRC.build_vocab(train_data, min_freq = 60) #中文
    TRG.build_vocab(train_data, min_freq = 80) #英文
    #原本这里中英文混杂了啊！，其实是把中文和英文都建了一遍。。

    BATCH_SIZE = 128

    iterators = BucketIterator.splits( 
        (train_data, val_data, test_data), batch_size = BATCH_SIZE,device = device)
    return iterators,(SRC,TRG),(train_data, val_data, test_data)

def train(model, iterator, optimizer, criterion, clip,max_len=100):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        if batch.trg.shape[1]>max_len or batch.src.shape[1]>max_len:
            #由于出现长于max_len的句子，只能截断了
            trg=batch.trg[:,:max_len]
            src=batch.src[:,:max_len]
        else:
            trg=batch.trg
            src=batch.src

        optimizer.zero_grad()
        output, _ = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = criterion(output, trg)
        #print("loss in batch:{}".format(loss))

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion,max_len=100):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
            if batch.trg.shape[1]>max_len or batch.src.shape[1]>max_len:
                #由于出现长于max_len的句子，只能截断了
                trg=batch.trg[:,:max_len]
                src=batch.src[:,:max_len]
            else:
                trg=batch.trg
                src=batch.src      
            output, _ = model(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


from torchtext.data.metrics import bleu_score

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 100):
    
    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('zh_core_web_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor_tmp = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    if src_tensor_tmp.shape[1] >max_len:
        src_tensor=src_tensor_tmp[:,:max_len]
    else:
        src_tensor=src_tensor_tmp

    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention

def calculate_bleu(data, src_field, trg_field, model, device, max_len = 100):
    
    trgs = []
    pred_trgs = []
    
    for datum in data:
        
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        
        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs)