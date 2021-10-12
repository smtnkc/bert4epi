import pandas as pd
import torch
import argparse
import random
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import logging
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, seq_fea = self.encoder(text, labels=label)[:2]

        return loss, seq_fea

def save_checkpoint(save_path, model, dev_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(), 'dev_loss': dev_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['dev_loss']

def save_metrics(save_path, train_loss_list, dev_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'dev_loss_list': dev_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Metrics saved to ==> {save_path}')

def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Metrics loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['dev_loss_list'], state_dict['global_steps_list']

def train(model, optimizer, train_loader, dev_loader, eval_every, num_epochs = 5, criterion = nn.BCELoss(), best_dev_loss = float("Inf")):

    # initialize running values
    running_loss = 0.0
    dev_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    dev_loss_list = []
    global_steps_list = []

    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists('metrics'):
        os.makedirs('metrics')

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (texts, labels), _ in train_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            texts = texts.type(torch.LongTensor)
            texts = texts.to(device)
            output = model(texts, labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():

                    # dev loop
                    for (texts, labels), _ in dev_loader:
                        labels = labels.type(torch.LongTensor)
                        labels = labels.to(device)
                        texts = texts.type(torch.LongTensor)
                        texts = texts.to(device)
                        output = model(texts, labels)
                        loss, _ = output
                        
                        dev_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_dev_loss = dev_running_loss / len(dev_loader)
                train_loss_list.append(average_train_loss)
                dev_loss_list.append(average_dev_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                dev_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Dev Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_dev_loss))
                
                # checkpoint
                if best_dev_loss > average_dev_loss:
                    best_dev_loss = average_dev_loss
                    save_checkpoint('models/{}.pt'.format(args.cell_line), model, best_dev_loss)
                    save_metrics('metrics/{}.pt'.format(args.cell_line), train_loss_list, dev_loss_list, global_steps_list)
    
    save_metrics('metrics/{}.pt'.format(args.cell_line), train_loss_list, dev_loss_list, global_steps_list)
    print('Finished Training!')

def evaluate(model, test_loader, args):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (texts, labels), _ in test_loader:

                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
                texts = texts.type(torch.LongTensor)
                texts = texts.to(device)
                output = model(texts, labels)

                _, output = output
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))

    # cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    # ax = plt.subplot()
    # sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    # ax.set_title('Confusion Matrix')

    # ax.set_xlabel('Predicted Labels')
    # ax.set_ylabel('True Labels')

    # ax.xaxis.set_ticklabels(['ENHANCER', 'PROMOTER'])
    # ax.yaxis.set_ticklabels(['ENHANCER', 'PROMOTER'])

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    print("AUC =", metrics.auc(fpr, tpr))

    # LOGS

    if not os.path.isdir("results"):
        os.makedirs("results")

    log_file = "results/{}.txt".format(args.cell_line)
    open(log_file, 'w').close() # clear file content
    logging.basicConfig(format='%(message)s', filename=log_file,level=logging.DEBUG)
    logging.info(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    logging.info("AUC = {:.5f}".format(metrics.auc(fpr, tpr)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert4epi')
    parser.add_argument('--cell_line', default='GM12878', type=str) # GM12878, HUVEC, HeLa-S3, IMR90, K562, NHEK, combined
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--balanced', default=True, type=bool)
    args = parser.parse_args()
    random.seed(args.seed)

    print(torch.__version__)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    frag_path = 'data/{}/frag_pairs{}.csv'.format(args.cell_line, '_balanced' if args.balanced else '')
    df_frag_pairs = pd.read_csv(frag_path)
    df_frag_pairs = df_frag_pairs[['enhancer_frag_name', 'enhancer_frag_seq', 'promoter_frag_name', 'promoter_frag_seq']]
    df_frag_pairs.columns = ['enhancer_name', 'enhancer_seq', 'promoter_name', 'promoter_seq']
    df_enh_frags = df_frag_pairs.drop_duplicates(subset=['enhancer_name'])[['enhancer_name', 'enhancer_seq']].reset_index(drop=True)
    df_pro_frags = df_frag_pairs.drop_duplicates(subset=['promoter_name'])[['promoter_name', 'promoter_seq']].reset_index(drop=True)

    df_enh_frags.columns = ['label', 'text']
    for i in range(len(df_enh_frags)):
        df_enh_frags.at[i, 'text'] = " ".join(df_enh_frags.at[i, 'text'])
        df_enh_frags.at[i, 'label'] = 1

    df_pro_frags.columns = ['label', 'text']
    for i in range(len(df_pro_frags)):
        df_pro_frags.at[i, 'text'] = " ".join(df_pro_frags.at[i, 'text'])
        df_pro_frags.at[i, 'label'] = 0

    first_column = df_enh_frags.pop('text')
    df_enh_frags.insert(0, 'text', first_column)

    first_column = df_pro_frags.pop('text')
    df_pro_frags.insert(0, 'text', first_column)

    print(df_enh_frags.head())
    print(df_pro_frags.head())
    print('{} enhancers - {} promoters'.format(len(df_enh_frags), len(df_enh_frags)))

    df_enh_train, df_enh_test = train_test_split(df_enh_frags, test_size=0.1, random_state=42)
    df_pro_train, df_pro_test = train_test_split(df_pro_frags, test_size=0.1, random_state=42)

    df_train_dev = df_enh_train.append(df_pro_train).sample(frac=1).reset_index(drop=True) # append and shuffle
    df_test = df_enh_test.append(df_pro_test).sample(frac=1).reset_index(drop=True) # append and shuffle

    # Model parameter

    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Fields

    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True, pad_token=PAD_INDEX, unk_token=UNK_INDEX)

    fields = [('text', text_field), ('label', label_field)]

    df_train, df_dev = train_test_split(df_train_dev, test_size=0.1, random_state=42, shuffle=True)

    df_train = df_train.reset_index(drop=True)
    df_dev = df_dev.reset_index(drop=True)

    df_train.to_csv("data/{}/train.csv".format(args.cell_line), index=False)
    df_dev.to_csv("data/{}/dev.csv".format(args.cell_line), index=False)
    df_test.to_csv("data/{}/test.csv".format(args.cell_line), index=False)

    print(len(df_train), "train", len(df_dev), "dev", len(df_test), "test")

    print(df_train.head())
    print(df_dev.head())
    print(df_test.head())

    train_set, dev_set, test_set = TabularDataset.splits(path='data/{}'.format(args.cell_line),
        train='train.csv', validation='dev.csv', test='test.csv', format='CSV', fields=fields, skip_header=True)

    train_iter = BucketIterator(train_set, batch_size=16, sort_key=lambda x: len(x.text), device=device, train=True, sort=True, sort_within_batch=True)
    dev_iter = BucketIterator(dev_set, batch_size=16, sort_key=lambda x: len(x.text), device=device, train=True, sort=True, sort_within_batch=True)
    test_iter = Iterator(test_set, batch_size=16, device=device, train=False, shuffle=False, sort=False)

    model = BERT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    train(model=model, optimizer=optimizer, train_loader = train_iter, dev_loader = dev_iter, eval_every = len(train_iter) // 2, num_epochs = 5)

    train_loss_list, dev_loss_list, global_steps_list = load_metrics('metrics/{}.pt'.format(args.cell_line))
    # plt.plot(global_steps_list, train_loss_list, label='Train')
    # plt.plot(global_steps_list, dev_loss_list, label='Dev')
    # plt.xlabel('Global Steps')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    best_model = BERT().to(device)
    load_checkpoint('models/{}.pt'.format(args.cell_line), best_model)

    evaluate(best_model, test_iter, args)
