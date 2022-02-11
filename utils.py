import torch
import os
import torch.nn as nn
import logging
import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import BertForSequenceClassification


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
    print('Model saved to ==> {}'.format(save_path))

def load_checkpoint(load_path, model, device):

    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print('Model loaded from <== {}'.format(load_path))

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['dev_loss']

def save_metrics(save_path, train_loss_list, dev_loss_list, global_steps_list):

    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'dev_loss_list': dev_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print('Metrics saved to ==> {}'.format(save_path))

def load_metrics(load_path, device):

    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print('Metrics loaded from <== {}'.format(load_path))

    return state_dict['train_loss_list'], state_dict['dev_loss_list'], state_dict['global_steps_list']

def train(model, device, optimizer, train_loader, dev_loader, eval_every, num_epochs, best_dev_loss, cell_line, cv_step):

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
    t1 = time.time()
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
                    save_checkpoint('models/{}_{}.pt'.format(cell_line, cv_step), model, best_dev_loss)
                    save_metrics('metrics/{}_{}.pt'.format(cell_line, cv_step), train_loss_list, dev_loss_list, global_steps_list)

    t2 = time.time()
    print('Finished training in {:.5f} seconds.'.format(t2 - t1))

    ### LOGS

    if not os.path.isdir("results"):
        os.makedirs("results")

    logger = logging.getLogger('training_logger_{}'.format(cv_step))
    logger.setLevel(logging.INFO)
    log_file = "results/training_time.txt"
    handler = logging.FileHandler(log_file, 'a')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('{} TRAINING TIME (FOLD {}) = {:.5f}'.format(cell_line, cv_step, t2 - t1))

def evaluate(model, device, test_loader, cell_line, cross_cell_line, cv_step):
    y_pred = []
    y_true = []

    t1 = time.time()
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

    cr = classification_report(y_true, y_pred, labels=[1,0], digits=4)
    print(cr)
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    print('TEST CONFUSION = {}'.format(cm.tolist()))
    f1 = f1_score(y_true, y_pred)
    print("TEST F1 = {:.5f}".format(f1))
    t2 = time.time()
    print('TEST TIME = {:.5f}{}'.format(t2 - t1, '\n' * 3))

    ### PLOT CONFUSION MATRIX

    # ax = plt.subplot()
    # sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")
    # ax.set_title('Confusion Matrix')
    # ax.set_xlabel('Predicted Labels')
    # ax.set_ylabel('True Labels')
    # ax.xaxis.set_ticklabels(['ENHANCER', 'PROMOTER'])
    # ax.yaxis.set_ticklabels(['ENHANCER', 'PROMOTER'])

    ### LOGS

    if not os.path.isdir("results"):
        os.makedirs("results")

    if cross_cell_line == None:
        cross_cell_line = cell_line

    logger = logging.getLogger('logger_{}'.format(cv_step))
    logger.setLevel(logging.INFO)
    log_file = "results/{}_{}_{}.txt".format(cell_line, cross_cell_line, cv_step)
    handler = logging.FileHandler(log_file, 'w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(cr)
    logger.info('TEST CONFUSION = {}'.format(cm.tolist()))
    logger.info('TEST F1 = {:.5f}'.format(f1))
    logger.info('TEST TIME = {:.5f}'.format(t2 - t1))
