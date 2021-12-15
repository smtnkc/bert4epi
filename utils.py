import torch
import os
import torch.nn as nn
import logging
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

def train(model, device, optimizer, train_loader, dev_loader, eval_every, num_epochs, best_dev_loss, cell_line):

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
                    save_checkpoint('models/{}.pt'.format(cell_line), model, best_dev_loss)
                    save_metrics('metrics/{}.pt'.format(cell_line), train_loss_list, dev_loss_list, global_steps_list)

    save_metrics('metrics/{}.pt'.format(cell_line), train_loss_list, dev_loss_list, global_steps_list)
    print('Finished Training!')

def evaluate(model, device, test_loader, cell_line, cross_cell_line):
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

    if cross_cell_line == None:
        log_file = "results/{}.txt".format(cell_line)
    else:
        log_file = "results/{}.txt".format(cell_line + '_' + cross_cell_line)
    open(log_file, 'w').close() # clear file content
    logging.basicConfig(format='%(message)s', filename=log_file,level=logging.DEBUG)
    logging.info(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    logging.info("AUC = {:.5f}".format(metrics.auc(fpr, tpr)))
