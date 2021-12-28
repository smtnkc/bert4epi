import argparse
import random
import torch
import torch.optim as optim
from transformers import BertTokenizer
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from utils import BERT, train, load_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert4epi')
    parser.add_argument('--cell_line', default='GM12878', type=str) # GM12878, HUVEC, HeLa-S3, IMR90, K562, NHEK, combined
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--k_fold', default=0, type=int) # set a positive int for cross-validation
    args = parser.parse_args()
    random.seed(args.seed)

    print("\n{}".format(torch.__version__))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("{}\n".format(device))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Model parameter
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Fields
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    fields = [('text', text_field), ('label', label_field)]

    for cv_step in range(args.k_fold + 1):

        if cv_step == 0 and args.k_fold > 0:
            continue # xxx_0.csv files are not for cross-validation

        train_set, dev_set, _ = TabularDataset.splits(path='data/{}'.format(args.cell_line),
            train='train_{}.csv'.format(cv_step),
            validation='dev_{}.csv'.format(cv_step),
            test='test_{}.csv'.format(cv_step),
            format='CSV', fields=fields, skip_header=True)

        train_iter = BucketIterator(train_set, batch_size=16, sort_key=lambda x: len(x.text), device=device, train=True, sort=True, sort_within_batch=True)
        dev_iter = BucketIterator(dev_set, batch_size=16, sort_key=lambda x: len(x.text), device=device, train=True, sort=True, sort_within_batch=True)

        print("Initializing BERT model...")
        bert_model = BERT().to(device)
        adam_opt = optim.Adam(bert_model.parameters(), lr=2e-5)

        train(bert_model, device, adam_opt, train_iter, dev_iter, eval_every = len(train_iter) // 2,
            num_epochs = 5, best_dev_loss = float("Inf"), cell_line = args.cell_line, cv_step = cv_step)

        # train_loss_list, dev_loss_list, global_steps_list = load_metrics('metrics/{}_{}.pt'.format(args.cell_line, i), device)
        # plt.plot(global_steps_list, train_loss_list, label='Train')
        # plt.plot(global_steps_list, dev_loss_list, label='Dev')
        # plt.xlabel('Global Steps')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()

        if cv_step == 0 and args.k_fold == 0:
            break # no cross-validation
