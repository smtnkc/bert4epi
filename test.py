import argparse
import random
import torch
from transformers import BertTokenizer
from torchtext.data import Field, TabularDataset, Iterator
from utils import BERT, evaluate, load_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert4epi')
    parser.add_argument('--cell_line', default='GM12878', type=str) # GM12878, HUVEC, HeLa-S3, IMR90, K562, NHEK, combined
    parser.add_argument('--cross_cell_line', default=None, type=str) # GM12878, HUVEC, HeLa-S3, IMR90, K562, NHEK, combined
    parser.add_argument('--seed', default=42, type=int)
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

    if args.cross_cell_line == None or (args.cell_line == args.cross_cell_line):
        # Use test data generated for the same cell-line
        print("TESTING ON SAME CELL-LINE ({})".format(args.cell_line))
        _, _, test_set = TabularDataset.splits(path='data/{}'.format(args.cell_line), train='train.csv', validation='dev.csv', test='test.csv', format='CSV', fields=fields, skip_header=True)
    else:
        # Use test data generated for the cross cell-line
        print("TESTING ON CROSS CELL-LINE ({})".format(args.cross_cell_line))
        _, _, test_set = TabularDataset.splits(path='data/{}'.format(args.cross_cell_line), train='train.csv', validation='dev.csv', test='test.csv', format='CSV', fields=fields, skip_header=True)

    test_iter = Iterator(test_set, batch_size=16, device=device, train=False, shuffle=False, sort=False)

    print("Initializing TEST model...")
    best_model = BERT().to(device)  # a new model instance
    load_checkpoint('models/{}.pt'.format(args.cell_line), best_model, device)

    evaluate(best_model, device, test_iter, args.cell_line, args.cross_cell_line)
