import pandas as pd
import argparse
import random
from sklearn.model_selection import train_test_split


def getFragments(cell_line, balanced):
    frag_path = 'data/{}/frag_pairs{}.csv'.format(cell_line, '_balanced' if balanced else '')
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

    print("{} - {} ENHANCERS:\n{}\n".format(cell_line, len(df_enh_frags), df_enh_frags.head()))
    print("{} - {} PROMOTERS:\n{}\n".format(cell_line, len(df_enh_frags), df_pro_frags.head()))

    return df_enh_frags, df_pro_frags

def trainDevTestSplit(cell_line, balanced, seed):

    df_enh_frags, df_pro_frags = getFragments(cell_line, balanced)

    df_enh_train, df_enh_test = train_test_split(df_enh_frags, test_size=0.1, random_state=seed)
    df_pro_train, df_pro_test = train_test_split(df_pro_frags, test_size=0.1, random_state=seed)

    df_train_dev = df_enh_train.append(df_pro_train).sample(frac=1, random_state=seed).reset_index(drop=True) # merge training enhancers and promoters and shuffle
    df_test = df_enh_test.append(df_pro_test).sample(frac=1, random_state=seed).reset_index(drop=True) # merge test enhancers and promoters and shuffle

    df_train, df_dev = train_test_split(df_train_dev, test_size=0.1, random_state=seed) # split train and dev data

    df_train = df_train.reset_index(drop=True)
    df_dev = df_dev.reset_index(drop=True)

    df_train.to_csv("data/{}/train.csv".format(cell_line), index=False)
    df_dev.to_csv("data/{}/dev.csv".format(cell_line), index=False)
    df_test.to_csv("data/{}/test.csv".format(cell_line), index=False)

    print("{} - {} TRAIN:\n{}\n".format(cell_line, len(df_train), df_train.head()))
    print("{} - {} DEV:\n{}\n".format(cell_line, len(df_dev), df_dev.head()))
    print("{} - {} TEST:\n{}\n".format(cell_line, len(df_test), df_test.head()))

    return df_train, df_dev, df_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert4epi')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--balanced', action='store_true') # set to balance enhancers and promoters
    args = parser.parse_args()
    random.seed(args.seed)

    cell_lines = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'combined']
    # Generate Train/Dev/Test CSVs for the cell-line
    for cell_line in cell_lines:
        _, _, _ = trainDevTestSplit(cell_line, args.balanced, args.seed) # balanced train/dev/test split
