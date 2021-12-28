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

    df_enh_train, df_enh_test = train_test_split(df_enh_frags, test_size=0.2, random_state=seed)
    df_pro_train, df_pro_test = train_test_split(df_pro_frags, test_size=0.2, random_state=seed)

    df_train_dev = df_enh_train.append(df_pro_train).sample(frac=1, random_state=seed).reset_index(drop=True) # merge training enhancers and promoters and shuffle
    df_test = df_enh_test.append(df_pro_test).sample(frac=1, random_state=seed).reset_index(drop=True) # merge test enhancers and promoters and shuffle

    df_train, df_dev = train_test_split(df_train_dev, test_size=0.2, random_state=seed) # split train and dev data

    df_train = df_train.reset_index(drop=True)
    df_dev = df_dev.reset_index(drop=True)

    df_train.to_csv("data/{}/train_0.csv".format(cell_line), index=False)
    df_dev.to_csv("data/{}/dev_0.csv".format(cell_line), index=False)
    df_test.to_csv("data/{}/test_0.csv".format(cell_line), index=False)

    print("{} TRAIN-0 HAS {} ELEMENTS:\n{}\n".format(cell_line, len(df_train), df_train.head()))
    print("{} DEV-0 HAS {} ELEMENTS:\n{}\n".format(cell_line, len(df_dev), df_dev.head()))
    print("{} TEST-0 HAS {} ELEMENTS:\n{}\n".format(cell_line, len(df_test), df_test.head()))


def trainDevTestSplitCV(cell_line, balanced, seed, k_fold):

    def getCVSplits(df, seed, k_fold):
        df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True) # shuffle rows

        L = len(df_shuffled)
        test_size = int(L/k_fold)
        trains = []
        tests = []
        starting = 0
        for cv_step in range(k_fold):
            ending = starting + test_size
            test_indices = range(starting, ending)
            print("K = {}: {}".format(cv_step+1, test_indices))
            df_test = df_shuffled.iloc[test_indices].reset_index(drop=True)
            df_train = df_shuffled.drop(test_indices).reset_index(drop=True)
            tests.append(df_test)
            trains.append(df_train)
            starting = ending
        print()
        return trains, tests

    df_enh_frags, df_pro_frags = getFragments(cell_line, balanced)
    print("ENHANCER PORTIONS FOR CV:")
    enh_trains, enh_tests = getCVSplits(df_enh_frags, seed, k_fold)
    print("PROMOTER PORTIONS FOR CV:")
    pro_trains, pro_tests = getCVSplits(df_pro_frags, seed, k_fold)

    cv_step = 1
    for df_enh_train, df_pro_train, df_enh_test, df_pro_test in zip(enh_trains, pro_trains, enh_tests, pro_tests):
        df_train_dev = df_enh_train.append(df_pro_train).sample(frac=1, random_state=seed).reset_index(drop=True) # merge training enhancers and promoters and shuffle
        df_test = df_enh_test.append(df_pro_test).sample(frac=1, random_state=seed).reset_index(drop=True) # merge test enhancers and promoters and shuffle

        df_train, df_dev = train_test_split(df_train_dev, test_size=0.2, random_state=seed) # split train and dev data

        df_train = df_train.reset_index(drop=True)
        df_dev = df_dev.reset_index(drop=True)

        df_train.to_csv("data/{}/train_{}.csv".format(cell_line, cv_step), index=False)
        df_dev.to_csv("data/{}/dev_{}.csv".format(cell_line, cv_step), index=False)
        df_test.to_csv("data/{}/test_{}.csv".format(cell_line, cv_step), index=False)

        print("{} TRAIN-{} HAS {} ELEMENTS:\n{}\n".format(cell_line, cv_step, len(df_train), df_train.head()))
        print("{} DEV-{} HAS {} ELEMENTS:\n{}\n".format(cell_line, cv_step, len(df_dev), df_dev.head()))
        print("{} TEST-{} HAS {} ELEMENTS:\n{}\n".format(cell_line, cv_step, len(df_test), df_test.head()))
        cv_step += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert4epi')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--balanced', action='store_true') # set to balance enhancers and promoters
    parser.add_argument('--k_fold', default=0, type=int) # set a positive int for cross-validation
    args = parser.parse_args()
    random.seed(args.seed)

    cell_lines = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'combined']
    # Generate Train/Dev/Test CSVs for the cell-line
    for cell_line in cell_lines:
        trainDevTestSplit(cell_line, args.balanced, args.seed) # prepare non-CV dataset (0)
        trainDevTestSplitCV(cell_line, args.balanced, args.seed, args.k_fold) # prepare CV datasets (1..K)
