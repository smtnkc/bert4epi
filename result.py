import os
import pandas as pd


files = []
train_cell_lines = []
test_cell_lines = []
aucs = []

for file in os.listdir("results"):
    if file.endswith(".txt"):
        files.append(file)

files = sorted(files)

for filename in files:
    f = open('results/' + filename, "r")
    auc = f.read().split('AUC = ')[1]
    f.close()

    train_cell_lines.append(filename.split('_')[0])
    test_cell_lines.append(filename.split('_')[1])
    aucs.append(float(auc))

data = {
    'train_cell_line': train_cell_lines,
    'test_cell_line': test_cell_lines,
    'auc': aucs
}

df = pd.DataFrame(data)
df.to_csv('results/results.csv', index=False)
