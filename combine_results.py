import os
import pandas as pd


files = []
train_cell_lines = []
test_cell_lines = []
f1_scores = []
test_times = []
confusions = []

for file in os.listdir("results"):
    if file.endswith(".txt") and not file.startswith('training'):
        files.append(file)

files = sorted(files)

for filename in files:
    f_name = 'results/' + filename
    f = open(f_name, "r")
    print(f_name)
    content = f.read()
    f1 = content.split('F1 = ')[1].split('\n')[0]
    print('F1 =', f1)
    test_time = content.split('TIME = ')[1]
    print('TIME =', test_time)
    confusion = content.split('CONFUSION = ')[1].split('\n')[0]
    f.close()

    train_cell_lines.append(filename.split('_')[0])
    test_cell_lines.append(filename.split('_')[1])
    f1_scores.append(float(f1))
    test_times.append(float(test_time))
    confusions.append(confusion)


data = {
    'train_cell_line': train_cell_lines,
    'test_cell_line': test_cell_lines,
    'f1_scores': f1_scores,
    'test_time': test_times,
    'confusion': confusions
}

df = pd.DataFrame(data)
df.to_csv('results/results.csv', index=False)
