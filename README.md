# bert4epi
BERT model for Enhancer-Promoter Classification

```bash
python3 prepare_data.py --k_fold=5 --balanced
python3 train.py --cell_line='GM12878' --k_fold=5
python3 test.py --cell_line='GM12878' --cross_cell_line='K562' --k_fold=5
```
:warning: By default `--seed=42`.

:warning: Set `--cell_line` and `--cross_cell_line` as same for testing on the same cell-line.

:warning: Set `--k_fold=0` to disable cross-validation.
