# bert4epi
BERT model for Enhancer-Promoter Classification

```bash
python3 prepare_data.py --balanced
python3 train.py --cell_line='GM12878'
python3 test.py --cell_line='GM12878' --cross_cell_line='K562'
```
:warning: By default `--seed=42`.

:warning: Unset `--cross_cell_line` for testing on the same cell-line.
