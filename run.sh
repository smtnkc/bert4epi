#!/bin/bash
for CELL_LINE in 'GM12878' 'HUVEC' 'HeLa-S3' 'K562' 'combined'; do
    printf "\n\n\n\n\npython3 train.py --cell_line='$CELL_LINE' --k_fold=5\n\n\n\n\n"
    python3 train.py --cell_line="$CELL_LINE" --k_fold=5
done

for CELL_LINE in 'GM12878' 'HUVEC' 'HeLa-S3' 'K562'; do
    for CROSS_CELL_LINE in 'GM12878' 'HUVEC' 'HeLa-S3' 'K562'; do
        printf "\n\n\n\n\npython3 test.py --cell_line='$CELL_LINE' --cross_cell_line='$CROSS_CELL_LINE' --k_fold=5\n\n\n\n\n"
        python3 test.py --cell_line="$CELL_LINE" --cross_cell_line="$CROSS_CELL_LINE" --k_fold=5
    done
done
