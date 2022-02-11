#!/bin/bash

K_FOLD=$1

for CELL_LINE in 'GM12878' 'HUVEC' 'HeLa-S3' 'K562' 'combined'; do
    printf "\n\n\npython3 train.py --cell_line='$CELL_LINE' --k_fold=$K_FOLD\n"
    python3 train.py --cell_line="$CELL_LINE" --k_fold=$K_FOLD
done

for CELL_LINE in 'GM12878' 'HUVEC' 'HeLa-S3' 'K562'; do
    for CROSS_CELL_LINE in 'GM12878' 'HUVEC' 'HeLa-S3' 'K562'; do
        printf "\n\n\npython3 test.py --cell_line='$CELL_LINE' --cross_cell_line='$CROSS_CELL_LINE' --k_fold=$K_FOLD\n"
        python3 test.py --cell_line="$CELL_LINE" --cross_cell_line="$CROSS_CELL_LINE" --k_fold=$K_FOLD
    done
done

printf "\n\n\npython3 test.py --cell_line='combined' --cross_cell_line='combined' --k_fold=$K_FOLD\n"
python3 test.py --cell_line=combined --cross_cell_line=combined --k_fold=$K_FOLD
