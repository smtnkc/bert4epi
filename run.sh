#!/bin/bash
for CELL_LINE in 'GM12878' 'HUVEC' 'HeLa-S3' 'IMR90' 'K562' 'NHEK' 'combined'; do
    echo python main.py --cell_line="$CELL_LINE" --balanced=True --seed=42
    python main.py --cell_line="$CELL_LINE" --balanced=True --seed=42
done