#!/bin/bash

for num_layers in 2 3 4 5 6
do
    python3 stage1_pretrain.py --num_layers $num_layers
done
