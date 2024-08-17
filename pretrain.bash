#!/bin/bash

for num_layers in 2 3 4 5 6
do
    echo "Starting training with $num_layers layers..."
    python3 stage1_pretrain.py --num_layers $num_layers --batch_size 64
    
    if [ $? -eq 0 ]; then
        echo "Training with $num_layers layers completed successfully."
    else
        echo "Error occurred during training with $num_layers layers."
        exit 1
    fi
done

echo "All training runs completed."
