#!/bin/bash

TRAIN_FOLDER="fbannks_train"
TEST_FOLDER="fbannks_test"
EPOCHS=20
BATCH_SIZE=2048
LR=0.0005

for num_layers in 2 3 4 5 6
do
    echo "Starting training with $num_layers layers..."
    
    python3 stage1_pretrain.py \
        --num_layers $num_layers \
        --train_folder $TRAIN_FOLDER \
        --test_folder $TEST_FOLDER \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR
    
    if [ $? -eq 0 ]; then
        echo "Training with $num_layers layers completed successfully."
    else
        echo "Error occurred during training with $num_layers layers."
        exit 1
    fi
done

echo "All training runs completed."
