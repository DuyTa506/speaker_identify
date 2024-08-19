LR=0.0005
EPOCHS=20
BATCH_SIZE=2048
OUTPUT_BASE="siamese_fbanks_saved/"
TRAIN_DATA="fbannks_train"
TEST_DATA="fbannks_test"

for NUM_LAYERS in 2 3 4 5 6
do
  PRETRAINED_MODEL_PATH="saved_models_cross_entropy/${NUM_LAYERS}/"
  OUTPUT_MODEL_PATH="${OUTPUT_BASE}${NUM_LAYERS}/"
  
  echo "Running training with num_layers=${NUM_LAYERS}, pretrained_model_path=${PRETRAINED_MODEL_PATH}, output_model_path=${OUTPUT_MODEL_PATH}"

  python3 stage2_finetune.py \
    --num_layers ${NUM_LAYERS} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --pretrained_model_path ${PRETRAINED_MODEL_PATH} \
    --output_model_path ${OUTPUT_MODEL_PATH} \
    --train_data ${TRAIN_DATA} \
    --test_data ${TEST_DATA}
  
  echo "Finished training with num_layers=${NUM_LAYERS}"
done
