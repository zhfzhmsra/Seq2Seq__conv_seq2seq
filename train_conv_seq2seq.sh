export PYTHONIOENCODING=UTF-8

export DATA_PATH=/home/deeprec/fuz/codes/nmt/data

export VOCAB_SOURCE=${DATA_PATH}/vocab.vi
export VOCAB_TARGET=${DATA_PATH}/vocab.en
export TRAIN_SOURCES=${DATA_PATH}/train.vi
export TRAIN_TARGETS=${DATA_PATH}/train.en
export DEV_SOURCES=${DATA_PATH}/tst2012.vi
export DEV_TARGETS=${DATA_PATH}/tst2012.en
export TEST_SOURCES=${DATA_PATH}/tst2013.vi
export TEST_TARGETS=${DATA_PATH}/tst2013.en

export TRAIN_STEPS=1000000


export MODEL_DIR=/home/deeprec/fuz/codes/Seq2Seq__conv_seq2seq/nmt_conv_seq2seq
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ./example_configs/conv_seq2seq.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipelineFairseq
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --eval_every_n_steps 5000 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR





