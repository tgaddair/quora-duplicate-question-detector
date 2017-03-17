#!/bin/bash

RUN="1489285163"  # baseline 50d
RUN="1489434039"  # cnn 50d
RUN="1489551799"  # cnn 50d 40K training -> overfitting
RUN="1489553936"  # cnn 100d 40K training -> overfitting
RUN="1489557450"  # cnn 100d 400K training -> overfitting
RUN="1489643981"  # cnn twitter 200d 400K training -> 0.802677

vectors="./glove.6B/glove.6B.100d.txt"
vectors="./glove.twitter.27B/glove.twitter.27B.200d.txt"

test_data="./data/test.full.tsv"
#test_data="./Quora_question_pair_partition/test.tsv"

# tensorboard --logdir ./runs/${RUN}/summaries/
python eval.py \
    --test_data_file="${test_data}" \
    --checkpoint_dir="runs/${RUN}/checkpoints" \
    --embeddings_file="${vectors}"
