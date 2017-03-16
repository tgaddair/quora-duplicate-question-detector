#!/bin/bash

RUN="1489285163"  # baseline 50d
RUN="1489434039"  # cnn 50d
RUN="1489551799"  # cnn 50d 40K training -> overfitting
RUN="1489553936"  # cnn 100d 40K training -> overfitting
RUN="1489557450"  # cnn 100d 400K training -> overfitting

vectors="./glove.6B/glove.6B.100d.txt"
vectors="./glove.twitter.27B/glove.twitter.27B.100d.txt"

# tensorboard --logdir ./runs/${RUN}/summaries/
python eval.py --checkpoint_dir="runs/${RUN}/checkpoints" --embeddings_file="${vectors}"
