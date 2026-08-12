#!/bin/bash
# Downloads the training dataset + encoding index from HuggingFace, unless
# already present (idempotent across re-runs on the same instance).

if [ ! -f "Preprocess/I(q)@L=50.h5" ] || [ ! -f "Preprocess/iq_train_set-ENCODING.sqlite3" ]; then
  hf download noshou/iq_train_set "I(q)@L=50.h5" "iq_train_set-ENCODING.sqlite3" \
    --repo-type dataset --local-dir Preprocess/
fi
