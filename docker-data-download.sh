#!/bin/bash

# get docker to download datasets
set -e

if [ ! -f "Preprocess/I(q)@L=50.h5" ] || [ ! -f "Preprocess/iq_train_set-ENCODING.sqlite3" ]; then
	hf download noshou/iq_train_set "I(q)@L=50.h5" "iq_train_set-ENCODING.seqlite3" \
		--repo-type dataset --local-dir Preprocess/
fi

exec "$@"