#!/usr/bin/env bash
python -W ignore trainer/extractor.py \
	--gpu $1 \
	--max_examples -1 \
	--batch_size 32 \
	--skip_no_answer True \
	--reader_type areader \
	--nhid 256 \
	--nlayers 3 \
	--checkpoint True \
	--data_dir ../data/SQuAD/ \
	--dev_json dev-v2.0.json \
	--optfile ../../models/elmo/elmo_options.json \
	--wgtfile ../../models/elmo/elmo_weights.hdf5 \
	--embed_dir ../../glove/ \
	--embedding_file glove.840B.300d.txt \
	--model_dir ./tmp/qa_models/ \
	--model_name areader
