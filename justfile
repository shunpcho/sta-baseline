test_lmdb:
	python scripts/dump_frame_to_lmdb_files.py \
		ego4d_data/v2/annotations \
		ego4d_data/v2/clips \
		results

test_new_lmdb:
	python scripts/dump_frame_to_lmdb_files.py \
		ego4d_data/v2/annotations \
		ego4d_data/v2/clips \
		--clip_uid clip_uids.json \
		results