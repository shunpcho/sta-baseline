create_lmdb:
    python scripts/dump_frame_to_lmdb_files.py \
    	ego4d_data/v2/annotations \
    	ego4d_data/v2/clips \
    	results

download_pretrained_model:
    ego4d --output_directory="~/ego4d_data" --datasets sta_models

train_sta:
    python src/sta_baseline/run_sta.py \
    	--cfg config_yaml/SLOWFAST_32x1_8x4_R50_v2.yaml \
    	EGO4D_STA.VIDEOS_DIR ego4d_data/v2/clips \
    	EGO4D_STA.ANNOTATION_DIR ego4d_data/v2/annotations \
    	EGO4D_STA.RGB_LMDB_DIR data/clip_lmdb \
    	EGO4D_STA.OBJ_DETECTIONS ego4d_data/v2/sta_models/object_detections.json \
    	OUTPUT_DIR short_term_anticipation/models/slowfast_model/
