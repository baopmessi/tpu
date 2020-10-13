
MODEL="segmentation"
IMAGE_SIZE=640
CHECKPOINT_PATH="../efficientnet-l2-nasfpn-ssl/model.ckpt"
PARAMS_OVERRIDE="../efficientnet-l2-nasfpn-ssl/pascal_seg_efficientnet-l2-nasfpn.yaml"  # if any.
LABEL_MAP_FILE="models/official/detection/datasets/coco_label_map.csv"
IMAGE_FILE_PATTERN="images/*"
OUTPUT_HTML="./test.html"
python3 models/official/detection/inference_segmentation.py \
  --model="${MODEL?}" \
  --image_size=${IMAGE_SIZE?} \
  --checkpoint_path="${CHECKPOINT_PATH?}" \
  --label_map_file="${LABEL_MAP_FILE?}" \
  --image_file_pattern="${IMAGE_FILE_PATTERN?}" \
  --output_html="${OUTPUT_HTML?}" \
  --max_boxes_to_draw=10 \
  --min_score_threshold=0.05\
  --config_file="${PARAMS_OVERRIDE?}"\
  --params_override="${PARAMS_OVERRIDE?}"

