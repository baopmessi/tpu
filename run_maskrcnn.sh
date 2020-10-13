

MODEL="mask_rcnn"
IMAGE_SIZE=640
CHECKPOINT_PATH="detection_maskrcnn_spinenet-49/model.ckpt"
PARAMS_OVERRIDE="detection_maskrcnn_spinenet-49/spinenet49_mrcnn.yaml"  # if any.
LABEL_MAP_FILE="models/official/detection/datasets/coco_label_map.csv"
IMAGE_FILE_PATTERN="images/*"
OUTPUT_HTML="./test.html"
python3 models/official/detection/inference.py \
  --model="${MODEL?}" \
  --image_size=${IMAGE_SIZE?} \
  --checkpoint_path="${CHECKPOINT_PATH?}" \
  --label_map_file="${LABEL_MAP_FILE?}" \
  --image_file_pattern="${IMAGE_FILE_PATTERN?}" \
  --output_html="${OUTPUT_HTML?}" \
  --max_boxes_to_draw=10 \
  --min_score_threshold=0.05\
  --params_override="${PARAMS_OVERRIDE?}"

