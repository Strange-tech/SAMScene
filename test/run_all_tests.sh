#!/bin/bash
set -e
export HF_ENDPOINT="https://hf-mirror.com"
IMAGE="data/images/room.png"
OUTPUT="test/output"
DEVICE="cuda"
QWEN_KEY=""
VLM="qwen"
for arg in "$@"; do
    case $arg in
        --qwen-key=*) QWEN_KEY="${arg#*=}" ;;
        --image=*)    IMAGE="${arg#*=}" ;;
        --device=*)   DEVICE="${arg#*=}" ;;
        --vlm=*)      VLM="${arg#*=}" ;;
    esac
done
PYTHON="/mnt/sda/johnli/miniconda3/envs/difix3d/bin/python3"
SAMSCENE="/mnt/sda/johnli/SAMScene"
echo "===== CAST Pipeline ===="
echo "Image: $IMAGE"
rm -rf "$OUTPUT"; mkdir -p "$OUTPUT"
cd "$SAMSCENE"
echo ""; echo ">>>> Stage 1: Load Image <<<<"
$PYTHON test/test_01_load_image.py --image "$IMAGE" --output "$OUTPUT"
echo ""; echo ">>>> Stage 2: Scene Analysis <<<<"
$PYTHON test/test_02_scene_analysis.py --image "$IMAGE" --output "$OUTPUT"
# echo ""; echo ">>>> Stage 3: Object Generation <<<<"
# $PYTHON test/test_03_object_generation.py --output "$OUTPUT" --device "$DEVICE"
# echo ""; echo ">>>> Stage 4: Relation Graph <<<<"
# if [ -n "$QWEN_KEY" ]; then
#     if [ "$VLM" = "qwen" ]; then
#         $PYTHON test/test_04_relation_graph.py --output "$OUTPUT" --vlm qwen --qwen-key "$QWEN_KEY"
#     else
#         $PYTHON test/test_04_relation_graph.py --output "$OUTPUT" --vlm openai --openai-key "$QWEN_KEY"
#     fi
# else
#     echo "  [NOTE] No VLM key"
#     $PYTHON test/test_04_relation_graph.py --output "$OUTPUT" --vlm "$VLM"
# fi
# echo ""; echo ">>>> Stage 5: Physics Correction <<<<"
# $PYTHON test/test_05_physics_correction.py --output "$OUTPUT"
# echo ""; echo ">>>> Stage 6: Export <<<<"
# $PYTHON test/test_06_export.py --output "$OUTPUT" --save-combined
# echo ""; echo "===== All done! Results: $OUTPUT/final_scene/ ====="