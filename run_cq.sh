#! /bin/bash
set -e
source /opt/conda/etc/profile.d/conda.sh

MODELS=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-7b-chat-hf")

for MODEL_PATH in "${MODELS[@]}"
do
  cd gradients
  conda activate grad
  OUTPUT_DIR="${MODEL_PATH//\//_}_fisher"

  # maxseqlen ~D| ~U
  if [ "$MODEL_PATH" == "meta-llama/Llama-2-7b-hf" ]; then
    MAXSEQLEN=4096
  elif [ "$MODEL_PATH" == "meta-llama/Llama-2-13b-hf" ]; then
    MAXSEQLEN=4096
  else
    MAXSEQLEN=4096  # ____~R
  fi

  python run-fisher.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dataset wikitext2 \
    --seqlen 2048 \
    --maxseqlen "$MAXSEQLEN" \
    --num_examples 16

  cd ../quant
  conda activate kvquant
  python llama_simquant.py "$MODEL_PATH" --num_coupled 4 --abits 8 --nsamples 16 --seqlen 2048 --nuq --fisher ../gradients/${OUTPUT_DIR} --quantize --include_sparse --sparsity-threshold 0.99 --quantizer-path quantizers_${MODEL_PATH//\//_}.pickle
  cd ../
done
