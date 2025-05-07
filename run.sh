#! /bin/bash
set -e
source /opt/conda/etc/profile.d/conda.sh

MODELS=("meta-llama/Llama-2-7b-hf"
        "meta-llama/Llama-2-7b-chat-hf"
        "meta-llama/Llama-2-13b-hf"
        "meta-llama/Llama-2-13b-chat-hf"
        "mistralai/Mistral-7B-v0.3"
        "mistralai/Mistral-7B-Instruct-v0.3"
        "meta-llama/Meta-Llama-3-8B"
        "meta-llama/Meta-Llama-3-8B-Instruct"
        "meta-llama/Llama-3.1-8B"
        "meta-llama/Llama-3.1-8B-Instruct")

MAXSEQLENS=(4096 4096 4096 4096 32768 32768 8192 8192 131072 131072)

for i in "${!MODELS[@]}"
do
  MODEL_PATH="${MODELS[$i]}"
  MAXSEQLEN="${MAXSEQLENS[$i]}"
  OUTPUT_DIR="${MODEL_PATH//\//_}_fisher"
  python run-fisher-my.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dataset wikitext2 \
    --seqlen 2048 \
    --maxseqlen "$MAXSEQLEN" \
    --num_examples 16

  cd ./quant
  python llama_simquant.py "$MODEL_PATH" --abits 1 --nsamples 16 --seqlen 2048 --norm --nuq --fisher ../${OUTPUT_DIR} --quantize --include_sparse --sparsity-threshold 0.99 --quantizer-path quantizers_${MODEL_PATH//\//_}_1bit_qnorm.pickle
  cd ../
  rm -rf ${OUTPUT_DIR}
done

