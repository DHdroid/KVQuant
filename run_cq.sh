#! /bin/bash
set -e
source /opt/conda/etc/profile.d/conda.sh

# 모델 및 각 모델의 maxseqlen 설정
MODELS=("mistralai/Mistral-7B-v0.3"
        "mistralai/Mistral-7B-Instruct-v0.3"
        "meta-llama/Meta-Llama-3-8B"
        "meta-llama/Meta-Llama-3-8B-Instruct")

MAXSEQLENS=(32768 32768 8192 8192)

for i in "${!MODELS[@]}"
do
  MODEL_PATH="${MODELS[$i]}"
  MAXSEQLEN="${MAXSEQLENS[$i]}"

  OUTPUT_DIR="${MODEL_PATH//\//_}_fisher"

  # run-fisher-my.py 실행
  python run-fisher-my.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dataset wikitext2 \
    --seqlen 2048 \
    --maxseqlen "$MAXSEQLEN" \
    --num_examples 16

  # quant 디렉터리로 이동 후 quant 스크립트 실행
  cd quant
  conda activate kvquant
  python llama_simquant.py "$MODEL_PATH" \
    --num_coupled 4 \
    --abits 9 \
    --nsamples 16 \
    --seqlen 2048 \
    --nuq \
    --fisher ../${OUTPUT_DIR} \
    --quantize \
    --include_sparse \
    --sparsity-threshold 0.99 \
    --quantizer-path "quantizers_${MODEL_PATH//\//_}_c4b9.pickle"

  python llama_simquant.py "$MODEL_PATH" \
    --num_coupled 8 \
    --abits 10 \
    --nsamples 16 \
    --seqlen 2048 \
    --nuq \
    --fisher ../${OUTPUT_DIR} \
    --quantize \
    --include_sparse \
    --sparsity-threshold 0.99 \
    --quantizer-path "quantizers_${MODEL_PATH//\//_}_c8b10.pickle"

  # 상위 디렉터리로 복귀
  cd ../
  rm -rf ${OUTPUT_DIR}
done
