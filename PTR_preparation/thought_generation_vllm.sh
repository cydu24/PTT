
# 定义要下载的模型路径列表
declare -a model_paths=(

)

# 定义要下载的模型名称列表
declare -a model_names=(

)

# 循环遍历模型列表并下载
for i in "${!model_paths[@]}"; do
    echo "Downloading ${model_names[$i]}..."
    python thought_generation_vllm.py \
    --model_path "${model_paths[$i]}" \
    --model_name "${model_names[$i]}" \
    --max_new_tokens 2048 \
    --prompt_paths  \
    --temperature 0.95 \
    --top_p 0.9 \
    --top_k 30 \
    --use_vllm \
    --data_len 
done


echo "All models have been downloaded successfully."
