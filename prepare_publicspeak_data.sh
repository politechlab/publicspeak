city="AA"
model_name="roberta-large"
lr=2e-5
epoch=7
seed=42
batch_size=4

python pipeline/transcribe_and_LLM/runner_get_llm_pred.py --city ${city}  && \
python pipeline/PLM/finetuned_one_val_out.py --model_name ${model_name} --city ${city} --lr ${lr} --epoch ${epoch} --seed ${seed} --batch_size ${batch_size} && \
python pipeline/generate_processed_data/generate_processed_data.py --city ${city} --plm_file_name ${city}"_pred_LOO_roberta.json" --city ${city}