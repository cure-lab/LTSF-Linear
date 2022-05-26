# 0: default
# 1: value embedding + temporal embedding + positional embedding
# 2: value embedding + temporal embedding
# 3: value embedding + positional embedding
# 4: value embedding 

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Embedding" ]; then
    mkdir ./logs/Embedding
fi

for embed_type in 1 2 3 4
do
for model_name in Autoformer Informer Transformer
do 
for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path exchange_rate.csv \
    --model_id exchange_96_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 1 --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_exchange_rate_'$pred_len.log

  python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path electricity.csv \
      --model_id electricity_96_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --itr 1 --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_electricity_'$pred_len.log

  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id traffic_96_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 3 --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_traffic_'$pred_len.log

  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id weather_96_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 2 --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_weather_'$pred_len.log

  python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_Etth1_'$pred_len.log
  
  python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_96_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_Etth2_'$pred_len.log
  
  python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm1.csv \
      --model_id ETTm1_96_$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_Ettm1_'$pred_len.log

  python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id ETTm2_96_$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_Ettm2_'$pred_len.log
done
done
done

for embed_type in 1 2 3 4
do
for model_name in Autoformer Informer Transformer
do 
for pred_len in 24 36 48 60
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path national_illness.csv \
    --model_id ili_36_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 36 \
    --label_len 18 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_ili_'$pred_len.log
done
done
done