if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi

# ETTm2, univariate results, pred_len= 24 48 96 192 336 720
# mse:0.06312712281942368, mae:0.18283361196517944
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_96 \
  --model DLinear \
  --data ETTm2 \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 --feature S >logs/LongForecasting/DLinear_fS_ETTm2_336_96.log

# mse:0.09205285459756851, mae:0.22704465687274933
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_192 \
  --model DLinear \
  --data ETTm2 \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 --feature S >logs/LongForecasting/DLinear_fS_ETTm2_336_192.log

# mse:0.11925001442432404, mae:0.26121848821640015
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_336 \
  --model DLinear \
  --data ETTm2 \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01 --feature S >logs/LongForecasting/DLinear_fS_ETTm2_336_336.log

# mse:0.17542317509651184, mae:0.3202616274356842
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_720 \
  --model DLinear \
  --data ETTm2 \
  --features M \
  --seq_len 336 \
  --pred_len 720 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01 --feature S >logs/LongForecasting/DLinear_fS_ETTm2_336_720.log