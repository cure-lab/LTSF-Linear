if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi

# ETTh2, univariate results, pred_len= 24 48 96 192 336 720
# mse:0.06645292043685913, mae:0.19250580668449402
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_24 \
  --model DLinear \
  --data ETTh2 \
  --features M \
  --seq_len 336 \
  --pred_len 24 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 --feature S >logs/LongForecasting/DLinear_fS_ETTh2_336_24.log

# mse:0.09506811946630478, mae:0.23489701747894287
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_48 \
  --model DLinear \
  --data ETTh2 \
  --features M \
  --seq_len 336 \
  --pred_len 48 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 --feature S >logs/LongForecasting/DLinear_fS_ETTh2_336_48.log

# mse:0.13114780187606812, mae:0.2793424427509308
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_96 \
  --model DLinear \
  --data ETTh2 \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 --feature S >logs/LongForecasting/DLinear_fS_ETTh2_336_96.log

# mse:0.17573289573192596, mae:0.3286595344543457
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_192 \
  --model DLinear \
  --data ETTh2 \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 --feature S >logs/LongForecasting/DLinear_fS_ETTh2_336_192.log

# mse:0.20884062349796295, mae:0.3673829436302185
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_336 \
  --model DLinear \
  --data ETTh2 \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 --feature S >logs/LongForecasting/DLinear_fS_ETTh2_336_336.log
 
# mse:0.27568498253822327, mae:0.4257437288761139 
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_720 \
  --model DLinear \
  --data ETTh2 \
  --features M \
  --seq_len 336 \
  --pred_len 720 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 --feature S >logs/LongForecasting/DLinear_fS_ETTh2_336_720.log