if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi

# ETTm1, univariate results, pred_len= 96 192 336 720
# mse:0.027964573353528976, mae:0.12343791872262955
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_336_96 \
  --model DLinear \
  --data ETTm1 \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0001 --feature S >logs/LongForecasting/DLinear_fS_ETTm1_336_96.log

# mse:0.04489120841026306, mae:0.15623459219932556
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_336_192 \
  --model DLinear \
  --data ETTm1 \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0001 --feature S >logs/LongForecasting/DLinear_fS_ETTm1_336_192.log

# mse:0.06130826100707054, mae:0.18232746422290802
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_336_336 \
  --model DLinear \
  --data ETTm1 \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0001 --feature S >logs/LongForecasting/DLinear_fS_ETTm1_336_336.log

# mse:0.07956517487764359, mae:0.20977751910686493
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_336_720 \
  --model DLinear \
  --data ETTm1 \
  --features M \
  --seq_len 336 \
  --pred_len 720 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0001 --feature S >logs/LongForecasting/DLinear_fS_ETTm1_336_720.log