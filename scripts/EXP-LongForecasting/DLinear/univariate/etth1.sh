if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi
 
# ETTh1, univariate results, pred_len= 24 48 96 192 336 720
# mse:0.0260348841547966, mae:0.12207671254873276
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_24 \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 24 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --feature S --learning_rate 0.005 >logs/LongForecasting/DLinear_fS_ETTh1_336_24.log
 
# mse:0.039807405322790146, mae:0.15160658955574036
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_48 \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 48 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --feature S --learning_rate 0.005 >logs/LongForecasting/DLinear_fS_ETTh1_336_48.log

# mse:0.05591415613889694, mae:0.1803210824728012
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_96 \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --feature S --learning_rate 0.005 >logs/LongForecasting/DLinear_fS_ETTh1_336_96.log

# mse:0.07117306441068649, mae:0.20417816936969757
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_192 \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --feature S --learning_rate 0.005 >logs/LongForecasting/DLinear_fS_ETTh1_336_192.log

# mse:0.09767461568117142, mae:0.24392887949943542
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_336 \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --feature S --learning_rate 0.005 >logs/LongForecasting/DLinear_fS_ETTh1_336_336.log

# mse:0.11864475905895233, mae:0.2740800678730011
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_720 \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 720 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --feature S --learning_rate 0.05 --train_epochs 5 >logs/LongForecasting/DLinear_fS_ETTh1_336_720.log

