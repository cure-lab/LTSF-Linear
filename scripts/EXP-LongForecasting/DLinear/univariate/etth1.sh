# Results on Paper with code: ETTh1, univariate results, pred_len = 24 48 720
# mse:0.0260348841547966, mae:0.12207671254873276
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336'_'24 \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 24 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --feature S --learning_rate 0.005

# mse:0.039807405322790146, mae:0.15160658955574036
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336'_'48 \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 48 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --feature S --learning_rate 0.005
  
# mse:0.11864475905895233, mae:0.2740800678730011
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336'_'720 \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 720 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --feature S --learning_rate 0.05 --train_epochs 5