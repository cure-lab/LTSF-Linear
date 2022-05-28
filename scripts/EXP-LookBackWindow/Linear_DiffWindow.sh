for pred_len in 24 720
do
for seq_len in 48 72 96 120 144 168 192 336 504 672 720
do
   python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id Electricity_$seq_len'_'$pred_len \
    --model DLinear \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len  \
    --enc_in 321 \
    --des 'Exp' \
    --itr 1 --batch_size 16  --learning_rate 0.001 >logs/LookBackWindow/DLinear_electricity_$seq_len'_'$pred_len.log

  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$seq_len'_'$pred_len \
    --model DLinear \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len  \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 --batch_size 8 >logs/LookBackWindow/DLinear_ETTh1_$seq_len'_'$pred_len.log

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Etth2.csv \
  --model_id Etth2_$seq_len'_'$pred_len \
  --model DLinear \
  --data Etth2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LookBackWindow/DLinear_Etth2_$seq_len'_'$pred_len.log

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'$pred_len \
  --model DLinear \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LookBackWindow/DLinear_exchange_rate_$seq_len'_'$pred_len.log

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'$pred_len \
  --model DLinear \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.05 >logs/LookBackWindow/DLinear_traffic_$seq_len'_'$pred_len.log

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model DLinear \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16 >logs/LookBackWindow/DLinear_weather_$seq_len'_'$pred_len.log
done
done

for pred_len in 24 720
do
for seq_len in 36 48 60 72 144 288
do
  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Ettm1.csv \
  --model_id Ettm1_$seq_len'_'$pred_len \
  --model DLinear \
  --data Ettm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0001 >logs/LookBackWindow/DLinear_Ettm1_$seq_len'_'$pred_len.log

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Ettm2.csv \
  --model_id Ettm2_$seq_len'_'$pred_len \
  --model DLinear \
  --data Ettm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LookBackWindow/DLinear_Ettm2_$seq_len'_'$pred_len.log
done
done

for pred_len in 24 60
do
for seq_len in 26 52 78 104 130 156 208
do
  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'$pred_len \
  --model DLinear \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len $pred_len  \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LookBackWindow/DLinear_ili_$seq_len'_'$pred_len.log
done
done