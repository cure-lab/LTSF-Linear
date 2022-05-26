# ETTh1
for pred_len in 96 192 336
do
python long_range_main.py -data ETTh1 -input_size 96 -predict_step $pred_len -n_head 6 >logs/ETTh1_$pred_len.log
done
python long_range_main.py -data ETTh1 -input_size 96 -predict_step 720 -inner_size 5 -n_head 6 >logs/ETTh1_720.log

# # ETTh2
for pred_len in 96 192 336
do
python long_range_main.py -data ETTh2 -input_size 96 -predict_step $pred_len -n_head 6 >logs/ETTh2_$pred_len.log
done
python long_range_main.py -data ETTh2 -input_size 96 -predict_step 720 -inner_size 5 -n_head 6 >logs/ETTh2_720.log

# ETTm1
python long_range_main.py -data ETTm1 -data_path ETTm1.csv -input_size 96 -predict_step 96 \
 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >logs/ETTm1_96.log
python long_range_main.py -data ETTm1 -data_path ETTm1.csv -input_size 96 -predict_step 192 \
 -batch_size 16 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >logs/ETTm1_192.log
python long_range_main.py -data ETTm1 -data_path ETTm1.csv -input_size 96 -predict_step 336 \ 
 -inner_size 5 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >logs/ETTm1_336.log
python long_range_main.py -data ETTm1 -data_path ETTm1.csv -input_size 96 -predict_step 720 \
 -batch_size 16 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >logs/ETTm1_720.log

# ETTm2
python long_range_main.py -data ETTm2 -data_path ETTm2.csv -input_size 96 -predict_step 96 \
 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >logs/ETTm2_96.log
python long_range_main.py -data ETTm2 -data_path ETTm2.csv -input_size 96 -predict_step 192 \
 -batch_size 16 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >logs/ETTm2_192.log
python long_range_main.py -data ETTm2 -data_path ETTm2.csv -input_size 96 -predict_step 336 \
 -inner_size 5 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >logs/ETTm2_336.log
python long_range_main.py -data ETTm2 -data_path ETTm2.csv -input_size 96 -predict_step 720 \
 -batch_size 16 -dropout 0.2 -n_head 6 -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 >logs/ETTm2_720.log

# ILI
python long_range_main.py  -window_size [2,2,2] -data_path national_illness.csv -data ili \
-input_size 24 -predict_step 24 -n_head 6 -lr 0.00001 -d_model 256 >logs/ILI_24.log
python long_range_main.py  -window_size [2,2,2] -data_path national_illness.csv -data ili \
-input_size 24 -predict_step 36 -n_head 6 -lr 0.00001 -d_model 256 >logs/ILI_36.log
python long_range_main.py  -window_size [2,2,2] -data_path national_illness.csv -data ili \
-input_size 24 -predict_step 48 -n_head 6 -lr 0.00001 -d_model 256 >logs/ILI_48.log
python long_range_main.py  -window_size [2,2,2] -data_path national_illness.csv -data ili \
-input_size 24 -predict_step 60 -n_head 6 -lr 0.00001 -d_model 256 >logs/ILI_60.log

# Electricity
python long_range_main.py  -data_path electricity.csv -data electricity \
-input_size 96 -predict_step 96 -n_head 6 -lr 0.00001 -d_model 256  >logs/ECL_96.log
python long_range_main.py  -data_path electricity.csv -data electricity \
-input_size 96 -predict_step 192 -n_head 6 -lr 0.00001 -d_model 256 >logs/ECL_192.log
python long_range_main.py  -data_path electricity.csv -data electricity \
-input_size 96 -predict_step 336 -n_head 6 -lr 0.00001 -d_model 256 >logs/ECL_336.log
python long_range_main.py  -data_path electricity.csv -data electricity \
-input_size 96 -predict_step 720 -n_head 6 -lr 0.00001 -d_model 256 >logs/ECL_720.log

# Exchange
python long_range_main.py  -data_path exchange_rate.csv -data exchange \
-input_size 96 -predict_step 96 -n_head 6 -lr 0.00001 -d_model 256  >logs/Exchange_96.log
python long_range_main.py  -data_path exchange_rate.csv -data exchange \
-input_size 96 -predict_step 192 -n_head 6 -lr 0.00001 -d_model 256  >logs/Exchange_192.log
python long_range_main.py  -data_path exchange_rate.csv -data exchange \
-input_size 96 -predict_step 336 -n_head 6 -lr 0.00001 -d_model 256  >logs/Exchange_336.log
python long_range_main.py  -data_path exchange_rate.csv -data exchange \
-input_size 96 -predict_step 720 -n_head 6 -lr 0.00001 -d_model 256  >logs/Exchange_720.log

# Traffic
python long_range_main.py  -data_path traffic.csv -data traffic \
-input_size 96 -predict_step 96 -n_head 6 -lr 0.00001 -d_model 256 >logs/Traffic_96.log
python long_range_main.py  -data_path traffic.csv -data traffic \
-input_size 96 -predict_step 192 -n_head 6 -lr 0.00001 -d_model 256 >logs/Traffic_192.log
python long_range_main.py  -data_path traffic.csv -data traffic \
-input_size 96 -predict_step 336 -n_head 6 -lr 0.00001 -d_model 256 >logs/Traffic_336.log
python long_range_main.py  -data_path traffic.csv -data traffic \
-input_size 96 -predict_step 720  -n_head 6 -lr 0.00001 -d_model 256 >logs/Traffic_720.log

# Weather
python long_range_main.py  -data_path weather.csv -data weather \
-input_size 96 -predict_step 96 -n_head 6 -lr 0.00001 -d_model 256 >logs/Weather_96.log
python long_range_main.py  -data_path weather.csv -data weather \
-input_size 96 -predict_step 192 -n_head 6 -lr 0.00001 -d_model 256 >logs/Weather_192.log
python long_range_main.py  -data_path weather.csv -data weather \
-input_size 96 -predict_step 336 -n_head 6 -lr 0.00001 -d_model 256 >logs/Weather_336.log
python long_range_main.py  -data_path weather.csv -data weather \
-input_size 96 -predict_step 720 -n_head 6 -lr 0.00001 -d_model 256 >logs/Weather_720.log