python train.py \
  --dataset_name 'town01' \
  --delim tab \
  --d_type 'local' \
  --obs_len 10 \
  --pred_len 20 \
  --encoder_h_dim_g 16 \
  --encoder_h_dim_d 16\
  --decoder_h_dim 32 \
  --embedding_dim 16 \
  --bottleneck_dim 32 \
  --mlp_dim 64 \
  --num_layers 1 \
  --noise_dim 20 \
  --noise_type gaussian \
  --noise_mix_type global \
  --pool_every_timestep 0 \
  --l2_loss_weight 0.1 \
  --batch_norm 0 \
  --dropout 0 \
  --batch_size 64 \
  --g_learning_rate 0.001 \
  --g_steps 1 \
  --d_learning_rate 0.0001 \
  --d_steps 5 \
  --checkpoint_every 10 \
  --print_every 50 \
  --num_iterations 10000 \
  --num_epochs 200 \
  --pooling_type 'pool_net' \
  --clipping_threshold_g 1.5 \
  --best_k 20 \
  --gpu_num 1 \
  --checkpoint_name gan_test_dis \
  --restore_from_checkpoint 0
