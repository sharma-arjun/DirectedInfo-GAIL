BATCH_SIZE=64

args=(
  --batch-size 64
  --episode_len 119
  --num-epochs 1000
  --use_rnn_goal 0

  --vae_state_size 2
  --vae_action_size 2
  --vae_history_size 5
  --vae_context_size 2 
  --vae_goal_size 1

  # Train policy params
  --use_separate_goal_policy 1
  --use_goal_in_policy 0
  --use_discrete_vae

  # loss
  --cosine_loss_for_context_weight 1.0

  --no-use_state_features

  --expert-path ./h5_trajs/circle_trajs/meta_1_traj_100_opposite_circles_equal_radii 

  --results_dir ./results/circle/vae/traj_meta_1_traj_100_opposite_circles_equal_radii_epoch_1000_batch_64

  # --checkpoint_path ./results/vae/tmp/meta_1_traj_50_traj_len_180_policy_output_normalized_context_3_goal_1_history_5_epoch_1000_temperature_5_noisy_next_state/checkpoint/cp_800.pth \

  --checkpoint_every_epoch 50
  --log-interval 1

  # Env info
  --env-type circle
  --continuous_action

  # Train config
  --run_mode train
  # --cuda
)

echo "${args[@]}"

python -m pdb vae_ms.py "${args[@]}"

#python -m pdb vae.py \
  #--batch-size 64 \
  #--episode_len 119 \
  #--use_rnn_goal 0 \
  #--num-epochs 1500 \
  #--vae_state_size 2 \
  #--vae_action_size 2 \
  #--vae_history_size 5 \
  #--no-use_state_features \
  #--expert-path ./h5_trajs/circle_trajs/meta_2_traj_100_opposite_circles/ \
  #--checkpoint_every_epoch 50 \
  #--results_dir ./results/vae/tmp_opposite_circles/meta_2_traj_100_traj_len_120_policy_output_normalized_context_4_goal_1_history_5_epoch_1500_temperature_5_noisy_next_state_lr_1e-3 \
  #--log-interval 1 \
  #--use_separate_goal_policy 1 \
  #--use_goal_in_policy 0 \
  #--use_discrete_vae \
  #--vae_context_size 2 \
  #--vae_goal_size 1 \
  #--continuous_action \
  #--run_mode train  \
  #--env-type circle \
  #--temperature 5.0 \
  # --cuda

