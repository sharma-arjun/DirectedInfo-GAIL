BATCH_SIZE=64

python -m pdb circle_world_gail.py \
  --expert_path ./h5_trajs/circle_trajs/meta_2_traj_50_multimodal_action/ \
  --state_size 2 \
  --action_size 2 \
  --history_size 1 \
  --context_size 3 \
  --goal_size 2 \
  --batch_size 1024 \
  --num_epochs 5000 \
  --max_ep_length 119 \
  --num_expert_trajs 8 \
  --vae_checkpoint_path /home/arjun/DirectedInfo-GAIL/IL/results/vae/tmp/meta_2_traj_50_multimodal_action_policy_output_normalized_context_2_goal_2/checkpoint/cp_1000.pth \
  --results_dir ./results/circle/gail/tmp/meta_2_traj_50_multimodal_action_policy_output_normalized_context_2_goal_2_posterior_lambda_0.1_history_1/ \
  --no-use_state_features \
  --no-use_goal_in_policy \
  --use_goal_in_value \
  --init_from_vae \
  --env-type circle \
  --posterior_learning_rate 0.0 \
  --lambda_posterior 0.1 \
  --use_value_net \
  --optim_batch_size 256 \
  --save_interval 100 \
  # --cuda \

# Walker
#python -m pdb mujoco_gail.py \
  #--expert_path ./h5_trajs/mujoco_trajs/normal_walker/rebuttal_walker_100/ \
  #--state_size 17 \
  #--action_size 6 \
  #--history_size 1 \
  #--context_size 4 \
  #--batch_size 1048 \
  #--num_epochs 5000 \
  #--max_ep_length 999 \
  #--num_expert_trajs 10 \
  #--vae_checkpoint_path ./results/walker/discrete_vae/batch_64_context_4_no_time/results/checkpoint/cp_440.pth \
  #--results_dir ./results/walker/gail/rebuttal_try_1 \
  #--cuda \
  #--no-use_state_features \
  #--use_goal_in_policy \
  #--use_goal_in_value \
  #--no-init_from_vae \
  #--env-type mujoco \
  #--env-name Hopper-v2 \
  #--posterior_learning_rate 0.0 \
  #--lambda_posterior 1.0 \
  #--use_value_net \
  #--optim_batch_size 256

#python -m pdb vae.py \
  #--batch-size 1 \
  #--episode_len 104 \
  #--use_rnn_goal 0 \
  #--num-epochs 1500 \
  #--vae_state_size 2 \
  #--vae_action_size 2 \
  #--vae_history_size 5 \
  #--no-use_state_features \
  #--expert-path ./h5_trajs/circle_trajs/meta_1_traj_1_len_105_inner_circle_constant_velocity/ \
  #--checkpoint_every_epoch 50 \
  #--results_dir ./results/vae/tmp/meta_1_traj_50_traj_len_180_policy_output_normalized_context_3_goal_1_history_5_epoch_1000_temperature_5_noisy_next_state_lr_1e-4_different_omega \
  #--log-interval 1 \
  #--use_separate_goal_policy 1 \
  #--use_goal_in_policy 0 \
  #--use_discrete_vae \
  #--vae_context_size 3 \
  #--vae_goal_size 1 \
  #--continuous_action \
  #--run_mode train  \
  #--env-type circle \
  #--temperature 5.0 \
  #--cuda

#python -m pdb vae.py \
  #--batch-size 1 \
  #--episode_len 104 \
  #--use_rnn_goal 0 \
  #--num-epochs 400 \
  #--vae_state_size 2 \
  #--vae_action_size 2 \
  #--vae_history_size 5 \
  #--no-use_state_features \
  #--expert-path ./h5_trajs/circle_trajs/meta_1_traj_1_len_105_inner_circle_constant_velocity/ \
  #--checkpoint_every_epoch 50 \
  #--results_dir ./results/vae/tmp/meta_1_traj_50_traj_len_180_policy_output_normalized_context_3_goal_1_history_5_epoch_1000_temperature_5_noisy_next_state_lr_1e-4_different_omega \
  #--log-interval 1 \
  #--use_separate_goal_policy 1 \
  #--use_goal_in_policy 0 \
  #--use_discrete_vae \
  #--vae_context_size 3 \
  #--vae_goal_size 1 \
  #--continuous_action \
  #--run_mode test \
  #--env-type circle \
  #--checkpoint_path ./results/vae/tmp/meta_1_traj_50_traj_len_180_policy_output_normalized_context_3_goal_1_history_5_epoch_1000_temperature_5_noisy_next_state/checkpoint/cp_800.pth \
  # --cuda

