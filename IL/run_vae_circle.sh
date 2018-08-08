BATCH_SIZE=64

python3.5 -m pdb vae.py \
  --batch-size 32 \
  --episode_len 119 \
  --use_rnn_goal 0 \
  --num-epochs 1000 \
  --vae_state_size 2 \
  --vae_action_size 2 \
  --vae_history_size 4 \
  --no-use_state_features \
  --expert-path ./h5_trajs/circle_trajs/meta_2_traj_50_multimodal_action/ \
  --checkpoint_every_epoch 50 \
  --results_dir ./results/vae/tmp/meta_2_traj_50_multimodal_action_policy_output_normalized_context_4_goal_2 \
  --log-interval 1 \
  --use_separate_goal_policy 1 \
  --use_goal_in_policy 0 \
  --use_discrete_vae \
  --vae_context_size 4 \
  --vae_goal_size 2 \
  --continuous_action \
  --run_mode train  \
  --env-type circle \
  # --cuda

#python3.5 -m pdb vae.py \
#  --batch-size 32 \
#  --episode_len 119 \
#  --use_rnn_goal 0 \
#  --num-epochs 400 \
#  --vae_state_size 2 \
#  --vae_action_size 2 \
#  --vae_history_size 4 \
#  --no-use_state_features \
#  --expert-path ./h5_trajs/circle_trajs/meta_2_traj_50_multimodal_action/ \
#  --checkpoint_every_epoch 50 \
#  --results_dir ./results/vae/tmp/meta_2_traj_50_multimodal_action_policy_output_normalized_context_3_goal_2 \
#  --log-interval 1 \
#  --use_separate_goal_policy 1 \
##  --use_goal_in_policy 0 \
#  --use_discrete_vae \
#  --vae_context_size 3 \
#  --vae_goal_size 2 \
#  --continuous_action \
#  --run_mode test \
#  --env-type circle \
#  --checkpoint_path ./results/vae/tmp/meta_2_traj_50_multimodal_action_policy_output_normalized_context_3_goal_2/checkpoint/cp_1000.pth \
#  # --cuda

