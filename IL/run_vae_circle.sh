BATCH_SIZE=64

python -m pdb vae.py \
  --batch-size 32 \
  --episode_len 359 \
  --use_rnn_goal 0 \
  --num-epochs 400 \
  --vae_state_size 2 \
  --vae_action_size 2 \
  --vae_history_size 4 \
  --no-use_state_features \
  --expert-path ./h5_trajs/circle_trajs/meta_2_traj_40/ \
  --checkpoint_every_epoch 50 \
  --results_dir ./results/circle/vae/epoch_400_traj_40_meta_2_r_0.011_Aug_6_5_50_PM \
  --log-interval 1 \
  --use_separate_goal_policy 1 \
  --use_goal_in_policy 0 \
  --use_discrete_vae \
  --vae_context_size 3 \
  --vae_goal_size 2 \
  --continuous_action \
  --run_mode train  \
  --env-type circle
  # --cuda

#python -m pdb vae.py \
  #--num-epochs 1000 \
  #--expert-path ./h5_trajs/mujoco_trajs/normal_hopper/rebuttal_hopper_100/ \
  #--use_rnn_goal 0 \
  #--use_goal_in_policy 0 \
  #--use_separate_goal_policy 1 \
  #--use_discrete_vae \
  #--vae_state_size 11 \
  #--vae_action_size 3 \
  #--vae_goal_size 1 \
  #--vae_history_size 1 \
  #--vae_context_size 4 \
  #--no-use_state_features \
  #--continuous_action \
  #--env-name Hopper-v2 \
  #--env-type mujoco \
  #--episode_len 1000 \
  #--run_mode train \
  #--cuda \
  #--batch-size 64 \
  #--checkpoint_every_epoch 20 \
  #--results_dir ./results/hopper/discrete_vae/batch_64_context_4_no_time/results \
  # --checkpoint_path ./results/pendulum/discrete_vae/batch_64_context_4_no_time/checkpoint/cp_1000.pth \

