python -m pdb mujoco_gail.py \
  --expert_path ./h5_trajs/mujoco_trajs/walker_expert_traj_1/ \
  --state_size 18 \
  --action_size 6 \
  --history_size 1 \
  --context_size 1 \
  --batch_size 9990 \
  --num_epochs 5000 \
  --max_ep_length 999 \
  --num_expert_trajs 10 \
  --vae_checkpoint_path ./results/walker/vae/epochs_1000_temp_1_0.1_batch_size_256_1_goal/checkpoint/cp_1000.pth \
  --results_dir ./results/walker/gail/try_long_try_2 \
  --cuda \
  --no-use_state_features \
  --use_goal_in_policy \
  --use_goal_in_value \
  --no-init_from_vae \
  --env-type mujoco \
  --env-name Walker2d-v2 \
  --posterior_learning_rate 0.0 \
  --lambda_posterior 0.0 \
  --use_value_net \
  --optim_batch_size 256
