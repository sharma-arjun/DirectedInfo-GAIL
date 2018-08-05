python -m pdb mujoco_gail.py \
  --expert_path ./h5_trajs/mujoco_trajs/normal_hopper/rebuttal_hopper_25/ \
  --state_size 11 \
  --action_size 3 \
  --history_size 1 \
  --context_size 4 \
  --batch_size 2048 \
  --num_epochs 5000 \
  --max_ep_length 999 \
  --num_expert_trajs 2 \
  --vae_checkpoint_path ./results/hopper/discrete_vae/batch_64_context_4_no_time/results/checkpoint/cp_640.pth \
  --results_dir ./results/hopper/gail/rebuttal_batch_2048_cp_640_ep_5000_num_expert_25 \
  --no-use_state_features \
  --use_goal_in_policy \
  --use_goal_in_value \
  --no-init_from_vae \
  --env-type mujoco \
  --env-name Hopper-v2 \
  --posterior_learning_rate 0.0 \
  --lambda_posterior 1.0 \
  --use_value_net \
  --optim_batch_size 256 \
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
