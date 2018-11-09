BATCH_SIZE=64

args=(
  # --expert_path ./h5_trajs/mujoco_trajs/normal_hopper/rebuttal_hopper_100/
  --expert_path ./h5_trajs/fetch_pick_and_place_trajs/state_with_obs_goal/fetch_500

  --state_size 28
  --action_size 4
  --history_size 5
  --context_size 2
  --batch_size 4096
  --num_epochs 10000
  --max_ep_length 50
  --num_expert_trajs 100

  --num_threads 4
  --save_interval 100

  --no-use_state_features
  --no-use_goal_in_policy
  --use_goal_in_value
  --no-init_from_vae
  --env-type mujoco
  --env-name FetchPickAndPlace-v1

  --posterior_learning_rate 0.0
  --lambda_posterior 0.1
  --use_value_net
  --optim_batch_size 256

  # --vae_checkpoint_path ./results/walker/vae/batch_64_context_4_no_time/results/checkpoint/cp_1000.pth
  # Old hopper path with context = 4
  # --vae_checkpoint_path ./results/hopper/discrete_vae/batch_64_context_4_no_time/results/checkpoint/cp_640.pth
  # --vae_checkpoint_path ./results/hopper/discrete_vae/batch_64_context_8_no_time_try_2/checkpoint/cp_2000.pth
  # --vae_checkpoint_path ./results/hopper/discrete_vae/batch_64_context_4_no_time_try_2_cos_similarity_0.1/checkpoint/cp_2000.pth

  --vae_checkpoint_path ./results/fetch_pick_and_place/state_with_obs_goal_500/discrete_vae/batch_128_context_4_no_time_cos_similarity_0.0_init_5_decay_5e-4_try_1/checkpoint/cp_5000.pth

  --results_dir ./results/fetch_pick_and_place/state_with_obs_goal_500/gail/context_2/gail_c_from_expert_and_policy_vae_cp/rebuttal_batch_4096_cp_5000_ep_5000_num_expert_100_policy_log_std_clamped_use_posterior_reward_lambda_posterior_0.1_time_Oct_8_7_00_PM

  # --checkpoint_path ./results/hopper/context_4/gail_fixed_fast_c_from_expert_only/rebuttal_batch_4096_cp_640_ep_5000_num_expert_100_policy_log_std_clamped_use_posterior_reward_lambda_posterior_0.1_time_Sept_21_7_18_PM/checkpoint/cp_1300.pth

  # --checkpoint_path ./results/hopper/context_8/gail_fixed_fast_save_25/rebuttal_batch_4096_cp_640_ep_5000_num_expert_100_policy_log_std_clamped_use_posterior_reward_lambda_posterior_0.1_time_Sept_19_14_18_PM/checkpoint/cp_1750.pth

  # --checkpoint_path ./results/hopper/gail/rebuttal_batch_4096_cp_640_ep_5000_num_expert_100_policy_log_std_clamped_use_posterior_reward_lambda_posterior_0.01_time_Aug_6_00_05_AM/checkpoint/cp_1000.pth

  # --cuda
)

echo "${args[@]}"

python -m pdb mujoco_gail_fast.py "${args[@]}"

#python -m pdb mujoco_gail_fast.py \
  #--expert_path ./h5_trajs/mujoco_trajs/normal_walker/rebuttal_walker_100/ \
  #--state_size 17 \
  #--action_size 6 \
  #--history_size 1 \
  #--context_size 4 \
  #--batch_size 4096 \
  #--num_epochs 5000 \
  #--max_ep_length 1000 \
  #--num_expert_trajs 4 \
  #--vae_checkpoint_path ./results/walker/vae/batch_64_context_4_no_time/results/checkpoint/cp_1000.pth \
  #--results_dir ./results/walker/gail/rebuttal_batch_4096_cp_1000_ep_5000_num_expert_100_policy_log_std_clamped_use_posterior_reward_0.1_time_Aug_9_06_10_PM_fast \
  #--no-use_state_features \
  #--no-use_goal_in_policy \
  #--use_goal_in_value \
  #--no-init_from_vae \
  #--env-type mujoco \
  #--env-name Walker2d-v2 \
  #--posterior_learning_rate 0.0 \
  #--lambda_posterior 0.1 \
  #--use_value_net \
  #--optim_batch_size 256 \
  ## --cuda \

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
