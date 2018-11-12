BATCH_SIZE=64

args=(
  --expert-path ./h5_trajs/fetch_pick_and_place_trajs/state_with_obs_goal/fetch_500
  # --expert-path ./h5_trajs/mujoco_trajs/normal_hopper/rebuttal_hopper_100/

  --batch-size 128
  --num-epochs 5000
  --use_rnn_goal 0
  --use_goal_in_policy 0
  --use_separate_goal_policy 1
  --use_discrete_vae
  --cosine_similarity_loss_weight 0.0
  --temperature 5.0

  --vae_state_size 28
  --vae_action_size 4
  --vae_goal_size 1
  --vae_history_size 5
  --vae_context_size 2

  --no-use_state_features
  --continuous_action
  --env-name FetchPickAndPlace-v1
  --env-type mujoco
  --episode_len 50
  --run_mode test

  --checkpoint_every_epoch 100

  --results_dir ./results/fetch_pick_and_place/state_with_obs_goal_500/discrete_vae/batch_128_context_4_no_time_cos_similarity_0.0_init_5_decay_5e-4_try_1/
  --checkpoint_path ./results/fetch_pick_and_place/state_with_obs_goal_500/discrete_vae/batch_128_context_4_no_time_cos_similarity_0.0_init_5_decay_5e-4_try_1/checkpoint/cp_2000.pth

 # --checkpoint_path ./results/hopper/discrete_vae/batch_64_context_6_no_time_cos_similarity_1.0_decay_33e-4/checkpoint/cp_540.pth

  # --checkpoint_path ./results/hopper/discrete_vae/batch_64_context_6_no_time_cos_similarity_0.0_decay_33e-4/checkpoint/cp_700.pth

  # --checkpoint_path ./results/hopper/discrete_vae/batch_64_context_6_no_time_try_2_cos_similarity_0.1/checkpoint/cp_1000.pth
  # --checkpoint_path ./results/hopper/discrete_vae/batch_64_context_8_no_time_try_2/checkpoint/cp_2000.pth

  --cuda
)

echo "${args[@]}"

CUDA_VISIBLE_DEVICES=1 python -m pdb vae.py "${args[@]}"

# Walker
#python -m pdb vae.py \
  #--num-epochs 1000 \
  #--expert-path ./h5_trajs/mujoco_trajs/normal_walker/rebuttal_walker_100/ \
  #--use_rnn_goal 0 \
  #--use_goal_in_policy 0 \
  #--use_separate_goal_policy 1 \
  #--use_discrete_vae \
  #--vae_state_size 17 \
  #--vae_action_size 6 \
  #--vae_goal_size 1 \
  #--vae_history_size 1 \
  #--vae_context_size 4 \
  #--no-use_state_features \
  #--continuous_action \
  #--env-name Walker2d-v2 \
  #--env-type mujoco \
  #--episode_len 1000 \
  #--run_mode train \
  #--cuda \
  #--batch-size 64 \
  #--checkpoint_every_epoch 20 \
  #--results_dir ./results/walker_2d/discrete_vae/expert_trajs_100/batch_64_context_4_no_time/results \
