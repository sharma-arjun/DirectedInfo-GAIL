BATCH_SIZE=64

args=(
  --expert-path ./h5_trajs/mujoco_trajs/normal_hopper/rebuttal_hopper_100/

  --batch-size 64
  --num-epochs 1000
  --use_rnn_goal 0
  --use_goal_in_policy 0
  --use_separate_goal_policy 1
  --use_discrete_vae

  --vae_state_size 11
  --vae_action_size 6
  --vae_goal_size 1
  --vae_history_size 1
  --vae_context_size 8

  --no-use_state_features
  --continuous_action
  --env-name Hopper-v2
  --env-type mujoco
  --episode_len 1000
  --run_mode train

  --checkpoint_every_epoch 20

  --results_dir ./results/hopper/discrete_vae/batch_64_context_8_no_time/

  --cuda
)

echo "${args[@]}"

python -m pdb vae.py "${args[@]}"

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
