BATCH_SIZE=64

args=(
  --batch_size 1024
  --num_epochs 5000
  --max_ep_length 119
  --num_expert_trajs 8

  --state_size 2
  --action_size 2
  --history_size 1
  --context_size 2
  --goal_size 2

  # Learning rates
  --learning_rate 1e-4 # Discriminator learning rate
  --gen_learning_rate 1e-4  # Generator learning rate
  --posterior_learning_rate 0.0  # Posterior learning rate

  --no-use_state_features
  --no-use_goal_in_policy
  --use_goal_in_value
  --init_from_vae

  --env-type circle

  --lambda_posterior 0.001
  --use_value_net
  --optim_batch_size 256
  --save_interval 100

  --expert_path ./h5_trajs/circle_trajs/meta_1_traj_100_opposite_circles_equal_radii

  --vae_checkpoint_path ./results/circle/vae/traj_meta_1_traj_100_opposite_circles_equal_radii_epoch_1000_batch_64_1-cos_cos_wt_ratio_50/checkpoint/cp_1000.pth
  
  --results_dir ./results/circle/gail/meta_1_traj_100_opposite_circles_equal_radii_action_policy_output_normalized_context_2_goal_1_posterior_lambda_0.1_history_1_posterior_0.001_lr_1e-4/

  # --checkpoint_path ./results/circle/gail/meta_1_traj_100_opposite_circles_equal_radii_action_policy_output_normalized_context_2_goal_1_posterior_lambda_0.1_history_1/checkpoint/cp_1200.pth
  
  # --cuda
)

echo "${args[@]}"

python -m pdb circle_world_gail.py "${args[@]}"

#python -m pdb circle_world_gail.py \
  #--expert_path ./h5_trajs/circle_trajs/meta_2_traj_50_multimodal_action/ \
  #--state_size 2 \
  #--action_size 2 \
  #--history_size 1 \
  #--context_size 3 \
  #--goal_size 2 \
  #--batch_size 1024 \
  #--num_epochs 5000 \
  #--max_ep_length 119 \
  #--num_expert_trajs 8 \
  #--vae_checkpoint_path /home/arjun/DirectedInfo-GAIL/IL/results/vae/tmp/meta_2_traj_50_multimodal_action_policy_output_normalized_context_2_goal_2/checkpoint/cp_1000.pth \
  #--results_dir ./results/circle/gail/tmp/meta_2_traj_50_multimodal_action_policy_output_normalized_context_2_goal_2_posterior_lambda_0.1_history_1/ \
  #--no-use_state_features \
  #--no-use_goal_in_policy \
  #--use_goal_in_value \
  #--init_from_vae \
  #--env-type circle \
  #--posterior_learning_rate 0.0 \
  #--lambda_posterior 0.1 \
  #--use_value_net \
  #--optim_batch_size 256 \
  #--save_interval 100 \
  ## --cuda \
