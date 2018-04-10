# DirectedInfo-GAIL

This repository contains a Pytorch implementation of an imitation learning framework modeled on GAIL that is designed to handle intra-trajectory variations in the expert demonstrations.


- Running `causal_gail` on diverse trajectories

```
python -m pdb causal_gail.py --num_epochs 1000 --state_size 2 --action_size 8 --history_size 1 --no-use_state_features --expert_path ./h5_trajs/diverse_trajs/correct_traj --results_dir ./results/gail/diverse_traj/lr_0003_expert_traj_50_batch_1024  --vae_checkpoint_path ./results/diverse_traj/results_lr_001_epoch_1000_save_freq_100_use_ht_as_goal_2/checkpoint/cp_500.pth --learning_rate 0.0003 --max_ep_length 20 --batch_size 1024 --num_expert_trajs 50
```
