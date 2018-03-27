"""
Find the value function associated with a policy. Based on Sutton & Barto, 1998.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import pdb
import time

from .sparse_transition_prob import SparseTransitionProbability

def softmax_arr(x):
  """
  Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.

  x: numpy array.
  -> softmax(x) f(x) = log (\sum(x) e(f(x)))
  """
  max_x = np.max(x)
  return max_x + np.log(np.sum(np.exp(x - max_x)))

def value(policy,
          n_states,
          transition_probabilities,
          reward,
          discount,
          threshold=1e-2,
          post_transition_reward=True,
          **kwargs):
    """
    Find the value function associated with a policy.

    policy: List of action ints for each state.
    n_states: Number of states. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    if type(transition_probabilities) != type(np.array([])):
        # Sparse transition probability
        return value_sparse(policy, n_states, transition_probabilities,
                            reward, discount, threshold=threshold,
                            post_transition_reward=post_transition_reward,
                            **kwargs)
    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            vs = v[s]
            a = policy[s]
            if post_transition_reward:
                v[s] = sum(transition_probabilities[s, a, k] *
                           (reward[k] + discount * v[k])
                           for k in range(n_states))
            else:
                v[s] = reward[s] + sum(transition_probabilities[s, a, k]
                        * discount * v[k] for k in range(n_states))

            diff = max(diff, abs(vs - v[s]))

    return v

def value_sparse(policy,
                 n_states,
                 sparse_transition_prob,
                 reward,
                 discount,
                 threshold=1e-2,
                 post_transition_reward=True):
  """
  Find the value function associated with a policy.

  policy: List of action ints for each state.
  n_states: Number of states. int.
  transition_probabilities: Function taking (state, action, state) to
      transition probabilities.
  reward: Vector of rewards for each state.
  discount: MDP discount factor. float.
  threshold: Convergence threshold, default 1e-2. float.
  -> Array of values for each state
  """
  v = np.zeros(n_states)

  diff = float("inf")
  while diff > threshold:
    diff = 0
    for s in range(n_states):
      vs = v[s]
      a = policy[s]
      next_states_prob_dict = \
              sparse_transition_prob.get_next_states_with_prob(s, a)
      if post_transition_reward:
        total_v = 0.0
        for k, prob in next_states_prob_dict.items():
            total_v += (prob * (reward[k] + discount * v[k]))
        v[s] = total_v
      else:
        total_v = reward[s]
        for k, prob in next_states_prob_dict.items():
            total_v += (prob * discount * v[k])
        v[s] = total_v

      diff = max(diff, abs(vs - v[s]))

  return v

def optimal_value(n_states,
                  n_actions,
                  transition_prob,
                  reward,
                  discount,
                  threshold=1e-2,
                  goal_state=None,
                  softmax=False,
                  backup_V=False,
                  post_transition_reward=True,
                  **kwargs):
  """
  Find the optimal value function.

  n_states: Number of states. int.
  n_actions: Number of actions. int.
  transition_prob: Function taking (state, action, state) to transition
      probabilities.
  reward: Vector of rewards for each state.
  discount: MDP discount factor. float.
  threshold: Convergence threshold, default 1e-2. float.
  goal_state: Goal state to use for value iteration. Value of goal state is
      set to 0 for faster convergence (reward shaping).
  softmax: Use softmax value iteration if True.
  backup_V: Backup value function while doing value iteration.
  post_transition_reward: If True get reward to transition into a state in
      contrast to away from a state.
  -> Array of values for each state
  """
  if type(transition_prob) != type(np.array([])):
    return optimal_value_sparse(
        n_states,
        n_actions,
        transition_prob,
        reward,
        discount,
        threshold=threshold,
        goal_state=goal_state,
        softmax=softmax,
        backup_V=backup_V,
        post_transition_reward=post_transition_reward,
        **kwargs)

  v, v_backup = np.zeros(n_states), np.zeros(n_states)
  num_iters = 0

  diff = float("inf")
  while diff > threshold:
    diff = 0
    for s in range(n_states):
      max_v = float("-inf")
      # Cannot transition away from goal state as well as get 0 reward
      # at goal state.
      if s == goal_state:
        v_backup[s] = 0
        continue

      if softmax:
        if post_transition_reward:
          Q_a = [np.dot(transition_probs[s,a,:], reward+discount*v)
              for a in range(n_actions)]
        else:
          Q_a = [reward[s] + np.dot(transition_probs[s,a,:], discount*v)
              for a in range(n_actions)]
        max_v = softmax_arr(Q_a)
      else:
        for a in range(n_actions):
          tp = transition_prob[s, a, :]
          if post_transition_reward:
            max_v = max(max_v, np.dot(tp, reward + discount*v))
          else:
            max_v = max(max_v, reward[s] + discount * np.dot(tp, v))

      new_diff = abs(v[s] - max_v)
      if new_diff > diff:
        diff = new_diff
      # Update the value
      if backup_V:
        v_backup[s] = max_v
      else:
        v[s] = max_v
    num_iters += 1
    if backup_V:
      v = np.copy(v_backup)

  return v

def policy_evaluation(n_states,
                      n_actions,
                      transition_probabilities,
                      reward,
                      discount,
                      policy,
                      threshold=1e-2,
                      post_transition_reward=True,
                      **kwargs):
    """
    Evaluate given policy.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    policy: Callable. Returns probability of taking action `a` in state `s`.
    -> Array of values for each state
    """
    if type(transition_probabilities) != type(np.array([])):
        return policy_evaluation_sparse(
                n_states,
                n_actions,
                transition_probabilities,
                reward,
                discount,
                policy,
                threshold=threshold,
                post_transition_reward=post_transition_reward,
                **kwargs)

    v = np.zeros(n_states)
    num_iters = 0

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            curr_v = 0.0
            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                if post_transition_reward:
                    curr_v += policy(s, a) * np.dot(tp, reward + discount*v)
                else:
                    curr_v += reward[s] + policy(s, a)*discount*np.dot(tp, v)

            new_diff = abs(v[s] - curr_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = curr_v
        num_iters += 1

    return v

def policy_evaluation_sparse(n_states,
                             n_actions,
                             transition_prob,
                             reward,
                             discount,
                             policy,
                             threshold=1e-2,
                             post_transition_reward=True,
                             use_transition_prob_list=True):
    """
    Evaluate given policy.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    policy: Callable. Returns probability of taking action `a` in state `s`.
    -> Array of values for each state
    """
    v = np.zeros(n_states)
    num_iters = 0

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            curr_v = 0.0
            for a in range(n_actions):
                if not use_transition_prob_list:
                    tp = transition_prob.get_all_states_with_prob(s, a)
                    if post_transition_reward:
                        curr_v += policy(s, a) * np.dot(tp, reward + discount*v)
                    else:
                        curr_v += reward[s] + policy(s, a)*discount*np.dot(tp, v)
                else:
                    # List of (next_state, next_state prob) tuples
                    next_s_list = transition_prob.get_next_states_as_list(s, a)
                    for next_s, p in next_s_list:
                        if post_transition_reward:
                            # pi(s|a) * sum(s') (r[s'] + d * v[s'] * p(s'|s,a))
                            curr_v += (policy(s, a)
                                        * (reward[next_s]+discount*v[next_s]*p))
                        else:
                            curr_v += (reward[s]
                                        + policy(s,a)*discount*v[next_s]*p)

            new_diff = abs(v[s] - curr_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = curr_v
        num_iters += 1

    return v

def optimal_value_sparse(n_states,
                         n_actions,
                         transition_probs,
                         reward,
                         discount,
                         threshold=1e-2,
                         goal_state=None,
                         softmax=False,
                         backup_V=False,
                         post_transition_reward=True,
                         use_transition_prob_list=True):
  """
  Find the optimal value function using sparse transition matrix.

  n_states: Number of states. int.
  n_actions: Number of actions. int.
  transition_probabilities: Sparse transition probability object mapping
      (state, action, state) to transition probabilities.
  reward: Vector of rewards for each state.
  discount: MDP discount factor. float.
  threshold: Convergence threshold, default 1e-2. float.
  -> Array of values for each state
  """

  # This initialization seems wrong to me.
  v, v_backup = np.zeros(n_states), np.zeros(n_states)
  num_iters = 0
  diff = float("inf")
  while diff > threshold:
    diff = float("-inf")
    val_iter_start_time = time.time()

    for s in range(n_states):
      # Cannot transition away from goal state, as well as get 0 reward at
      # goal state.
      if s == goal_state:
        v[s], v_backup[s] = 0., 0.
        continue

      Q_s_a = []
      for a in range(n_actions):
        # Initialize to the lowest reward
        next_state_v = float("-inf")

        if not use_transition_prob_list:
          tp = transition_probs.get_all_states_with_prob(s, a)
          if post_transition_reward:
            next_state_v = np.dot(tp, reward + discount*v)
          else:
            next_state_v = reward[s] + discount*np.dot(tp, v)
        else:
          # Use next state transition probability list directly
          next_states = transition_probs.get_next_states_as_list(s, a)
          next_state_v = 0 if post_transition_reward else reward[s]
          for next_s, p in next_states:
            if post_transition_reward:
              next_state_v += p * (reward[next_s] + discount*v[next_s])
            else:
              next_state_v += p * discount * v[next_s]

        # Add to the Q-function list
        Q_s_a.append(next_state_v)

      # Update current value based on softmax or hardmax
      if softmax:
        max_v = softmax_arr(Q_s_a)
      else:
        max_v = np.max(Q_s_a)

      new_diff = abs(v[s] - max_v)
      if new_diff > diff:
        diff = new_diff
      if backup_V:
        v_backup[s] = max_v
      else:
        v[s] = max_v

    num_iters += 1
    if backup_V:
      v = np.copy(v_backup)

    val_iter_end_time = time.time()

  return v

def find_policy(n_states,
                n_actions,
                transition_probabilities,
                reward,
                discount,
                threshold=1e-2,
                v=None,
                stochastic=True,
                temperature=1.0,
                post_transition_reward=True,
                **kwargs):
    """
    Find the optimal policy.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Value function (if known). Default None.
    stochastic: Whether the policy should be stochastic. Default True.
    temperature: Temperature to use to scale policies appropriately.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """
    if type(transition_probabilities) != type(np.array([])):
        return find_policy_sparse(
                n_states,
                n_actions,
                transition_probabilities,
                reward,
                discount,
                threshold=threshold,
                v=v,
                temperature=temperature,
                stochastic=stochastic,
                post_transition_reward=post_transition_reward,
                **kwargs)

    if v is None:
        v = optimal_value(n_states,
                          n_actions,
                          transition_probabilities,
                          reward,
                          discount,
                          threshold,
                          post_transition_reward=post_transition_reward)

    if stochastic:
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = np.zeros((n_states, n_actions))
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities[i, j, :]
                Q[i, j] = p.dot(reward + discount*v)
        Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        Q = Q / temperature
        Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        return Q

    def _policy(s):
        '''
        q = [0] * n_actions
        for a in range(n_actions):
            q[a] = sum(transition_probabilities[s, a, k]
                        * (reward[k] + discount * v[k])
                        for k in range(n_states))
        '''
        if post_transition_reward:
            return max(range(n_actions),
                       key=lambda a: sum(transition_probabilities[s, a, k] *
                                         (reward[k] + discount * v[k])
                                         for k in range(n_states)))
        else:
            return max(range(n_actions),
                       key=lambda a: reward[s]
                        + sum(transition_probabilities[s, a, k]
                            * discount * v[k] for k in range(n_states)))

    policy = np.array([_policy(s) for s in range(n_states)])
    return policy

def find_policy_sparse(n_states,
                       n_actions,
                       transition_probs,
                       reward,
                       discount,
                       threshold=1e-2,
                       v=None,
                       temperature=1.0,
                       stochastic=True,
                       post_transition_reward=True,
                       use_transition_prob_list=True):
  if v is None:
    v = optimal_value_sparse(
        n_states,
        n_actions,
        transition_probs,
        reward,
        discount,
        threshold,
        post_transition_reward=post_transition_reward,
        use_transition_prob_list=use_transition_prob_list)

  if stochastic:
    Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
      for a in range(n_actions):
        if not use_transition_prob_list:
          full_p = transition_probs.get_all_states_with_prob(s, a)
          Q[s, a] = full_p.dot(reward + discount * v)
        else:
          # List of (next_state, next_state_prob) values
          next_s_p = transition_probs.get_next_states_as_list(s, a)
          q_value = 0
          for next_s, p in next_s_p:
            q_value += p * (reward[next_s] + discount * v[next_s])
          Q[s, a] = q_value

    # Normalization for numerical stability
    Q -= Q.max(axis=1).reshape((n_states, 1))
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))

    return Q

  # There should be no need to use sparse transition probability arryas for
  # deterministic transitions??
  def _policy(s):
    if post_transition_reward:
      return max(range(n_actions),
          key=lambda a: sum(prob * (reward[k] + discount * v[k])
            for k, prob in transition_probs.get_next_states_with_prob(
              s, a).items()))
    else:
      return max(range(n_actions),
          key=lambda a: reward[s] + \
              sum(prob * discount * v[k] for k, prob in \
              transition_probs.get_next_states_with_prob(s,a).items()))

  policy = np.array([_policy(s) for s in range(n_states)])
  return policy

def get_log_likelihood(policy, trajectories):
    '''Get the log likelihood for trajectories using the policy probabilities.

    policy: Stochastic policy as required by some reward function.
    trajectories: The set of trajectories to be used in estimation.
    '''
    num_states, num_actions = policy.shape
    num_traj, traj_len, _ = trajectories.shape
    total_ll = 0
    for i in range(num_traj):
        log_likelihood = 0.0
        for j in range(traj_len):
            state, action = trajectories[i, j, 0], trajectories[i, j, 1]
            state, action = int(state), int(action)
            if policy[state, action] <= 1e-4: # Prevent infinity
                log_likelihood += np.log(1e-2)
            else:
                log_likelihood += np.log(policy[state, action])
            # We are at the destination the episode should stop.
            # if state == num_states - 1: break
        total_ll += (-1.0 * log_likelihood) / (traj_len - 1.0)

    return total_ll


if __name__ == '__main__':
    # Quick unit test using gridworld.
    import mdp.gridworld as gridworld
    gw = gridworld.Gridworld(3, 0.3, 0.9)
    v = value([gw.optimal_policy_deterministic(s) for s in range(gw.n_states)],
              gw.n_states,
              gw.transition_probability,
              [gw.reward(s) for s in range(gw.n_states)],
              gw.discount)
    assert np.isclose(v,
                      [5.7194282, 6.46706692, 6.42589811,
                       6.46706692, 7.47058224, 7.96505174,
                       6.42589811, 7.96505174, 8.19268666], 1).all()
    opt_v = optimal_value(gw.n_states,
                          gw.n_actions,
                          gw.transition_probability,
                          [gw.reward(s) for s in range(gw.n_states)],
                          gw.discount)
    assert np.isclose(v, opt_v).all()
