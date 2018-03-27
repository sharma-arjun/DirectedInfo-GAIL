from itertools import product

import numpy as np
import numpy.random as rn

import copy
import h5py
import pdb
import pprint
import time

def is_approx_equal(a, b, eps=1e-4):
  return abs(a - b) <= eps

class SparseTransitionProbability(object):

  @staticmethod
  def convert_transition_prob_tuple_to_list(transition_probs):
    '''Convert Transition probability tuples to list of next states.

    transition_probs: Dictionary with (state, action, next_state) as key and
        the probability as value.

    Return: Dictionary with (state, action) tuple as key and a list of next
        states that can be transitioned to from `state` using `action`. Each
        list item contains the next state and the probability to transition to
        that state.
    '''
    new_transition_probs = {}
    T, all_states = set(), set()
    # ==== Debugging ====
    for s, a, _ in transition_probs.keys():
      if (s, a) not in T:
        T.add((s, a))
      if s not in all_states:
        all_states.add(s)
    # ===================

    for s, a, next_s in transition_probs.keys():
      old_key, new_key = (s, a, next_s), (s, a)
      if new_transition_probs.get(new_key) is None:
        new_transition_probs[new_key] = []

      new_transition_probs[new_key].append((next_s, transition_probs[old_key]))

    # Verify that transition probability sums to 1 for all (s, a) pairs
    for k, v in new_transition_probs.items():
      total_prob = sum([x[1] for x in v])
      assert is_approx_equal(total_prob, 1.0), \
          "Total prob value for transition dynamics should sum to 1.0"

    return new_transition_probs

  @staticmethod
  def convert_dense_transition_prob_to_sparse(transition_prob):
    '''Convert dense transition probability matrix to sparse object.

    transition_prob: Numpy array of shape (S, A, S) where each tp[i, j, k] is
      the probability to transition from state `i` to state `k` using action
      `j`.
    '''
    T = {}
    x, y, z = np.where(transition_prob > 0.001)
    for i in range(len(x)):
      T[(x[i], y[i], z[i])] = transition_prob[x[i], y[i], z[i]]

    return SparseTransitionProbability(T,
                                       transition_prob.shape[0],
                                       transition_prob.shape[1])

  def __init__(self, transition_probs, num_states, num_actions,
               next_states_as_list=False):
    '''Initialize sparse transition probability.

    There are two ways of representing transition probabilities sparsely.
    First, transition_probs is a dict with key as (s, a, s') values and value
    as the prob(s, a, s'). Alternately, transition_prob is a dict with keys as
    (s, a) and value as list [prob(s, a, s') ...].

    transition_probs: Dictionary with (state, action, next_state) tuples as keys
        with prob value.
    num_states: Number of states in MDP. Int.
    num_actions: Number of actions in MDP. Int.
    next_states_as_list: True if `transition_probs` is a dictionary with
        key (s, a) and value [(s', p_(s,s',a)) ...]
    '''
    if next_states_as_list is False:
        self._next_states_as_list = \
            SparseTransitionProbability.convert_transition_prob_tuple_to_list(
              transition_probs)
    else:
        self._next_states_as_list = transition_probs

    self._num_states = num_states
    self._num_actions = num_actions

    # Verify transition probability values are correct or not.
    self.verify_transition_probability(self._next_states_as_list,
                                       raise_exception=True)

  def verify_transition_probability(self, transition_prob_as_list,
                                    raise_exception=False):
    '''Verify if the transition probability values are valid.

    Transition probability for every state should sum to one i.e.,
      sum(s'){p(s'|s,a) =1}
    Return: True if transition probability values are correct else False.
    '''
    # Verify that transition probability sums to 1 for all (s, a) pairs
    for k, v in transition_prob_as_list.items():
      total_prob = 0.0
      for _, p in v:
          total_prob += p

      if not is_approx_equal(total_prob, 1.0):
        if raise_exception:
          raise ValueError("Transition prob does not sum to 1.0")
        return False
    return True

  def get_next_states_with_prob(self, state, action):
    '''Get next states that we can go to.

    state: Current state we are at.
    action: Action taken at state s.

    Return: Dictionary with next_states as keys and transition probability as
        values.
    '''
    if self._next_states_as_list.get((state, action)) is None:
      # NOTE: Engineering hack. There are states that are unreachable ever
      # for those states we put the transition prob. to just those states.

      # raise ValueError("We don't know how to transition from state: {} using" \
      #    " action: {}".format(state, action))
      #
      return {state: 1.0}
    else:
      next_states_rewards = self._next_states_as_list[(state, action)]
      next_states_map = {}
      total_prob = 0.0
      for next_state, prob in next_states_rewards:
        assert next_states_map.get(next_state) is None, \
            "Cannot have next state already."

        next_states_map[next_state] = prob
        total_prob += prob

      assert is_approx_equal(total_prob, 1.0), \
          "Total transition probability should sum to 1."
      return next_states_map

  def get_next_states_as_list(self, state, action, raise_exception=True):
    '''Get next states that we can go to as a list of tuples.

    This is more optimal than calling `get_next_states_with_prob` since the
    latter creates a new map and returns the map. But this list should be
    immutable.

    Return: List of (state', prob) tuple where state' is the next state that
    can be traveled to from (state, action) with probability `prob`.
    '''
    if self._next_states_as_list.get((state, action)) is None:
      #if raise_exception:
      #  raise ValueError("Cannot go to any other state. No such state exists.")
      return []
    return self._next_states_as_list[(state, action)]

  def get_all_states_with_prob(self, state, action):
    '''Get transition probability for all states.

    state: Current state we are at.
    action: Current action taken.

    Return: Numpy array of shape (N,) where each element is the probability to
      transition to that state given (s, a).
    '''
    all_probs = np.zeros((self._num_states))
    next_state_probs_map = self.get_next_states_with_prob(state, action)

    next_states = list(next_state_probs_map.keys())
    next_probs = [next_state_probs_map[k] for k in next_states]

    all_probs[next_states] = next_probs
    assert is_approx_equal(np.sum(all_probs), 1.0), \
        "Invalid probability does not sum to 1."
    return all_probs

