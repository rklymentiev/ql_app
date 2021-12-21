import numpy as np
from scipy.special import softmax


def generate_agent_data(
        alpha_rew, alpha_pun, beta, rew_sens, reward_val, punish_val,
        pun_sens, rew_prob, decay_rate,
        n_arms=3, n_trials=100, rnd_generator=None):
    """
    Simulate agent data for N-armed bandit task.

    Arguments
    ----------
    alpha_rew : float
        Learning rate for rewarding trials. Should be in a (0, 1) range.

    alpha_pun : float
        Learning rate for punishing trials. Should be in a (0, 1) range.

    beta : float
        Inverse temperature.

    rew_sens : float
        Reward sensitivity (R).

    pun_sens : float
        Punishment sensitivity (P).

    rew_prob : array_like
        Probability of reward for each arm.

    pun_prob : array_like
        Probability of punishment for each arm.

    rew_vals : array_like
        Values of reward (first element) and punishment (second element).
        For example [100, -50] means that agent receives 100 a.u. during
        the reward and looses 50 a.u. during the punishment.

    n_arms : int, default=4
        Number of arms in N-armed bandit task.

    n_trials : int, default=100
        Number of simulated trials for an agent

    Returns
    ----------
    actions : array of shape (n_trials,)
        Selected action by the agent at each trial.

    gains : array of shape (n_trials,)
        Agent's reward value at each trial.

    losses : array of shape (n_trials,)
        Agent's punishment value at each trial.

    Qs : array of shape (n_trials, n_arms)
        Value function for each arm at each trial.
    """

    if rnd_generator is None:
        rnd_generator = np.random.default_rng()

    actions = np.zeros(shape=(n_trials,), dtype=np.int32)
    gains = np.zeros(shape=(n_trials,), dtype=np.int32)
    Qs = np.zeros(shape=(n_trials+1, n_arms))

    for i in range(n_trials):

        # choose the action based of softmax function
        prob_a = softmax(beta * Qs[i, :])
        a = rnd_generator.choice(a=range(n_arms), p=prob_a)  # select the action
        # list of actions that were not selected
        a_left = list(range(n_arms))
        a_left.remove(a)

        # reward

        if rnd_generator.random() < rew_prob[i, a]:  # if arm brings reward
            r = reward_val
        else:
            r = punish_val

        # value function update for a chosen arm
        if r > 0:
            Qs[i+1, a] = Qs[i, a] + alpha_rew * (rew_sens*r - Qs[i, a])
        else:
            Qs[i+1, a] = Qs[i, a] + alpha_pun * (pun_sens*r - Qs[i, a])

        for a_l in a_left:
            Qs[i+1, a_l] = (1 - decay_rate) * Qs[i, a_l]

        # save the records
        actions[i] = a
        gains[i] = r

    return actions, gains, Qs[:n_trials, :]
