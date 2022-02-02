Author: [**Rulan Klymentiev**](https://rklymentiev.com/)

Q-Learning is one of the most popular Reinforcement Learning models that has been used in the computational neuroscience field to model behavioral data. This app allows visual comparison of *simulated* agents' performance and the effect of specific parameters based on the modification of the Q-Learning model.

**Paradigm**: 3-armed bandit problem. Each option can bring either a reward or a punishment, based on the probability value $\scriptsize \left( P_t(\text{Punishment}) = 1 - P_t(\text{Reward}) \right)$.

**Model algorithm**:

* Initialize value function $\scriptsize Q(a_j)$ with zeros
* **for** *each step $\scriptsize t$ in episode* **do**
  * Select action $a_t$ using softmax policy $\scriptsize P(a_j = a_t) = \frac{e^{\beta Q_j}}{\sum_{j}e^{\beta Q_j}}$
  * Observe outcome $\scriptsize r_t$ (either reward or punishment)
  * **for** *each action $a_j$ in possible actions* **do**
    * **if** $\scriptsize a_j == a_t$ **do**
      * **if** $\scriptsize r_t > 0$ **do**
        * $\scriptsize Q(a_j)_{t+1} - Q(a_j)_t + \alpha_{\text{positive}} \left( R \cdot r_t - Q(a_j)_t \right)$
      * **else**
        * $\scriptsize Q(a_j)_{t+1} - Q(a_j)_t + \alpha_{\text{negative}} \left( P \cdot r_t - Q(a_j)_t \right)$
      * **end**
    * **else**
      * $\scriptsize Q(a_j)_{t+1} \leftarrow \left( 1 - \alpha \right) Q(a_j)_t$
    * **end**
  * **end**
* **end**
