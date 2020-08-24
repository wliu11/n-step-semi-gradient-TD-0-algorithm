import numpy as np
from policy import Policy


class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)
        does the += 
        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

"""
An implementation of the semi-gradient n-step TD algorithm as described in page 209 of Sutton & Barto
"""
def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """

    for episode in range(num_episode):
        
        state = env.reset()
        T = np.inf
        t = 0
        
        rewards = [0]
        states = [state]
        action = pi.action(state)
        
        while True:
            if t < T:
                next_state, reward, done, info = env.step(action)
                states.append(next_state)
                rewards.append(reward)
                
                if done:
                    T = t + 1
                
                else:
                    next_action = pi.action(next_state) 

            tau = t - n + 1
            
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += np.power(gamma, i - tau - 1) * rewards[i]
                    
                if tau + n < T:
                    state = states[tau + n]
                    G += np.power(gamma, n) * V(state)

                state = states[tau]
                V.update(alpha, G, state)

            t += 1

            action = next_action

            if tau == T - 1:
                break

    
