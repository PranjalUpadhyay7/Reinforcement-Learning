from abc import ABC, abstractmethod
import random
import numpy as np
from src.environment import MDPEnvironment
from src.utils import Logger
random.seed(0)
class MDPSolver(ABC):
    def __init__(self, environment: MDPEnvironment, logger: Logger):
        self.env = environment
        self.config = environment.config
        self.V_t1 = {s: 0.0 for s in self.env.states}
        self.policy = {s: random.choice(self.env.actions) for s in self.env.states}
        self.iterations=0
        self.logger= logger

    def choose_action(self, state, Q_table=None, epsilon=0.0):
        """
        Selects an action using an epsilon-greedy strategy.
        If epsilon is 0.0, it acts purely greedily.
        """
        if random.random() < epsilon:
            # 1. Exploration: Pick a random action
            return random.choice(self.env.actions)
        else:
            # 2. Exploitation: Pick the best action
            if Q_table is not None:
                # Extract the dictionary of {action: value} for this specific state
                action_values = Q_table[state]
                max_val = max(action_values.values())
                
                # Retrieve all actions that share this maximum value (tie-breaking)
                best_actions = [act for act, val in action_values.items() if val == max_val]
                return random.choice(best_actions)
            else:
                # Fallback to the deterministic policy if no Q-table is provided
                return self.policy[state]
    @abstractmethod
    def solve(self):
        pass

class Valueiterations(MDPSolver):
    def solve(self):
        self.iterations=0
        while True:
            self.iterations+=1
            delta=0
            Q_iter={}
            old_v= self.V_t1.copy()
            for states in self.env.states:
                q_value=dict()
                for action in self.env.actions:
                    transitions=self.env.get_transition_dynamics(states, action)
                    q_val=0
                    for i in range(len(transitions)):
                        prob,next_state, reward, done= transitions[i]
                        q_val+=prob*reward + self.config.gamma*prob*self.V_t1[next_state]
                    q_value[action]=q_val
                Q_iter[states]=q_value
                self.V_t1[states]= max(q_value.values())
                self.policy[states]=max(q_value, key=q_value.get)
                delta = max(delta, abs(old_v[states] - self.V_t1[states]))
            self.logger.record("ValueIteration", step=self.iterations, V=self.V_t1, delta= delta, policy=self.policy) # Q= Q_iter,
            if delta<=self.config.threshold:
                break

class Policyiterations(MDPSolver):
    def solve(self):
        self.iterations=0
        while True:
            self.iterations+=1
            # eval_step=0
            while True:
                # eval_step+=1
                delta=0
                old_v= self.V_t1.copy()
                for state in self.env.states:
                    action=self.policy[state]
                    transitions=self.env.get_transition_dynamics(state, action)
                    v_val=0
                    for prob,next_state, reward, done in transitions:
                        v_val+=prob*reward + self.config.gamma*prob*self.V_t1[next_state]
                    self.V_t1[state]=v_val
                    delta = max(delta, abs(old_v[state] - self.V_t1[state]))
                # self.logger.record("PolicyIteration", self.iterations, sub_step=f"eval_{eval_step}", V=self.V_t1)
                if delta<= self.config.threshold:
                    break
        
            policy_stable=True
            Q_iter={}
            for state in self.env.states:
                old_action= self.policy[state]
                q_value=dict()
                for action in self.env.actions:
                    transitions=self.env.get_transition_dynamics(state, action)
                    q_val=0
                    for i in range(len(transitions)):
                        prob,next_state, reward, done= transitions[i]
                        q_val+=prob*reward + self.config.gamma*prob*self.V_t1[next_state]
                    q_value[action]=q_val
                Q_iter[state]=q_value
                best_action = max(q_value,key=q_value.get)
                self.policy[state]=best_action
                if old_action!= best_action:
                    policy_stable=False
            self.logger.record("PolicyIteration", step=self.iterations, V=self.V_t1, delta= delta, policy=self.policy)
            if policy_stable:
                break

class MonteCarlo(MDPSolver):
    """
    Solves MDP using Every-Visit Monte Carlo Control (Task g).
    Since we have the transition model, we can simulate episodes.
    """
    def solve(self):
        self.iterations = 0
        max_episodes = getattr(self.config, 'mc_episodes', 1000)
        epsilon = 0.2
        
        # Q-table initialization
        Q = {s: {a: 0.0 for a in self.env.actions} for s in self.env.states}
        Returns = {s: {a: [] for a in self.env.actions} for s in self.env.states}
        
        while self.iterations < max_episodes:
            self.iterations += 1
            episode = []
            
            # 1. Generate an episode using epsilon-greedy
            state = self.env.reset()
            for _ in range(100): # Max steps per episode to prevent infinite loops
                # Epsilon-greedy action selection
                action = self.choose_action(state, Q_table=Q, epsilon=epsilon)
                # if random.random() < epsilon:
                #     action = random.choice(self.env.actions)
                # else:
                #     action = self.policy[state]
                
                next_state, reward, done = self.env.step(action)
                episode.append((state, action, reward))
                
                if done:
                    break
                state = next_state
                
            # 2. Update Q-values and Policy
            G = 0
            # Iterate backwards through the episode
            for state, action, reward in reversed(episode):
                G = self.config.gamma * G + reward
                
                # Every-Visit MC update
                Returns[state][action].append(G)
                Q[state][action] = sum(Returns[state][action]) / len(Returns[state][action])
                
                # Update Value and Policy greedily
                self.policy[state] = max(Q[state], key=Q[state].get)
                self.V_t1[state] = max(Q[state].values())
            # Assuming 'G' is your total accumulated reward for the episode
            self.logger.record("MonteCarlo", self.iterations, V=self.V_t1, policy=self.policy, episode_return=G)

class TDLearning(MDPSolver):
    """
    Solves MDP using Temporal Difference TD(0) Learning for State-Values V(s).
    Follows the update rule: V(s) = V(s) - alpha * [V(s) - (R + gamma * V(s'))]
    """
    def solve(self):
        self.iterations = 0
        max_episodes = getattr(self.config, 'td_episodes', 1000)
        # alpha = getattr(self.config, 'alpha', 0.1)  # Learning rate

        epsilon = 0.2
        self.N = {s: 0 for s in self.env.states}
        while self.iterations < max_episodes:
            self.iterations += 1
            episode_return = 0
            
            # 1. Start a new episode
            state = self.env.reset()
            
            for _ in range(100): # Max steps to prevent infinite loops
                # Select action using epsilon-greedy
                # Since we don't pass a Q-table, it falls back to self.policy
                action = self.choose_action(state, Q_table=None, epsilon=epsilon)
                
                # Take step in the environment
                next_state, reward, done = self.env.step(action)
                episode_return += reward
                self.N[state] += 1
                alpha = 1.0 / self.N[state]
                # 2. TD(0) State-Value Update Rule
                # Handle terminal states where there is no next_state value
                if done:
                    td_target = reward
                else:
                    td_target = reward + self.config.gamma * self.V_t1[next_state]
                
                # V(s) <- V(s) - alpha * [V(s) - target]  (Mathematically identical to V(s) += alpha * TD_Error)
                self.V_t1[state] = self.V_t1[state] - alpha * (self.V_t1[state] - td_target)
                
                # 3. Policy Improvement (Because we are using V(s) instead of Q(s,a))
                # To act greedily with only V(s), we must use the transition dynamics 
                # to see which action leads to the best expected next state.
                best_action = self.policy[state]
                max_q = float('-inf')
                
                for a in self.env.actions:
                    transitions = self.env.get_transition_dynamics(state, a)
                    q_val = 0
                    for prob, ns, r, _ in transitions:
                        q_val += prob * (r + self.config.gamma * self.V_t1[ns])
                        
                    if q_val > max_q:
                        max_q = q_val
                        best_action = a
                        
                self.policy[state] = best_action
                
                # Check for episode termination
                if done:
                    break
                    
                # Move to next state
                state = next_state
                
            # Log the progress at the end of every episode
            self.logger.record("TDLearning", step=self.iterations, V=self.V_t1, policy=self.policy, episode_return=episode_return)

class SARSA(MDPSolver):
    def solve(self):
        self.iterations = 0
        max_episodes = getattr(self.config, 'sarsa_episodes', 1000)
        epsilon = 0.2
        # Q-table initialization
        Q = {s: {a: 0.0 for a in self.env.actions} for s in self.env.states}
        N = {s: {a: 0 for a in self.env.actions} for s in self.env.states}

        while self.iterations< max_episodes:
            self.iterations +=1
            episode_return =0 
            state = self.env.reset()
            action = self.choose_action(state, Q_table=Q, epsilon=epsilon)
            for _ in range(100):
                next_state, reward, done = self.env.step(action)
                episode_return += reward
                next_action = self.choose_action(next_state, Q_table=Q, epsilon=epsilon)
                N[state][action] += 1
                alpha = max(0.01,1.0 / N[state][action])
                if done:
                    td_target = reward
                else:
                    td_target = reward + self.config.gamma * Q[next_state][next_action]
                Q[state][action] = Q[state][action] - alpha * (Q[state][action] - td_target)

                self.policy[state] = max(Q[state], key=Q[state].get)
                self.V_t1[state] = max(Q[state].values())
                if done:
                    break
                state = next_state
                action = next_action
            self.logger.record("SARSA", step=self.iterations, V=self.V_t1, policy=self.policy, episode_return=episode_return)

class NStepSARSA(MDPSolver):
    """
    Solves MDP using Real-Time (Online) n-step SARSA.
    Updates Q-values mid-episode, trailing the agent's current position by n steps.
    """
    def solve(self):
        self.iterations = 0
        max_episodes = getattr(self.config, 'n_sarsa_episodes', 1000)
        n = getattr(self.config, 'n_sarsa', 3)  # Lookahead steps
        max_steps = 100
        epsilon = 0.2
        
        Q = {s: {a: 0.0 for a in self.env.actions} for s in self.env.states}
        N = {s: {a: 0 for a in self.env.actions} for s in self.env.states}
        
        while self.iterations < max_episodes:
            self.iterations += 1
            
            # Real-time memory buffers for the current episode
            # We initialize R with 0.0 so indices align with RL math notation (R_1 is the first reward)
            S = [self.env.reset()]
            A = [self.choose_action(S[0], Q_table=Q, epsilon=epsilon)]
            R = [0.0] 
            
            T = float('inf') # Terminal time step
            t = 0            # Current time step
            
            while True:
                # 1. Take action, observe next state and reward
                if t < T:
                    next_state, reward, done = self.env.step(A[t])
                    S.append(next_state)
                    R.append(reward)
                    
                    if done or t == max_steps - 1:
                        T = t + 1
                    else:
                        # Real-time on-policy action selection
                        next_action = self.choose_action(next_state, Q_table=Q, epsilon=epsilon)
                        A.append(next_action)
                
                # 2. tau is the step we are currently updating (it trails t by n steps)
                tau = t - n + 1
                
                if tau >= 0:
                    # Calculate n-step return G
                    G = 0.0
                    for i in range(tau + 1, min(tau + n, T) + 1):
                        G += (self.config.gamma ** (i - tau - 1)) * R[i]
                    
                    # Bootstrap if the n-step lookahead didn't hit the end of the episode
                    if tau + n < T:
                        G += (self.config.gamma ** n) * Q[S[tau + n]][A[tau + n]]
                    
                    # 3. REAL-TIME Q-TABLE UPDATE
                    update_s = S[tau]
                    update_a = A[tau]
                    
                    N[update_s][update_a] += 1
                    alpha = max(0.01, 1.0 / N[update_s][update_a])
                    
                    Q[update_s][update_a] += alpha * (G - Q[update_s][update_a])
                    
                    # Update base variables so the logger (and the agent's mid-episode greedy choices) see it immediately
                    self.V_t1[update_s] = max(Q[update_s].values())
                    self.policy[update_s] = max(Q[update_s], key=Q[update_s].get)
                
                # 4. Stop when we have successfully updated the very last step of the episode
                if tau == T - 1:
                    break
                    
                t += 1
                
            episode_return = sum(R)
            self.logger.record("NStepSARSA", step=self.iterations, V=self.V_t1, policy=self.policy, episode_return=episode_return)

class QLearningOnline(MDPSolver):
    """
    Q-Learning where the agent acts in the environment using an epsilon-greedy 
    version of its own learning Q-table.
    """
    def solve(self):
        self.iterations = 0
        max_episodes = getattr(self.config, 'q_episodes', 1000)
        epsilon = 0.2
        
        Q = {s: {a: 0.0 for a in self.env.actions} for s in self.env.states}
        N = {s: {a: 0 for a in self.env.actions} for s in self.env.states}
        
        while self.iterations < max_episodes:
            self.iterations += 1
            episode_return = 0
            
            state = self.env.reset()
            
            for _ in range(100):
                # 1. ACT: Generate action using epsilon-greedy policy derived from current Q
                action = self.choose_action(state, Q_table=Q, epsilon=epsilon)
                next_state, reward, done = self.env.step(action)
                episode_return += reward
                
                # 2. UPDATE Q: 
                N[state][action] += 1
                alpha = max(0.01, 1.0 / N[state][action])
                
                if done:
                    td_target = reward
                else:
                    td_target = reward + self.config.gamma * max(Q[next_state].values())
                
                Q[state][action] += alpha * (td_target - Q[state][action])
                
                # 3. UPDATE POLICY: Base class policy strictly tracks the greedy argmax
                self.V_t1[state] = max(Q[state].values())
                self.policy[state] = max(Q[state], key=Q[state].get)
                
                if done: break
                state = next_state
                
            self.logger.record("QLearning_Online", step=self.iterations, V=self.V_t1, policy=self.policy, episode_return=episode_return)


class QLearningOffPolicy(MDPSolver):
    """
    Pure Off-Policy Q-Learning.
    Generates data using a completely separate behavior policy (random walk),
    and strictly updates the optimal Target Policy offline.
    """
    def solve(self):
        self.iterations = 0
        max_episodes = getattr(self.config, 'q_episodes', 1000)
        
        Q = {s: {a: 0.0 for a in self.env.actions} for s in self.env.states}
        N = {s: {a: 0 for a in self.env.actions} for s in self.env.states}
        
        # Explicit Target Policy (pi_T in your pseudocode). It is strictly greedy (0 or 1).
        self.target_policy = {s: random.choice(self.env.actions) for s in self.env.states}
        
        while self.iterations < max_episodes:
            self.iterations += 1
            
            # --- PHASE 1: GENERATE EPISODE USING BEHAVIOR POLICY (pi_b) ---
            # To make it truly off-policy, our pi_b will be completely random (epsilon = 1.0)
            episode_buffer = []
            state = self.env.reset()
            episode_return = 0
            
            for _ in range(100):
                # Behavior policy: purely random (has NO IDEA what the Q-table says)
                action = random.choice(self.env.actions) 
                next_state, reward, done = self.env.step(action)
                
                episode_buffer.append((state, action, reward, next_state, done))
                episode_return += reward
                
                if done: break
                state = next_state
                
            # --- PHASE 2: LEARN OPTIMAL TARGET POLICY (pi_T) FROM EXPERIENCE ---
            for state, action, reward, next_state, done in episode_buffer:
                
                # Update Q-value
                N[state][action] += 1
                alpha = max(0.01, 1.0 / N[state][action])
                
                if done:
                    td_target = reward
                else:
                    td_target = reward + self.config.gamma * max(Q[next_state].values())
                
                Q[state][action] += alpha * (td_target - Q[state][action])
                
                # Update strict Target Policy
                # Probability is 1 for argmax, 0 otherwise.
                best_action = max(Q[state], key=Q[state].get)
                self.target_policy[state] = best_action
                
                # Sync with base class for logger
                self.V_t1[state] = max(Q[state].values())
                self.policy[state] = best_action 
                
            self.logger.record("QLearning_OffPolicy", step=self.iterations, V=self.V_t1, policy=self.policy, episode_return=episode_return)