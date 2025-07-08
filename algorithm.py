import numpy as np
from pettingzoo.classic import leduc_holdem_v4
from utils import is_action_node, discountRegret, policy_update, tree_traversal, regret_matching, get_infoset_key
import matplotlib.pyplot as plt
from tqdm import trange
from visualizer import plot_regret

env = leduc_holdem_v4.env(render_mode=None)
env.reset(seed=42)

iter_max = 10000
discountRate = 0.0
updateInterval = 100
discountInterval = 20
history = []
regret_table = {}  # (infoset -> action -> cumulative regret)
regret_history = []

class Node:
    def __init__(self, agent, observation, reward, terminated, truncated, info, history, children, parent):
        self.observation = observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
        self.history = history
        self.children = children
        self.parent = parent
        self.agent = agent

# initial root node
init_obs, init_reward, init_term, init_trun, init_info = env.last()
root = Node(env.agent_selection, init_obs, init_reward, init_term, init_trun, init_info, [], {}, None)

eval_interval = 1000
num_eval_games = 1000
mean_rewards = []
exploitabilities = []
mean_rewards_vs_random = []
eval_points = []

def evaluate_policy(policy_func, num_games=1000):
    total_reward = 0
    for _ in range(num_games):
        env.reset()
        obs, reward, term, trunc, info = env.last()
        done = False
        while not term and not trunc:
            agent = env.agent_selection
            mask = obs["action_mask"]
            # Use regret-matching policy for current infoset
            info_set = get_infoset_key(Node(agent, obs, reward, term, trunc, info, [], {}, None))
            if info_set in regret_table:
                sigma = regret_matching(regret_table[info_set])
                actions = [a for a, flag in enumerate(mask) if flag]
                probs = [sigma.get(a, 0) for a in actions]
                if sum(probs) == 0:
                    probs = [1/len(actions)] * len(actions)
                action = np.random.choice(actions, p=np.array(probs)/sum(probs))
            else:
                actions = [a for a, flag in enumerate(mask) if flag]
                action = np.random.choice(actions)
            env.step(action)
            obs, reward, term, trunc, info = env.last()
        total_reward += reward
    return total_reward / num_games

def evaluate_vs_random(policy_func, num_games=1000):
    total_reward = 0
    for _ in range(num_games):
        env.reset()
        obs, reward, term, trunc, info = env.last()
        while not term and not trunc:
            agent = env.agent_selection
            mask = obs["action_mask"]
            if agent == "player_0":
                # Use learned policy
                info_set = get_infoset_key(Node(agent, obs, reward, term, trunc, info, [], {}, None))
                if info_set in regret_table:
                    sigma = regret_matching(regret_table[info_set])
                    actions = [a for a, flag in enumerate(mask) if flag]
                    probs = [sigma.get(a, 0) for a in actions]
                    if sum(probs) == 0:
                        probs = [1/len(actions)] * len(actions)
                    action = np.random.choice(actions, p=np.array(probs)/sum(probs))
                else:
                    actions = [a for a, flag in enumerate(mask) if flag]
                    action = np.random.choice(actions)
            else:
                # Random bot
                action = env.action_space(agent).sample(mask)
            env.step(action)
            obs, reward, term, trunc, info = env.last()
        total_reward += reward  # reward is from player_0's perspective
    return total_reward / num_games

def compute_exploitability_proxy():
    # Sum of positive regrets across all infosets and actions
    total = 0
    for info_set in regret_table:
        total += sum(max(r, 0) for r in regret_table[info_set].values())
    return total

def outcome_sampling():
    for i in trange(iter_max, desc="MCCFR Iterations"):
        if i % updateInterval == 0:
            total_regret = 0
            for info_set in regret_table:
                total_regret += sum(abs(r) for r in regret_table[info_set].values())
            regret_history.append(total_regret)
        env.reset(seed=42)  # added: reset env each iteration for fresh trajectory
        init_obs, init_reward, init_term, init_trun, init_info = env.last()
        root = Node(env.agent_selection, init_obs, init_reward, init_term, init_trun, init_info, [], {}, None)

        curr = root
        while not curr.terminated:
            curr_next = tree_policy(curr.history, curr)
            curr = curr_next
            if is_action_node(curr):
                current_player = curr.agent
                opponent = "player_1" if current_player == "player_0" else "player_0"
                env.agent_selection = opponent  

        u = {"player_0": curr.reward, "player_1": -curr.reward}  # added: utility for both players

        back_up(curr, u) 

        if i % updateInterval == 0:
            for info_set in regret_table:
                policy_update(info_set)
            tree_traversal(root)  

        if i % discountInterval == 0:
            for info_set in regret_table:
                discountRegret(regret_table[info_set], discountRate)

        
        if (i + 1) % eval_interval == 0:
            mean_reward = evaluate_vs_random(regret_matching, num_eval_games)
            mean_rewards_vs_random.append(mean_reward)
            eval_points.append(i + 1)

def tree_policy(h, v):
    curr = v
    while not curr.terminated and not curr.truncated:
        mask = curr.observation["action_mask"]
        a = env.action_space(agent=curr.agent).sample(mask)  # MC rollout
        curr = expand(curr, a)
        h.append(a)
    return curr

def expand(v, a):
    if a not in v.children:
        env.step(a)
        obs, reward, term, trunc, info = env.last()
        child = Node(env.agent_selection, obs, reward, term, trunc, info, v.history + [a], {}, v)
        v.children[a] = child
        return child
    else:
        env.step(a)
        return v.children[a]

def back_up(v, u):  # changed: added u param for sampled utility
    while v is not None and v != root:
        if is_action_node(v):
            traverser = v.agent
            info_set = get_infoset_key(v)
            actions = [i for i, flag in enumerate(v.observation["action_mask"]) if flag]  

            if info_set not in regret_table:
                regret_table[info_set] = {a: 0.0 for a in actions}  

            sigma = regret_matching(regret_table[info_set])
            sampled_action = v.history[-1]

            v_I = u[traverser]

            for a in actions:
                if a not in regret_table[info_set]:
                    regret_table[info_set][a] = 0.0
                if a == sampled_action:
                    regret = 0
                else:
                    regret = 0 - v_I  

                regret_table[info_set][a] += regret  # changed: regret update

        v = v.parent