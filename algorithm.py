import numpy as np
from pettingzoo.classic import leduc_holdem_v4
from utils import is_action_node, discountRegret, policy_update, tree_traversal, regret_matching, get_infoset_key
env = leduc_holdem_v4.env(render_mode="human")
env.reset(seed=42)

iter = 10000
discountRate = 0.0
updateInterval = 100
discountInterval = 20
history = []
regret_table = {}



class Node:
    def __init__(self, agent, observation, reward, terminated, truncated, info, history, children, parent):
        self.observation = observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
        self.history = history
        self.children = children
        self.parent=parent
        self.agent = agent

init_obs, init_reward, init_term, init_trun, init_info = env.last()   
root = Node(env.agent_selection, init_obs, init_reward, init_term, init_trun, init_info, [], {}, None)

def outcome_sampling():
    init_obs, init_reward, init_term, init_trun, init_info = env.last()   
    root = Node(env.agent_selection, init_obs, init_reward, init_term, init_trun, init_info, [], {})
    while iter < 1000:
        curr = root
        while not curr.terminated:
            curr_next = tree_policy(curr.history, curr)
            curr = curr_next
            if is_action_node(curr):
                current_player = curr.agent
                opponent = "player_1" if current_player == "player_0" else "player_0"
                env.agent_selection=opponent
        

        u = curr.reward
        back_up(curr, u)

    #TODO fix iteration over regret_table
    if iter == updateInterval:
        for info_set in regret_table:
            policy_update(info_set)
        tree_traversal(curr)

    if iter == discountInterval:
        for info_set in regret_table:
            discountRegret(info_set, discountRate)


def tree_policy(h, v):
    curr = v
    while not curr.terminated and not curr.truncated:
        mask = curr.observation["action_mask"]
        a = env.action_space(agent=curr.agent).sample(mask)
        curr = expand(curr, a)
        h.append(a)
    return curr
    

def expand(v, a):
    if a not in v.children:
        env.step(a)
        obs, reward, term, trunc, info = env.last()
        child = Node(env.agent_selection, obs, reward, term, trunc, info, v.history+[a], {}, v)
        v.children[a] = child
        return child
    else:
        env.step(a)
        return v.children[a]
    

#TODO correct backprop
def back_up(v):
    while v != root:
        if is_action_node(v):
            traverser = v.agent
            info_set = get_infoset_key(v)
            actions = env.legal_actions
            sigma = regret_matching(regret_table[info_set])
            sampled_action = v.history[-1]

            v_I = v.reward

            for a in actions:
                if a == sampled_action:
                    regret = 0
                else:
                    regret = 0 - v_I

                regret_table[info_set] += regret

        v = v.parent
