import random

import copy
import numpy as np
from game import State, Player

from copy import deepcopy


class TreeNode(object):
    """
    A node in the MCTS tree. Each node keeps track of its total utility U, and its visit-count n_visit.
    """

    def __init__(self, parent, state: State):
        """
        Parameters:
            parent (TreeNode | None): the parent node of the new node.
            state (State): the state corresponding to the new node.
        """
        self.parent = parent
        self.actions = deepcopy(state.get_all_actions())  # a list of all actions
        self.children = {}  # a map from action to TreeNode
        self.n_visits = 0
        self.U = 0  # total utility

    def expand(self, action, next_state):
        """
        Expand tree by creating a new child.

        Parameters:
            action: the action taken to achieve the child.
            next_state: the state corresponding to the child.
        """
        # TODO
        child = TreeNode(parent= self,state= next_state)
        self.children[action] = child
        child.n_visits = 0
        child.U = 0

    def get_ucb(self, c):
        """Calculate and return the ucb value for this node in the parent's perspective.
        It is a combination of leaf evaluations U/N and the ``uncertainty'' from the number
        of visits of this node and its parent.
        Note that U/N is in this node's perspective, so a negation is required.

        Parameters:
            c: the trade-off hyperparameter.
        """
        # TODO
        if self.n_visits == 0:
            return float("inf")
        exploitation = -self.U / self.n_visits  
        exploration = c * np.sqrt(np.log(self.parent.n_visits) / self.n_visits)  
        ucb = exploitation + exploration
        return ucb

    def select(self, c):
        """Select action among children that gives maximum UCB value.

        Parameters:
            c: the hyperparameter in the UCB value.

        Return: A tuple of (action, next_node)
        """
        # TODO
        if len(self.children) == 0:
            return None, self
        max_ucb = -float("inf")
        for action, child in self.children.items():
            ucb = child.get_ucb(c)
            if ucb > max_ucb:
                max_ucb, argmax = ucb, (action, child)
        return argmax[0], argmax[1]

    def update(self, leaf_value):
        """
        Update node values from leaf evaluation.

        Parameters:
            leaf_value: the value of subtree evaluation from the current player's perspective.
        """
        # TODO
        self.n_visits += 1
        self.U += leaf_value
        if self.parent is not None:
            self.parent.update(-leaf_value)

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_unexpanded_actions(self):
        return list(set(self.actions) - set(self.children.keys()))


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, start_state: State, c=5, n_playout=10000):
        """
        Parameters:
            c: the hyperparameter in the UCB value.
            n_playout: the number of total playouts.
        """
        self.start_state = start_state
        self.root = TreeNode(None, start_state)
        self.c = c
        self.n_playout = n_playout

    def playout(self, state: State):
        """
        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self.root
        while not state.game_end()[0]:
            unexpanded_actions = node.get_unexpanded_actions()
            if len(unexpanded_actions) > 0:
                action = random.choice(unexpanded_actions)
                state.perform_action(action)
                node.expand(action, state)
                node = node.children[action]
                break
            else:
                action, node = node.select(self.c)
                state.perform_action(action)

        leaf_value = self.get_leaf_value(state)
        node.update_recursive(leaf_value)

    def get_leaf_value(self, state: State):
        """
        Randomly playout until the end of the game, returning +1 if the current
        player wins, -1 if the opponent wins, and 0 if it is a tie.

        Note: the value should be under the perspective of state.get_current_player()
        """
        # TODO
        value = None
        while not state.game_end()[0]:
            action = random.choice(state.get_all_actions())
            state.perform_action(action)
        winner = state.game_end()[1]
        current_player = state.get_current_player()
        if winner == -1:
            value = 0
        else:
            value = (1 if winner == current_player else -1)
        return value

class MCTSPlayer(Player):
    """AI player based on MCTS"""
    def __init__(self, c=5, n_playout=2000):
        super().__init__()
        self.c_puct = c
        self.n_playout = n_playout

    def get_action(self, state: State):
        mcts = MCTS(state, self.c_puct, self.n_playout)
        for n in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            mcts.playout(state_copy)
        return max(mcts.root.children.items(),
                   key=lambda act_node: act_node[1].n_visits)[0]
