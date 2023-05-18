from typing import Tuple  # 元组类型
from copy import deepcopy  # 深层次拷贝，可以只修改副本，不影响原始数据
from game import State, Player

inf = 10000


class MinimaxSearchPlayer(Player):
    """
    Player based on minimax search.
    """

    def get_action(self, state: State):
        """
        An interface for recursively searching.
        """
        assert state.get_current_player() == self.player

        def minimax_search(s: State) -> Tuple:
            """
            Recursively search values of all succeeding nodes, taking maximum of children
            when current player is the agent (self.player) and minimum for opponent.

            Parameters:
                s: the current state

            Return:
                Tuple(value, action): the node value and the best action (if exists)

            Note: before you perform an action, you might need to copy the original state for in-place update.
            """
            end, winner = s.game_end()
            value, action = None, None
            if end:
                if winner == -1:
                    value = 0
                else:
                    value = (1 if winner == self.player else -1)
            else:
                # TODO
                if s.get_current_player() == self.player:
                    value = -inf
                    # 遍历动作集中的动作
                    for a in s.get_all_actions():
                        next_state = deepcopy(s)
                        next_state.perform_action(a)
                        new_value = minimax_search(next_state)[0]
                        if new_value > value:
                            value = new_value
                            action = a
                else:
                    value = inf
                    for a in s.get_all_actions():
                        next_state = deepcopy(s)
                        next_state.perform_action(a)
                        new_value = minimax_search(next_state)[0]
                        if new_value < value:
                            value = new_value
                            action = a
            return value, action
        return minimax_search(state)[1]  # return best action


class AlphaBetaSearchPlayer(Player):
    """
    Player based on alpha-beta search.
    """

    def get_action(self, state: State):
        """
        An interface for recursively searching.
        """
        assert state.get_current_player() == self.player

        def alpha_beta_search(s: State, alpha, beta):
            """
            Based on minimax search, record current maximum value of the max player (alpha)
            and current minimum value of the min player (beta), use alpha and beta to prune.

            Parameters:
                s: the current state
                alpha: the current maximum value of the max player
                beta: the current minimum value of the min player

            Return:
                Tuple(value, action): the node value and the best action (if exists)

            Note: before you perform an action, you might need to copy the original state for in-place update.
            """
            end, winner = s.game_end()
            value, action = None, None
            if end:
                if winner == -1:
                    value = 0
                else:
                    value = (1 if winner == self.player else -1)
            else:
                # TODO
                if s.get_current_player() == self.player:
                    value = -inf
                    for a in s.get_all_actions():
                        new_state = deepcopy(s)
                        new_state.perform_action(a)
                        v = alpha_beta_search(new_state, alpha, beta)[0]
                        if v > value:
                            value = v
                            action = a
                        if v >= beta:
                            break
                        alpha = max(alpha, v)
                else:
                    value = inf
                    for a in s.get_all_actions():
                        new_state = deepcopy(s)
                        new_state.perform_action(a)
                        v = alpha_beta_search(new_state, alpha, beta)[0]
                        if v < value:
                            value = v
                            action = a
                        if v <= alpha:
                            break
                        beta = min(beta, v)
            return value, action
        return alpha_beta_search(state, -inf, inf)[1]


class CuttingOffAlphaBetaSearchPlayer(Player):

    def __init__(self, max_depth, evaluation_func=None):
        """
        Player based on cutting off alpha-beta search.
        Parameters:
            max_depth: maximum searching depth. The search will stop when the depth exists max_depth.
            evaluation_func: a function taking a state as input and
                outputs the value in the current player's perspective.
        """
        super().__init__()
        self.max_depth = max_depth
        self.evaluation_func = (
            lambda s: 0) if evaluation_func is None else evaluation_func

    def evaluation(self, state: State):
        """
        Calculate the evaluation value relative to the agent player (rather than state's current player),
        i.e., take negation if the current player is opponent or do nothing else wise.
        """
        value = self.evaluation_func(state)
        if self.player != state.get_current_player():
            value = -value
        return value

    def get_action(self, state: State):
        """
        An interface for recursively searching.
        """
        assert state.get_current_player() == self.player

        def cutting_off_alpha_beta_search(s: State, d, alpha, beta):
            """
            Search for several depth and use evaluation value as cutting off.

            Parameters:
                s: the current state
                d: the remaining search depth, the search will stop when d=0
                alpha: the current maximum value of the max player
                beta: the current minimum value of the min player

            Return:
                Tuple(value, action): the node value and the best action (if exists)

            Note: before you perform an action, you might need to copy the original state for in-place update.
            """
            end, winner = s.game_end()
            value, action = None, None
            if end:
                if winner == -1:
                    value = 0
                else:
                    value = (1 if winner == self.player else -1)
            elif d == 0:
                value = self.evaluation(s)
            else:
                # TODO
                if s.get_current_player() == self.player:
                    value = -inf
                    for a in s.get_all_actions():
                        new_state = deepcopy(s)
                        new_state.perform_action(a)
                        v = cutting_off_alpha_beta_search(
                            new_state, d-0.5, alpha, beta)[0]
                        if v > value:
                            value = v
                            action = a
                        if v >= beta:
                            break
                        alpha = max(alpha, v)
                else:
                    value = inf
                    for a in s.get_all_actions():
                        new_state = deepcopy(s)
                        new_state.perform_action(a)

                        v = cutting_off_alpha_beta_search(
                            new_state, d-0.5, alpha, beta)[0]
                        if v < value:
                            value = v
                            action = a
                        if v <= alpha:
                            break
                        beta = min(beta, v)
            return value, action
        return cutting_off_alpha_beta_search(state, self.max_depth, -inf, inf)[1]
