from __future__ import print_function

import numpy as np

from typing import List, Tuple


class State(object):
    """A general state for two-player zero-sum game."""

    def __init__(self):
        self._players = [1, 2]
        self._current_player = None

    def reset(self):
        raise NotImplementedError

    def get_current_player(self) -> int:
        raise NotImplementedError

    def get_all_actions(self) -> List:
        raise NotImplementedError

    def perform_action(self, action):
        raise NotImplementedError

    def game_end(self) -> Tuple[bool, int]:
        raise NotImplementedError

    def get_info(self):
        return None


class Player(object):
    """A general player for two-player zero-sum game."""

    def __init__(self):
        self.player = None

    def set_player(self, p):
        self.player = p

    def get_action(self, state: State):
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__} {self.player}"


class Board(State):
    """board for the game"""

    def __init__(self, **kwargs):
        super().__init__()
        self._width = int(kwargs.get('width', 8))
        self._height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self._states = {}
        # need how many pieces in a row to win
        self._n_in_row = int(kwargs.get('n_in_row', 5))
        self._availables, self._last_move = None, None

    def move_to_location(self, move):
        h = move // self._width
        w = move % self._width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self._width + w
        if move not in range(self._width * self._height):
            return -1
        return move

    def reset(self, start_player=0):
        if self._width < self._n_in_row or self._height < self._n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self._n_in_row))
        self._current_player = self._players[start_player]  # start player
        # keep available moves in a list
        self._availables = list(range(self._width * self._height))
        self._states = {}
        self._last_move = -1

    def get_current_player(self):
        return self._current_player

    def get_all_actions(self):
        return self._availables

    def perform_action(self, action):
        self._states[action] = self._current_player
        self._availables.remove(action)
        self._current_player = (
            self._players[0] if self._current_player == self._players[1]
            else self._players[1]
        )
        self._last_move = action
        return self

    def has_a_winner(self):
        width = self._width
        height = self._height
        states = self._states
        n = self._n_in_row

        moved = list(set(range(width * height)) - set(self._availables))
        if len(moved) < self._n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self._availables):
            return True, -1
        return False, -1

    def get_info(self):
        info_dict = {
            "live_four": [
                np.array([0, 1, 1, 1, 1, 0]),
            ],
            "four": [
                np.array([0, 1, 1, 1, 1]),
                np.array([0, 1, 1, 1, 0, 1]),
                np.array([0, 1, 1, 0, 1, 1]),
                np.array([0, 1, 0, 1, 1, 1]),
                np.array([1, 1, 1, 1, 0]),
                np.array([1, 0, 1, 1, 1, 0]),
                np.array([1, 1, 0, 1, 1, 0]),
                np.array([1, 1, 1, 0, 1, 0]),
            ],
            "live_three": [
                np.array([0, 1, 1, 1, 0]),
                np.array([0, 1, 1, 0, 1, 0]),
                np.array([0, 1, 0, 1, 1, 0]),
            ],
            "three": [
                np.array([0, 1, 1, 1]),
                np.array([0, 1, 1, 0, 1]),
                np.array([0, 1, 0, 1, 1]),
                np.array([1, 1, 1, 0]),
                np.array([1, 1, 0, 1, 0]),
                np.array([1, 0, 1, 1, 0]),
            ],
            "live_two": [
                np.array([0, 1, 1, 0]),
                np.array([0, 1, 0, 1, 0]),
            ],
        }

        state = np.zeros((self._width, self._height)) # 初始化全0数组
        '''
        list(zip(*self._states.items()))
        将self._states字典中的每个键值对(棋子和它的位置)打包成一个元组
        再将所有元组打包成一个列表
        然后使用np.array()函数将该列表转换为NumPy数组
        '''
        if len(self._states) > 0:
            moves, players = np.array(list(zip(*self._states.items())))
            state[moves // self._width, moves % self._height] = players
        '''
        创建一个4维的NumPy数组 all_state，用于保存在五子棋游戏中的所有可能状态
        第一维表示棋子的方向（0表示横向，1表示纵向，2表示左上到右下斜向，3表示右上到左下斜向），
        第二维表示棋子数量（从1到5），
        第三维和第四维表示棋盘的宽度和高度
        '''
        # 4表示棋子的方向，6表示棋子数量（0到5），self._width和self._height分别表示棋盘的宽度和高度
        # 初始全为-1
        all_state = -np.ones((4, 6, self._width, self._height))
        all_state[0, 0] = state
        all_state[1, 0] = state
        all_state[2, 0] = state
        all_state[3, 0] = state
        '''
        使用循环语句和数组的切片和赋值功能，计算出所有可能的五子棋状态，
        并将它们赋值给 all_state 数组的对应位置
        '''
        for i in range(1, 6):
            all_state[0, i, :-i, :] = state[i:, :]
            all_state[1, i, :, :-i] = state[:, i:]
            all_state[2, i, :-i, :-i] = state[i:, i:]
            all_state[3, i, :-i, i:] = state[i:, :-i]

        info = {}

        for player in self._players:
            info[player] = {}
            occupied = np.zeros((4, 6, self._width, self._height), dtype=bool)
            for shape_name, shape_list in info_dict.items():
                info[player][shape_name] = 0
                for shape in shape_list:
                    match = np.all((~occupied[:, :len(shape), :, :]) & (all_state[:, :len(shape), :, :] == player * shape[None, :, None, None]), axis=1)
                    info[player][shape_name] += match.sum()
                    for d, w_0, h_0 in np.transpose(match.nonzero()):
                        for j in range(len(shape)):
                            if d == 0:
                                w, h = w_0 + j, h_0
                            elif d == 1:
                                w, h = w_0, h_0 + j
                            elif d == 2:
                                w, h = w_0 + j, h_0 + j
                            else:
                                w, h = w_0 + j, h_0 - j
                            for i in range(6):
                                if d == 0 and w >= i:
                                    occupied[0, i, w - i, h] = 1
                                if d == 1 and h >= i:
                                    occupied[1, i, w, h - i] = 1
                                if d == 2 and w >= i and h >= i:
                                    occupied[2, i, w - i, h - i] = 1
                                if d == 3 and w >= i and h + i < self._height:
                                    occupied[3, i, w - i, h + i] = 1
            max_distance = max([0.] + [abs(location // self._width - (self._height - 1) / 2)
                                       + abs(location % self._width - (self._width - 1) / 2)
                                       for location in self._states.keys() if
                                       self._states.get(location, 0) == player])
            max_distance /= ((self._height - 1) / 2 + (self._width - 1) / 2)
            info[player]["max_distance"] = max_distance

        return info


class DummyPlayer(Player):

    def get_action(self, state):
        return state.get_all_actions()[0]


class Human(Player):

    def get_action(self, state: Board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = state.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in state.get_all_actions():
            print("invalid move")
            move = self.get_action(state)
        return move


class Game(object):
    """game server"""

    def __init__(self, board: Board, **kwargs):
        self.board = board

    def graphic(self, board: Board, player1, player2):
        """Draw the board and show game info"""
        width = board._width
        height = board._height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board._states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1: Player, player2: Player, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.reset(start_player)
        p1, p2 = self.board._players #[1,2]
        player1.set_player(p1)
        player2.set_player(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.perform_action(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner
