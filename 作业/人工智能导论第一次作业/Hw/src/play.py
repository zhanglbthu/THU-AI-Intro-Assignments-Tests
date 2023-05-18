from __future__ import print_function

from game import Board, DummyPlayer, Human, Game
from minimax import MinimaxSearchPlayer, AlphaBetaSearchPlayer, CuttingOffAlphaBetaSearchPlayer
from mcts import MCTSPlayer
from alphazero import AlphaZeroPlayer
from evaluation import get_evaluation_func


def get_player(player_name, args):
    if player_name == "DummyPlayer":
        return DummyPlayer()
    elif player_name == "Human":
        return Human()
    elif player_name == "MinimaxSearchPlayer":
        return MinimaxSearchPlayer()
    elif player_name == "AlphaBetaSearchPlayer":
        return AlphaBetaSearchPlayer()
    elif player_name == "CuttingOffAlphaBetaSearchPlayer":
        return CuttingOffAlphaBetaSearchPlayer(args.max_depth, get_evaluation_func(args.evaluation_func))
    elif player_name == "MCTSPlayer":
        return MCTSPlayer(args.c, args.n_playout)
    elif player_name == "AlphaZeroPlayer":
        return AlphaZeroPlayer(get_evaluation_func(args.evaluation_func), args.c, args.n_playout)
    else:
        raise KeyError(player_name) # raise抛出异常


def run(args):
    n = args.n_in_row
    width, height = args.width, args.height
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)
        player_1 = get_player(args.player_1, args)
        player_2 = get_player(args.player_2, args)
        # set start_player=0 for human first
        game.start_play(player_1, player_2, start_player=0, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=9, help="Width of board.")
    parser.add_argument("--height", type=int, default=9, help="Height of board.")
    parser.add_argument("--n_in_row", type=int, default=5, help="Number of pieces in a row to win.")
    parser.add_argument("--player_1", type=str, default="DummyPlayer", help="Agent of Player 1")
    parser.add_argument("--player_2", type=str, default="DummyPlayer", help="Agent of Player 2")
    parser.add_argument("--max_depth", type=int, default=1, help="Maximum search depth (CuttingOffAlphaBetaSearch only).")
    parser.add_argument("--evaluation_func", type=str, default="dummy_evaluation_func", help="Evaluation function (CuttingOffAlphaBetaSearch/AlphaZero only).")
    parser.add_argument("--c", type=float, default=1, help="Trade-off hyperparameter (MCTS/AlphaZero only).")
    parser.add_argument("--n_playout", type=int, default=5000, help="Number of playouts (MCTS/AlphaZero only).")
    args = parser.parse_args()

    run(args)
