#
#   PYTEST TESTING...
#
import pytest
from minimax import minimax, minimax_no_alphabeta
from two_player_games.games.connect_four import ConnectFour, ConnectFourMove

game = ConnectFour()

@pytest.fixture
def initial_state():
    return game.state

@pytest.fixture
def midgame_state():
    state = game.state
    moves = [ConnectFourMove(0), ConnectFourMove(1), ConnectFourMove(0), ConnectFourMove(1), ConnectFourMove(2)]
    for move in moves:
        game.make_move(move)
    return state

@pytest.fixture
def example_depth():
    return 3

@pytest.fixture
def deeper_depth():
    return 5

@pytest.fixture
def max_player_char():
    return game.get_current_player()

#region
# --- Singular Tests ---

def test_minimax_with_ab_initial(initial_state, example_depth, max_player_char):
    assert isinstance(minimax(initial_state, example_depth, max_player_char), float)

def test_minimax_without_ab_initial(initial_state, example_depth, max_player_char):
    assert isinstance(minimax_no_alphabeta(initial_state, example_depth, max_player_char), float)

def test_minimax_with_ab_midgame(midgame_state, example_depth, max_player_char):
    assert isinstance(minimax(midgame_state, example_depth, max_player_char), float)

def test_minimax_without_ab_midgame(midgame_state, example_depth, max_player_char):
    assert isinstance(minimax_no_alphabeta(midgame_state, example_depth, max_player_char), float)
#endregion

#region
# --- Benchmark Tests ---

def test_minimax_with_ab_benchmark_initial(benchmark, initial_state, deeper_depth, max_player_char):
    def setup():
        return (initial_state, deeper_depth, max_player_char), {}

    benchmark.pedantic(lambda state, depth, char: minimax(state, depth, char), setup=setup)


def test_minimax_without_ab_benchmark_initial(benchmark, initial_state, deeper_depth, max_player_char):
    def setup():
        return (initial_state, deeper_depth, max_player_char), {}

    benchmark.pedantic(lambda state, depth, char: minimax_no_alphabeta(state, depth, char), setup=setup)


def test_minimax_with_ab_benchmark_midgame(benchmark, midgame_state, deeper_depth, max_player_char):
    def setup():
        return (midgame_state, deeper_depth, max_player_char), {}

    benchmark.pedantic(lambda state, depth, char: minimax(state, depth, char), setup=setup)


def test_minimax_without_ab_benchmark_midgame(benchmark, midgame_state, deeper_depth, max_player_char):
    def setup():
        return (midgame_state, deeper_depth, max_player_char), {}

    benchmark.pedantic(lambda state, depth, char: minimax_no_alphabeta(state, depth, char), setup=setup)
#endregion