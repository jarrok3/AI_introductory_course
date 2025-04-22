from two_player_games.games.connect_four import ConnectFourState, ConnectFourMove
import math
import random

def minimax(state : ConnectFourState, depth : int, max_player_char : str) -> float:
    if depth == 0 or state.is_finished():
        winner = state.get_winner()
        if winner and winner.char == max_player_char:
            return math.inf
        elif winner == None:
            # if came across draw
            return 0.0
        else:
            return -math.inf
    
    # Player(self, char)
    if state.get_current_player().char == max_player_char:
        best_move_val = -math.inf
        for move in state.get_moves():
            calc = minimax(state=state.make_move(move), depth=depth-1, max_player_char=max_player_char)
            best_move_val = max(best_move_val, calc)
        return best_move_val
    # Calc the optimal round for other player
    else:
        best_move_val_second_pl = math.inf
        for move in state.get_moves():
            calc = minimax(state=state.make_move(move), depth=depth-1, max_player_char=max_player_char)
            best_move_val_second_pl = min(best_move_val_second_pl, calc)
        return best_move_val_second_pl
    
def get_best_move(state: ConnectFourState, depth: int, max_player_char: str) -> ConnectFourMove:
    best_moves = []

    for move in state.get_moves():
        new_state = state.make_move(move)
        score = minimax(new_state, depth - 1, max_player_char)

        if score == math.inf:
            best_moves.append(move)

    if best_moves:
        return random.choice(best_moves)
    else:
        return random.choice(list(state.get_moves()))