import sys
from minimax import get_best_move
from two_player_games.games.connect_four import ConnectFour

if __name__ == "__main__":
    try:
        depth_player1 = sys.argv[1]
    except IndexError as e:
        print(f"Please submit the depth of search for player one\nError message: {e}")
        sys.exit()
    except ValueError as e:
        print(f'Depth must be an integer\nError message: {e}')
        
        
    the_game = ConnectFour()
    init_state = the_game.state
    max_player_char = "1"
    
    while not the_game.is_finished():
        pass
    