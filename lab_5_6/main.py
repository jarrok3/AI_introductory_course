import sys
from minimax import get_best_move as gbm
from two_player_games.games.connect_four import ConnectFour
import random

class simulation():
    def __init__(self, 
                 depth_max_player : int,depth_min_player : int, max_player_char : str = "1", min_player_char : str = "2"):
        self.depth_max_player = depth_max_player
        self.depth_min_player = depth_min_player
        self.max_player_char = max_player_char
        self.min_player_char = min_player_char
        self.the_game = ConnectFour()
    
    def run(self):
        while not self.the_game.is_finished():
            if self.the_game.get_current_player() == self.the_game.first_player:
                n_depth_move = gbm(self.the_game.state, depth_player1, self.max_player_char)
                self.the_game.make_move(n_depth_move)
            
            else:
                moves = self.the_game.get_moves()
                move = random.choice(moves)
                self.the_game.make_move(move)
    
    def result(self):
        winner = self.the_game.get_winner()
        return winner.char if winner else "draw"    
    
        

if __name__ == "__main__":
    try:
        depth_player1 = int(sys.argv[1])
        depth_player2 = int(sys.argv[2])
    except IndexError as e:
        print(f"Please submit the depth of search for player one and two\nError message: {e}")
        sys.exit()
    except ValueError as e:
        print(f'Depth must be an integer\nError message: {e}')
    
    sim_connectfour = simulation(depth_player1,depth_player2)
    sim_connectfour.run()
    print(sim_connectfour.result())
