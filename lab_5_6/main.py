import sys
from minimax import get_best_move as gbm
from two_player_games.games.connect_four import ConnectFour
import random
import time

class Simulation():
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
                n_depth_move = gbm(self.the_game.state, self.depth_max_player, self.max_player_char)
                self.the_game.make_move(n_depth_move)
            
            else:
                m_depth_move = gbm(self.the_game.state, self.depth_min_player, self.min_player_char)
                self.the_game.make_move(m_depth_move)
    
    def result(self):
        winner = self.the_game.get_winner()
        return winner.char if winner else "draw"    
    
    def run_multiple(self, amount=100) -> dict:
        results  = {
            "1" : 0,
            "2" : 0,
            "draw" : 0
        }
        
        for _ in range(amount):
            sim = Simulation(self.depth_max_player,self.depth_min_player)
            sim.run()
            results[sim.result()] += 1
        
        return results
            
        

if __name__ == "__main__":
    try:
        depth_player1 = int(sys.argv[1])
        depth_player2 = int(sys.argv[2])
        game_amount = int(sys.argv[3])
    except IndexError as e:
        print(f"Please submit the depth of search for player one and two\nError message: {e}")
        sys.exit()
    except ValueError as e:
        print(f'Depth must be an integer\nError message: {e}')

    sim_connectfour = Simulation(depth_player1,depth_player2)
    if game_amount == 1:
        sim_connectfour.run()
        print(sim_connectfour.result())
    if game_amount >=1:
        start_time = time.time()
        res = sim_connectfour.run_multiple(game_amount)
        elapsed_time = time.time() - start_time
        with open(f"n_against_m_results\\{depth_player1}_against_{depth_player2}_{game_amount}.txt", "w") as f:
            for k,val in res.items():
                f.write(f"{str(val)}\n")
            f.write(str(elapsed_time))
    
