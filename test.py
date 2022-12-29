from game import Game
# from player import Player
# from ai import AI
from constants import *
import cProfile # analyse runtime of each function
import pstats # show runtime of each function


def main():
  game = Game(n_players=3)
  game.run(debug=False)


if __name__ == '__main__':
  # main()
  profile = cProfile.Profile() # initialize profile
  profile.runcall(main) # run main function with given parameters, returns same as main()
  ps = pstats.Stats(profile)
  ps.sort_stats(("tottime")) # sort by total time
  ps.print_stats(100) # show 30 most time intensive function calls
