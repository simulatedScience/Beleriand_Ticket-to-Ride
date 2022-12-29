from constants import *
from copy import deepcopy
from random import shuffle
from game import Game

def test_game():
  #try creating test game
  _test_config = DEFAULT_GAME_CONFIG
  _test_config['memory'] = 2

  _test_game = Game(n_players=3, config=_test_config)

  _test_game.run(debug=False)

  return _test_config, _test_game

  # for player in test_game.players:
  #   print(player)
  #   print(player.trains)

def test_model_training(_test_game):
  # test model training
  # _test_game.players[0].ai.train_win()
  _test_game.players[0].ai.train_q()
  print("finished training")

def test_new_game(_test_config, _test_game):
  # try new game with existing players\
  players = _test_game.players
  p0 = players[0]
  p1 = players[1]
  p2 = players[2]
  player_wins = [[0,0,0] for _ in range(5)]

  for n in range(5):
      print(f'------N={n}--------')
      for i in range(100):
          if i % 50 == 0:
              if i > 0:
                  # p1.ai.train_win(20)
                  p1.ai.train_q(20)
                  # p2.ai.train_win()
                  p2.ai.train_q()
          print(i)
          shuffle(players)
          game = Game(pre_existing_players = players, config=_test_config)
          game.run(False)
          print(game)
          player_wins[n][0]+= p0.win
          player_wins[n][1]+= p1.win
          player_wins[n][2]+= p2.win
          for player in game.players:
              print(player)
      print('extended training...')
      # p1.ai.train_win(2)
      p1.ai.train_q(2)
      # p2.ai.train_win(2)
      p2.ai.train_q(2)
      print('resetting history...')
      p1.ai.reset_history()
      p2.ai.reset_history()
      print(f"{player_wins = }")
      p0.ai.save_models('test_000_p0_%s.h5')
      p1.ai.save_models('test_000_p1_%s.h5')
      p2.ai.save_models('test_000_p2_%s.h5')

if __name__ == '__main__':
  _test_config, _test_game = test_game()
  test_model_training(_test_game)
  test_new_game(_test_config, _test_game)
