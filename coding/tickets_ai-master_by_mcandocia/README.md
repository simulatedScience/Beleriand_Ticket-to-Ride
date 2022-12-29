# Ticket AI

This repository contains the code and notebooks used to train keras networks to play [Ticket to Ride](https://en.wikipedia.org/wiki/Ticket_to_Ride_(board_game)), a 2 to 5-player board game that involves building rail lines between US cities and scoring points for building tracks, completing routes, and having the longest rail line.

This project is for purely academic purposes, and I hope the eventual findings are interesting to a wide audience. See [Using Neural Networks to Play Board Games](http://maxcandocia.com/article/2017/Jul/22/using-neural-networks-to-play-board-games/) and [Using AI to Determine Strategy in Machi Koro](http://maxcandocia.com/article/2017/Jul/30/using-ai-for-machi-koro-strategy/) for examples of a game I have analyzed this way.

For a general idea of how reinforcement learning works, see [my demo on my website](http://maxcandocia.com/article/2017/Nov/05/reinforcement-learning-demo-keras/).

## game.py

This file contains the `Game` class, which organizes the players' turn order, the state of cards in the train and ticket decks, and end-of-game scoring/methods.

## player.py

This file contains the `Player` class, which contains all of the player data. Actions which the players take are directly executed here. Decisions, apart from no-brainer decisions, such as accepting tickets for routes already completed, are handled in the `AI` class, which is `self.ai` when used inside the `Player` class.

## ai.py

This file contains the `AI` class, which handles all decisionmaking for players. They utilize a special shared neural network in Keras that predicts both probabilities of winning, as well as a "q" score, which describes gains made within a few turns of taking an action.

## constants.py

This file contains constants that are necessary for the game to function

## tickets.csv

This file contains the ticket data from the base Ticket to Ride game

## tickets_bigcities.csv

This file contains the ticket data from the Big Cities expansion for Ticket to Ride. I also recommend the expansion to the base game, since it has larger train cards that are easier to handle.

## tracks.csv

This file describes the connections between each city. It reads like an edgelist with a length and 2 color attributes.

## test.ipynb

I am currently testing out the networks and seeing how quickly a 3-player game can be completed (and with what scores). Network architecture is still being tested, and win performance is being evaluated by testing out differences in player win rates vs. training amounts.

The fastest a single player can end the game is in `ceiling((45-4-2)/2) + 36/6 + 3/3 + 1 =` 28 turns. This would require the player to exclusively build length-6 and length-5 tracks, with the possible exception of one track, and to use up all but one train card, and to draw no more tickets after the beginning. Ideally the average game length would be around 40-50 turns/player.