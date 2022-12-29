from constants import *
from player import Player 
from copy import copy, deepcopy
from random import shuffle

import numpy as np



class Game(object):
	def __init__(self, pre_existing_players=None, 
		config=DEFAULT_GAME_CONFIG, n_players=None, q_lag=8,
		separated_models=True):
		self.config = config 
		self.trains = copy(TRACK_DECK)
		shuffle(self.trains)
		self.discarded_trains = []
		self.face_up_trains = []
		if n_players is not None:
			self.n_players = n_players
		else:
			self.n_players = len(pre_existing_players)
		if not pre_existing_players:
			self.players = [Player(id = i, memory=config['memory'], 
				n_players=n_players, game=self, q_lag=q_lag, other_configs=config) for i in range(n_players)]
		else:
			self.players = pre_existing_players 
			self.reset_players_history()
		for i, player in enumerate(self.players):
			player.order = i 
			player.game = self
			player.ai.update_discount_configs(config) 		
		#organize tracks
		self.tracks = deepcopy(TRACKS)
		for track in self.tracks:
			#these will refer to player order numbers when they become
			#occupied
			track['occupied1'] = None
			track['occupied2'] = None 
		self.city_city_connections = deepcopy(CITY_CITY_CONNECTIONS)

		if config['ticket_versions'] == 'base':
			self.tickets = deepcopy(BASE_TICKETS)
		elif config['ticket_versions'] == 'bigcities':
			self.tickets = deepcopy(BIGCITIES_TICKETS)
		else:
			self.tickets = deepcopy(BASE_TICKETS + BIGCITIES_TICKETS)
		shuffle(self.tickets)

		self.longest_track_player = None
		#controls for the end of the game
		self.last_round = False
		self.final_countdown = len(self.players)
		self.end_game = False
		#some things used to check at end of turn
		self.tiles_updated = False
		self.turn = -1

	def run(self, debug=False):
		self.debug=debug
		if debug:
			print 'giving random cards'
		self.give_random_cards_to_players()
		if debug:
			print 'laying out cards'
		self.lay_out_cards()
		if debug:
			print 'passing out first tickets'
		self.pass_first_tickets()
		while self.final_countdown > 0:
			self.turn += 1
			if debug:
				print 'turn %d' % self.turn 
			current_player = self.players[self.turn % self.n_players]
			current_player.take_turn()
			if self.last_round:
				if debug:
					print 'on last round'
				self.final_countdown -= 1
			else:
				self.check_if_final_countdown(current_player)
			self.shuffle_if_needed()
		#end game
		if debug:
			print 'calculating end of game'
		self.calculate_end_game()

	def calculate_end_game(self):
		self.calculate_longest_track()
		max_points = -200
		self.winning_score = max_points
		for player in self.players:
			player.update_points(True)
			max_points = max(max_points, player.total_points)
			self.winning_score = max_points
		first_win_check = []
		for player in self.players:
			if player.total_points==max_points:
				first_win_check.append(player)

		if len(first_win_check)==1:
			first_win_check[0].win=True
		else:
			max_completed_tickets = 0
			for player in first_win_check:
				max_completed_tickets = max(max_completed_tickets, len(player.completed_tickets))
			second_win_check = []
			for player in first_win_check:
				if len(player.completed_tickets)==max_completed_tickets:
					player.win=True
		for player in self.players:
			player.apply_history()

	def __str__(self):
		winners = str([i for i, player in enumerate(self.players) if player.win])
		return '%d-player game on turn %d; winners: %s' % (self.n_players, self.turn, winners)

	def has_grabbable_train_pile(self):
		"""
		if the deck size + discard pile is less than 4, then this returns
		false because there aren't enough trains in supply (they need to be spent)
		"""
		return len(self.trains) + len(self.discarded_trains) > 3

	def reset_players_history(self):
		"""
		call after any desired information has been extracted
		"""
		for player in self.players:
			player.reset(True)


	def check_if_final_countdown(self, player):
		if player.n_cars <= 2:
			self.last_round=True

	def pass_first_tickets(self):
		for player in self.players:
			player.begin_selecting_tickets(min_tickets=2)

	def give_random_cards_to_players(self, n_cards=4):
		for player in self.players:
			for i in range(n_cards):
				player.trains[self.trains.pop()] += 1
				player.n_trains += 1
		return 0 

	def shuffle_if_needed(self):
		if len(self.trains) < 3:
			self.shuffle_discarded_trains_into_deck()

	def shuffle_discarded_trains_into_deck(self):
		self.trains = self.discarded_trains + self.trains 
		self.discarded_trains = []
		shuffle(self.trains)
		self.discard_vector = np.zeros(9)
		return 0 

	def lay_out_cards(self):
		"""
		will fill out face up trains and make sure that 
		"""
		while len(self.face_up_trains) < 5 and len(self.trains) > 0:
			self.face_up_trains.append(self.trains.pop())
		return 0

	def check_if_too_many_locomotives(self):
		"""
		will not shuffle trains into deck if too few cards in discard pile
		"""
		n_locomotives = len([0 for color in self.face_up_trains if color=='locomotive'])
		while n_locomotives >= 3 and len(self.discarded_trains) > 5:
			self.discarded_trains += self.face_up_trains 
			self.face_up_trains = []
			if len(self.trains) < 5:
				self.shuffle_discarded_trains_into_deck()
			self.lay_out_cards()
			n_locomotives = len([0 for color in self.face_up_trains if color=='locomotive'])
		return 0


	def calculate_longest_track(self):
		maxlen = 0
		best_players = []
		for player in self.players:
			player.has_longest_track = False
			track_length = player.calculate_longest_track()
			if track_length > maxlen:
				maxlen = track_length
				best_players = [player]
			elif track_length == maxlen:
				best_players.append(player)
		for player in best_players:
			player.has_longest_track = True 
		return maxlen

	#used for saving/loading game states
	def save_state(self):
		pass

	def load_state(self):
		pass

	#some general data structure serialization
	#I forget if this is even used...
	def serialize_track_segment(self, segment):
		#vector that shows which cities it connects
		city_vector = np.zeros(N_CITIES)
		for city in segment['cities']:
			city_vector[CITY_INDEXES[city]] = 1
		#index vector is another way of correlating the data to the segment on the map
		index_vector = np.zeros(N_TRACKS)
		index_vector[segment['index']] = 1
		#6-length vector to describe track length
		length_vector = np.zeros(6)
		length_vector[segment['length'] - 1] = 1
		#determines if track is a double track
		double_track_vector = 1*segment['double']
		#vector that shows ownership of tracks
		track_ownership_vector = np.zeros( 2*self.n_players)
		track_color_vector = np.zeros(2*N_BASE_COLORS)
		if segment['double']:
			if segment['occupied1']:
				track_ownership_vector[segment['occupied1']] = 1
			if segment['occupied2']:
				track_ownership_vector[segment['occupied2'] + self.n_players] = 1
			track_color_vector[segment['color1']] = 1
			track_color_vector[segment['color2'] + N_BASE_COLORS] = 1
		else:
			if segment['occupied1']:
				track_ownership_vector[segment['occupied1']] = 1
			track_color_vector[segment['color1']] = 1
		return np.hstack([city_vector, index_vector, length_vector, double_track_vector, track_color_vector, track_ownership_vector])

	def calc_serialized_board_state(self, player=None):
		"""
		for each track/segment, there is a 2*n_player vector describing the ownership of any double tiles
		updates are permanent, so this should be updated instead of calculated each time

		the order shall be P0 (city1_1, city1_2) (city2_1, city2_2)| P1 (cities_1, cities_2) ETC.
		this allows players to share the same vector and apply numpy slicing 
		some entries will always be empty

		total length of 2*n_players*n_tracks
		"""
		if not hasattr(self, 'serialized_board_state'):
			self.serialized_board_state = np.zeros(N_TRACKS * self.n_players * 2)
		if player==None:
			return self.serialized_board_state 
		else:
			if player.order == 0:
				return self.serialized_board_state
			else:
				shift_index = 2*N_TRACKS * player.order
				return np.roll(self.serialized_board_state, -shift_index)


	def serialize_deck_state(self):
		"""
		vector describing counts of cards in pile, as well as how many cards are in 
		important: discard_vector must be updated elsewhere so that the discard pile doesn't have to be looped
		through so many times
		total length of 19
		"""
		color_vector = np.zeros(9)
		for card in self.face_up_trains:
			color_vector[DECK_COLOR_INDEXES[color]] += 1
		if not hasattr(self, 'discard_vector'):
			self.discard_vector = np.zeros(9)
		cards_remaining_vector = len(self.trains)/50.#50 makes the number easier on the network regularizers
		return np.hstack([color_vector, self.discard_vector, cards_remaining_vector])

	def full_serialization(self):
		return np.hstack([self.calc_serialized_board_state(), self.serialize_deck_state()])

	def plot(self):
		"""
		when implemented, will produce an image that represents the board state in an intuitive way
		"""