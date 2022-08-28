from constants import *
from collections import defaultdict
from ai import AI 
from itertools import combinations

from copy import copy, deepcopy
import time 

"""
unsupported strategies:
* revealing information about your hand to enhance collusion with other players against a third
  (less necessary if extra info is known about hand composition)
"""

class Player(object):
	def __init__(self, n_players, game, pre_existing_ai=None, id=None, memory=0, q_lag=8, other_configs=None):
		"""
		n_players must be specified so that an AI can be created if none is given
		id is not necessary except for identification (order iwll be more important once assigned to game)
		this object interacts with the game and records its own history
		"""
		#length of memories, essentially (serialization minus decision serialization)
		self.SERIALIZATION_LENGTH = 2*n_players * N_TRACKS + 19 + 24 + 14*(n_players-1)
		self.game = game 
		self.id = id
		if not pre_existing_ai:
			self.ai = AI(n_players, memory, model_discount_config=other_configs['model_discount_config'])
		else:
			self.ai = pre_existing_ai
		self.q_lag = q_lag
		self.trains = defaultdict(int)
		self.visible_trains = defaultdict(int)
		self.n_trains = 0
		self.uncompleted_tickets = []
		self.completed_tickets = []
		#set of frozensets
		#each frozenset is a pair of cities
		self.tracks = set()
		#supply of cars
		self.n_cars = 45
		#needs to be defined by game
		self.order = -1
		#points
		self.positive_ticket_points = 0
		self.negative_ticket_points = 0
		self.track_points = 0
		self.total_points = 0
		#not sure if I want this calculated constantly...
		self.has_longest_track = False
		self.longest_track_length = 0
		self.turn = -1
		self.win=False
		#zipped ticket serialization, which is updated whenever
		#new tickets are acquired; this function also uses 
		#iterate over second element of this
		self.base_ticket_serialization = zip(*self.self_ticket_serialization())[1]

		self.initialize_history()
		#memories
		self.memory=memory
		if self.memory==0:
			self.memories = []
		else:
			self.memories = [np.zeros(self.SERIALIZATION_LENGTH) for _ in range(memory)]

	def reset(self, reset_history=False):
		"""
		resets everything except history (unless set by param)
		"""
		self.trains = defaultdict(int)
		self.visible_trains = defaultdict(int)
		self.n_trains = 0
		self.uncompleted_tickets = []
		self.completed_tickets = []
		#set of frozensets
		#each frozenset is a pair of cities
		self.tracks = set()
		#supply of cars
		self.n_cars = 45
		#needs to be defined by game
		self.order = -1
		#points
		self.positive_ticket_points = 0
		self.negative_ticket_points = 0
		self.track_points = 0
		self.total_points = 0
		#not sure if I want this calculated constantly...
		self.has_longest_track = False
		self.longest_track_length = 0
		self.turn = -1
		self.win=False 
		#zipped ticket serialization, which is updated whenever
		#new tickets are acquired; this function also uses 
		#iterate over second element of this
		self.base_ticket_serialization = zip(*self.self_ticket_serialization())[1]
		if self.memory==0:
			self.memories = []
		else:
			self.memories = [np.zeros(self.SERIALIZATION_LENGTH) for _ in range(self.memory)]
		if reset_history:
			self.initialize_history()

	def __str__(self):
		return """player with %d/%d completed/uncompleted tickets, %d train cards, %d cars, and 
		%d total points""" % (len(self.completed_tickets), len(self.uncompleted_tickets), self.n_trains, self.n_cars, self.total_points)

	def apply_history(self, update_ai=True):
		"""
		applies history to self at end of game
		"""
		self.actual_scores[-1] = self.total_points 
		self.win_history = [self.win*1 for _ in self.serialization_history]
		max_turn = self.turn
		for i, delay in enumerate(self.ai.delay_values):
			for turn in self.decision_turn_indexes:
				self.q_score_histories[i].append(self.actual_scores[min(turn+delay, max_turn)])
		if update_ai:
			for i, delay in enumerate(self.ai.delay_values):
				self.ai.q_score_histories[i] += self.q_score_histories[i]
			self.ai.serialization_history += self.serialization_history 
			self.ai.ticket_serialization_history += self.ticket_serialization_history
			self.ai.memory_history += self.memory_history

	def update_memory(self):
		if self.memory==0:
			return 0
		#remove last and prepend
		self.memories.pop()
		serializations = self.append_serializations()
		self.memories = [serializations] + self.memories
		

	def initialize_history(self):
		self.serialization_history = []
		self.ticket_serialization_history = []
		self.win_history = []
		self.q_score_histories = [ [] for _ in self.ai.delay_values]

		#extra trackers for assisting in update
		#this will mark the player turn number at each decision
		self.decision_turn_indexes = []
		#this will mark what that player's score was for a given turn
		self.actual_scores = []

		self.decision_type_history = []
		#just stores the memories at a given point in time
		self.memory_history = []

	def record_history(self, serialization, ticket_serialization=None,
		decision_type=None):
		self.serialization_history.append(deepcopy(serialization))
		if ticket_serialization<>None:
			self.ticket_serialization_history.append(deepcopy(ticket_serialization))
		else:
			self.ticket_serialization_history.append(deepcopy(self.base_ticket_serialization[0]))
		self.decision_turn_indexes.append(self.turn)
		self.decision_type_history.append(decision_type)
		self.memory_history.append(deepcopy(self.memories))

	def reset_history(self):
		self.initialize_history()

	#these are actions that change the state of the player
	def begin_selecting_tickets(self, min_tickets=1):
		new_tickets = self.game.tickets[0:3]
		n_already_chosen_tickets = 0
		#will automatically add completed tickets
		for i in range(len(new_tickets))[::-1]:
			ticket = new_tickets[i]
			if self.finished_ticket(ticket, False):
				self.uncompleted_tickets.append(new_tickets.pop(i))
				self.finished_ticket(ticket, True)
				n_already_chosen_tickets+=1
		ticket_selection = self.ai.decide_on_tickets(self, new_tickets, max(0, min_tickets-n_already_chosen_tickets))
		ticket_indexes = [t['index'] for t in ticket_selection]
		reject_tickets = [ticket for ticket in new_tickets if ticket['index'] not in ticket_indexes ]
		self.select_tickets(ticket_selection, reject_tickets, 3)

	def select_tickets(self, ticket_list, reject_tickets = [], max_top_of_deck_range=4):
		"""
		has cities, points, and index
		"""
		max_top_of_deck_range = min(max_top_of_deck_range, len(self.game.tickets))
		#self.game.mt = max_top_of_deck_range 
		#self.game.tl = ticket_list 
		#self.game.rt = reject_tickets
		#ranges are reversed so that order doesn't change if anything is popped
		for ticket in ticket_list:
			if self.finished_ticket(ticket):
				self.completed_tickets.append(ticket)
				self.positive_ticket_points += ticket['points']
			else:
				self.uncompleted_tickets.append(ticket)
				self.negative_ticket_points += ticket['points']
			#remove ticket from game's ticket deck
			for top_ticket_index in range(max_top_of_deck_range)[::-1]:
				top_ticket = self.game.tickets[top_ticket_index]
				if top_ticket['index'] == ticket['index']:
					self.game.tickets.pop(top_ticket_index)
					max_top_of_deck_range = min(max_top_of_deck_range, len(self.game.tickets))
					break
		#sends unused tickets to the back of the deck
		for ticket in reject_tickets:
			for top_ticket_index in range(max_top_of_deck_range)[::-1]:
				top_ticket = self.game.tickets[top_ticket_index]
				if top_ticket['index'] == ticket['index']:
					self.game.tickets.append(self.game.tickets.pop(top_ticket_index))
		self.base_ticket_serialization = zip(*self.self_ticket_serialization())[1]
		return 0 

	def finished_ticket(self, ticket, mark_ticket=False):
		"""
		checks to see if a ticket is finished by the player
		if mark_ticket=True, this will move the player's ticket 
		from unfinished to finished when appropriate
		"""
		ticket_complete = check_tracks_for_completion(self.tracks, ticket)
		if mark_ticket and ticket_complete:
			for i in range(len(self.uncompleted_tickets))[::-1]:
				owned_ticket = self.uncompleted_tickets[i]
				if owned_ticket['index']==ticket['index']:
					self.completed_tickets.append(self.uncompleted_tickets.pop(i))
					self.negative_ticket_points -= ticket['points']
					self.positive_ticket_points += ticket['points']
		return ticket_complete

	def calculate_longest_track(self):
		self.longest_track_length = calculate_longest_track(self.tracks)
		return self.longest_track_length

	#this one has a second phase to decide what to do if 
	#the first decision isn't to take a locomotive (or only one ticket is left)
	def select_trains(self):
		"""
		decides one of the following
		 * pick face up
		 * pick top of deck
		then, if first card wasn't face-up locomotive
		 * pick face up
		 * pick top of deck
		"""
		train_choice = self.ai.decide_on_train(self, move=0)
		self.select_train(train_choice)
		if train_choice <> 'locomotive':
			second_train_choice = self.ai.decide_on_train(self, move=1)
			self.select_train(second_train_choice)

	def select_train(self, train_choice):
		self.game.shuffle_if_needed()
		if train_choice <> 'top':
			self.trains[train_choice]+=1
			idx = self.game.face_up_trains.index(train_choice)
			self.game.face_up_trains.pop(idx)
			self.game.lay_out_cards()
			self.game.check_if_too_many_locomotives()
			self.n_trains+=1
		else:
			new_train = self.game.trains.pop()
			self.trains[new_train]+=1
			self.n_trains+=1
			if len(self.game.trains) == 0:
				self.game.shuffle_discarded_trains_into_deck()

	def build_tracks(self, segment, cost, color_id = 0):
		"""
		builds tracks on a segment
		cost must be selected ahead of time
		color_id refers to which part of the track is being built
		"""
		#pays cost
		for color, amount in cost.iteritems():
			self.trains[color]-=amount
			self.game.discarded_trains += [color] * amount
			self.n_trains -= amount 
			self.n_cars -= amount
			#updates the discard pile vector
			self.game.discard_vector[DECK_COLOR_INDEXES[color]] += amount 
			#public info on colors
			self.visible_trains[color] = max(0, self.visible_trains[color] - amount)
		#updates track ownership
		track_id = segment['index']
		self.game.tracks[i]['occupied' + str(1+color_id)] = self.order
		self.tracks.add(segment['cities'])
		#updates points
		self.track_points += SEGMENT_POINTS[segment['length']]
		#update game serialization vector
		self.game.serialized_board_state[2*N_TRACKS*self.order + segment['index'] + color_id] = 1
		return 0 

	def calculate_train_cost(self, segment):
		"""
		determines possible costs of segment, taking rainbow cars into account
		shows options for gray/double tracks
		also checks if any particular color is occupied
		"""
		possible_costs1 = []
		possible_costs2 = []
		length = segment['length']
		color1 = segment['color1']
		color2 = segment['color2']
		occupied1 = segment['occupied1']
		occupied2 = segment['occupied2']
		#check to see if player already as trains on track or if player has too few trains to complete track
		if self.order in [occupied1, occupied2] or self.n_trains < length:
			return [ [], [] ]
		#check to see which ones are already occupied
		if occupied1 == None:
			possible_costs1 = self.calc_color_costs(length, color1)

		if occupied2 == None and color2 is not '':
			possible_costs2 = self.calc_color_costs(length, color2)

		return [possible_costs1, possible_costs2]

	def calc_color_costs(self, length, color):
		n_locomotives = self.trains['locomotive']
		possibilities = []
		if color=='gray':
			#note that the definition of "color" changes here; it has no impact on code, though
			for color in DECK_COLORS:
				amount = self.trains[color]
				if amount + n_locomotives >= length:
					lower_bound = max([0, length - n_locomotives])
					upper_bound = min([length, amount])
					for n_regular in range(lower_bound, upper_bound+1):
						possibilities.append({color:n_regular,'locomotive':length-n_regular})
		else:
			amount = self.trains[color]
			if amount + n_locomotives >= length:
				lower_bound = max([0, length - n_locomotives])
				upper_bound = min([length, amount])
				for n_regular in range(lower_bound, upper_bound+1):
					possibilities.append({color:n_regular,'locomotive':length-n_regular})
		return possibilities

	def determine_possible_tracks(self):
		"""
		look at all the different open tracks and calculate what they would cost
		returns zipped lists of tracks with their possible costs (a list of 2 lists containing dicts)
		"""
		track_segment_costs = []
		track_segments = []
		for track in self.game.tracks:
			if self.n_cars < track['length']:
				continue
			cost = self.calculate_train_cost(track) 
			if len(cost[0])==0 and len(cost[1])==0:
			 	continue
			track_segment_costs.append(cost)
			track_segments.append(track)
		return zip(track_segments, track_segment_costs)

	#take turn
	def take_turn(self):
		self.update_points()
		self.turn+=1
		#iterate through building possibilities
		possible_tracks = self.determine_possible_tracks()
		#self.game.pt = possible_tracks
		#decide on what action to take
		data, action = self.ai.decide_on_action(self, possible_tracks)
		if self.game.debug:
			print 'action: %s' % action 
		#print 'data:'
		#print data 
		#time.sleep(1)
		#take action
		if action=='should_choose_train':
			self.select_trains()
		elif action=='should_choose_tickets':
			#implement
			self.begin_selecting_tickets()
		elif action=='should_build_tracks':
			data = self.ai.decide_on_tracks(self, possible_tracks)
			segment = TRACKS[data['index']]
			cost = data['cost']
			color_id = data['color_id']
			self.build_tracks(segment, cost, color_id)
		#if tracks were built, update longest track calculation
		#and check ticket completion
		self.calculate_longest_track()
		for i in range(len(self.uncompleted_tickets))[::-1]:
			self.finished_ticket(self.uncompleted_tickets[i], True)
		self.update_points()
		self.update_memory()
		self.actual_scores.append(self.total_points)

	#puts a hard limit to avoid disastrous behavior
	def can_select_tickets(self):
		return (self.game.config['excess_ticket_limit'] > len(self.uncompleted_tickets)) and (len(self.game.tickets) >= 3)

	#used for loading/saving player states; might go unimplemented
	def save_state(self):
		pass

	def load_state(self):
		pass

	def update_points(self, check_longest_track=False):
		self.total_points = self.track_points + self.positive_ticket_points - self.negative_ticket_points
		if check_longest_track and self.has_longest_track:
			self.total_points += 10
		return 0 

	def other_players(self):
		return self.game.players[self.order+1:] + self.game.players[:self.order]

	def append_serializations(self, serializations=None):
		"""
		takes a bunch of serialized decisions and then appends the player/board game states to them, resulting in 
		vectors that describe the entire game
		"""
		game_serialization = self.game.full_serialization()
		self_serialization = self.serialize_self()
		player_serializations = [player.serialize_self(external=True) for player in self.other_players()]
		serializations_to_append = np.hstack([game_serialization, self_serialization] + player_serializations)
		if serializations==None:
			return serializations_to_append
		return [np.hstack([s, serializations_to_append]) for s in serializations]

	def serialize_self(self, external=False):
		"""
		#anyone
		n_tickets - 1
		n_points (visible) - 1
		visible_trains - 9
		n_trains - 1
		train_cars left - 1
		longest_track - 1
		#self only
		all_trains - 9
		net_ticket_points - 1
		#24 + (n_players-1)*14 total
		"""
		if external:
			serialization = np.zeros(14)
		else:
			#this will be extended below
			serialization = np.zeros(14 + 9 + 1)
		n_completed_tickets = len(self.completed_tickets)
		n_uncompleted_tickets = len(self.uncompleted_tickets)
		#assign public
		serialization[0] = (n_completed_tickets + n_uncompleted_tickets)/7.
		serialization[1] = self.track_points
		for color, amount in self.visible_trains.iteritems():
			serialization[2+DECK_COLOR_INDEXES[color]] = amount/4.
		serialization[11] = self.n_trains
		serialization[12] = self.n_cars/45.
		serialization[13] = self.longest_track_length/30.
		if not external:
			for color, amount in self.trains.iteritems():
				serialization[14 + DECK_COLOR_INDEXES[color]] = amount/4.
			serialization[23] = self.positive_ticket_points
			#ticket_serialization = self.game.serialize_tickets(self.tickets)
			#serialization = np.hstack(serialization, ticket_serialization)
		return serialization

	def self_ticket_serialization(self, new_tickets=None, min_tickets=None):
		"""
		tickets themselves contain their serialization, so all we need to do is return all valid
		combinations of tickets
		"""
		n_tickets = len(self.uncompleted_tickets)
		if new_tickets is None:
			n_new_tickets = 0
		else:
			n_new_tickets = len(new_tickets)
		ticket_limit = self.game.config['excess_ticket_limit']
		#this records which tickets are used beyond owned ones
		if new_tickets==None:
			additional_ticket_combos  = [ [], ]
			return zip(additional_ticket_combos,
				pad_with_np_zeros([[t for t in self.uncompleted_tickets]], ticket_limit))
		additional_ticket_combos = []
		ticket_combinations = []
		#minimum is the pre-determined lower-bound, the number of tickets to choose from, and how many tickets are
		#  left in the player's ticket limit
		#maximum is the minimum of the number of new tickets and how many tickets are left in the player's ticket limit
		for num_new_tickets in range(min(min_tickets, n_new_tickets, ticket_limit - n_tickets),min(ticket_limit - n_tickets, n_new_tickets)+1):
			if num_new_tickets == 0:
				continue
			for subset in combinations(new_tickets, num_new_tickets):
				ticket_combinations.append(self.uncompleted_tickets + list(subset))
				additional_ticket_combos.append(subset)
				#print num_new_tickets
		return zip(additional_ticket_combos,
			pad_with_np_zeros(ticket_combinations, ticket_limit))


#function will later need config updated if limit is to be adjusted
def pad_with_np_zeros(serializations, limit=7):
	"""
	iterates through each combination and makes sure that all slots are filled
	"""
	#I am lazy so I am doing the expansion here...
	#print serializations
	serializations = [[x['serialization'] for x in t] for t in serializations]
	for serialization_set in serializations:
		if len(serialization_set)==limit:
			continue
		else:
			serialization_set+= (limit-len(serialization_set))*[np.zeros(N_CITIES+1)]
	return serializations


#a couple of recursive path-navigating algorithms
#the first isn't efficient, but I don't think it's a huge issue for the given map sizes
def check_tracks_for_completion(tracks, ticket, 
	unexplored_paths='default', starting_city=None):
	#checks if city is destination
	if starting_city <> None:
		if starting_city==ticket[1]:
			return True
	#default behavior for first call
	if unexplored_paths == 'default':
		unexplored_paths = deepcopy(tracks)
	if isinstance(ticket, dict):
		ticket = list(ticket['cities'])
	if not starting_city:
		starting_city = ticket[0]
	#general part of algorithm
	#check all cities that have connections to starting city
	connecting_cities = CITY_CITY_CONNECTIONS[starting_city]
	#for each of these, check which ones exist in unexplored_paths
	#then, temporarily pop off this track and then 
	valid_connections = [city for city in connecting_cities if frozenset([starting_city, city]) in unexplored_paths]
	for city in valid_connections:
		elem_to_remove = frozenset([city, starting_city])
		unexplored_paths.remove(elem_to_remove)
		if check_tracks_for_completion(tracks, ticket, 
			unexplored_paths=unexplored_paths,
			starting_city=city):
			return True
		unexplored_paths.add(elem_to_remove)
	return False 

def calculate_longest_track(tracks,  
	unexplored_paths='default', starting_city=None):
	if unexplored_paths == 'default':
		unexplored_paths = deepcopy(tracks)
	if starting_city==None:
		#iterate over all cities
		#check which cities are valid, then make recursive call for each
		valid_starting_cities = set()
		#if no paths exist, longest path=0
		if len(tracks)==0:
			return 0
		for track in tracks:
			for city in track:
				valid_starting_cities.add(city)
		#iterates over all cities and returns the maximum length of a path that begins
		#at one of those cities
		path_lengths = []
		for city in valid_starting_cities:
			path_lengths.append(calculate_longest_track(tracks,
				unexplored_paths, starting_city=city))
		return max(path_lengths)
	else:
		#iterate over connecting cities
		#general part of algorithm
		#check all cities that have connections to starting city
		connecting_cities = CITY_CITY_CONNECTIONS[starting_city].keys()
		#for each of these, check which ones exist in unexplored_paths
		#then, temporarily pop off this track and then 
		valid_connections = [city for city in connecting_cities if frozenset([starting_city, city]) in unexplored_paths]
		#if no valid connections, then terminates path
		if len(valid_connections)==0:
			return 0
		#for each
		connection_lengths = [None for _ in valid_connections]
		for i, city in enumerate(valid_connections):
			#removes and re-adds the path element from the set being passed so that it 
			#doesn't spend too much time using copy/deepcopy
			elem_to_remove = frozenset([city, starting_city])
			unexplored_paths.remove(elem_to_remove)
			max_sublength = calculate_longest_track(tracks, unexplored_paths, starting_city=city)
			#second index of the CITY_CITY_CONNECTIONS object is the length (first is a numeric ID for the card)
			connection_lengths[i] = max_sublength + CITY_CITY_CONNECTIONS[starting_city][city][1]
			unexplored_paths.add(elem_to_remove)
		return max(connection_lengths)
	
	
