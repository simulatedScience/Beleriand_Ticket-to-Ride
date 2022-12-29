from constants import *
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model

from copy import copy
import warnings


# calculate some dimension constants
TICKET_SERIALIZATION_LENGTH = N_CITIES + 1


class AI(object):
  def __init__(self, n_players, memory, n_tickets=7, model_discount_config=DEFAULT_MODEL_DISCOUNT_CONFIG):
    # define constants used for serialization
    self.decision_template = np.zeros(23 + 2 * N_TRACKS)
    self.MEMORY_SERIALIZATION_LENGTH = (2 * n_players * N_TRACKS + 19 + 24 + 14 * (n_players - 1))
    # add decision serialization
    self.SERIALIZATION_LENGTH = self.MEMORY_SERIALIZATION_LENGTH + 23 + 2 * N_TRACKS
    self.TICKET_SERIALIZATION_LENGTH = N_CITIES + 1
    self.memory = memory
    self.n_tickets = n_tickets
    self.n_players = n_players
    self.discount_config = model_discount_config

    # construct input layers
    self.main_input = Input(shape=(self.SERIALIZATION_LENGTH,))
    self.memory_inputs = [Input(shape=(self.MEMORY_SERIALIZATION_LENGTH,)) for _ in range(memory)]
    self.ticket_inputs = [Input(shape=(self.TICKET_SERIALIZATION_LENGTH,)) for _ in range(n_tickets)]
    self.model_inputs = [self.main_input] + self.memory_inputs + self.ticket_inputs

    self.discount_values = [x[1] for x in model_discount_config]
    self.delay_values = [x[0] for x in model_discount_config]
    # placeholder, will be initialized in initialize_network(...)
    self.q_models = [None for _ in self.delay_values]

    for i, delay in enumerate(self.delay_values):
      self.initialize_network(i)

    self.initialize_history()

  def initialize_network(self, index):
    memory = self.memory
    n_tickets = self.n_tickets
    n_players = self.n_players

    main_input_dense_layer = Dense(256, activation='relu')
    memory_input_dense_layers = [Dense(256, activation='relu') for _ in range(memory)]
    ticket_input_dense_layer = Dense(64, activation='relu')

    # connect
    main_encoded = main_input_dense_layer(self.main_input)
    memory_encoded = [dense_layer(input_layer) for dense_layer, input_layer in zip(memory_input_dense_layers,
                                                                                   self.memory_inputs)]
    ticket_encoded = [ticket_input_dense_layer(ticket_input) for ticket_input in self.ticket_inputs]

    # make first full layer
    merged_input = keras.layers.concatenate([main_encoded] + memory_encoded + ticket_encoded)

    # make next 2 dense layers
    LAYER_2 = Dense(256, activation='relu')(merged_input)
    LAYER_3 = Dense(128, activation='relu')(LAYER_2)

    # separate win prediction and q learning into 2 dense layers
    # introduce skip layers from main_input and tickets to enhance predictive capabilities
    Q_DENSE_LAYER = Dense(32, activation='relu')(
        keras.layers.concatenate([LAYER_3] + [self.main_input] + self.ticket_inputs))

    # output layers
    Q_PREDICTIONS = Dense(1, activation='relu')(Q_DENSE_LAYER)

    # formalize inputs
    #model_inputs = [main_input] + memory_inputs + ticket_inputs

    self.q_models[index] = Model(inputs=self.model_inputs, outputs=Q_PREDICTIONS)
    self.q_models[index].compile(optimizer='rmsprop',
                                 loss='mean_squared_error',
                                 metrics=['MSE'])

  def initialize_history(self):
    self.serialization_history = []
    self.ticket_serialization_history = []
    self.win_history = []
    self.q_score_histories = [[] for _ in self.delay_values]
    self.memory_history = []

  def reset_history(self):
    self.initialize_history()

  # the filename can be anything, as the delay and numeric index are handled independently
  def save_models(self, filename):
    filename_template = filename + '_N%d_D%d.h5'
    for i, delay in enumerate(self.delay_values):
      self.q_models[i].save(filename_template % (i, delay))

  def load_models(self, filename):
    filename_template = filename + '_N%d_D%d.h5'
    filename_template = filename + '_N%d_D%d.h5'
    for i, delay in enumerate(self.delay_values):
      self.q_models[i] = load_model(filename_template % (i, delay))

  def receive_models_from_other_ai(self, other_ai, name_template='transferable_ai'):
    """
    will allow transfer of AI from one AI to another...not a graceful method, but not used enough to have that be necessary
    """
    other_ai.save_models(name_template)
    self.load_models(name_template)

  # when a new game is started, these configs will be updated according to game configs
  def update_discount_configs(self, config):
    model_discount_config = config['model_discount_config']
    self.discount_values = [x[1] for x in model_discount_config]
    self.delay_values = [x[0] for x in model_discount_config]

  def prepare_data(self, index):
    """
    prepares input & output data for a player based on their history
    """

    y = np.asarray(self.q_score_histories[index])

    # inputs is a list of multiple numpy arrays
    x = []
    x.append(np.asarray(self.serialization_history))
    for i in range(len(self.memory_history[0])):
      x.append(np.asarray([mem_hist[i] for mem_hist in self.memory_history]))
    for i in range(len(self.ticket_serialization_history[0])):
      x.append(np.asarray([ticket_hist[i] for ticket_hist in self.ticket_serialization_history]))
    return x, y

  def train(self, **kwargs):
    for i, discount in self.discount_values:
      if discount == 0:
        continue
      kwargs['index'] = i
      self.train_q(**kwargs)

  def train_q(self, n_epochs=10, index=0):
    x, y = self.prepare_data(index)
    self.q_models[index].fit(x, y, epochs=n_epochs, batch_size=1000, verbose=0)

  def decide_on_action(self, player, possible_tracks):
    """
    this function returns the action and action data
    it also appends history to players' actions for training
    """
    # these are used to describe the action being taken
    action_types = []
    action_data = []
    serializations = []

    # should choose train
    if player.game.has_grabbable_train_pile():
      action_types.append('should_choose_train')
      action_data.append(None)
      serializations.append(self.decision_serialize(player, 'SHOULD_CHOOSE_TRAIN'))
    # should choose tickets
    if player.can_select_tickets():
      action_types.append('should_choose_tickets')
      action_data.append(None)
      serializations.append(self.decision_serialize(player, 'SHOULD_CHOOSE_TICKETS'))
    # build tracks
    if len(possible_tracks) > 0:
      action_types.append('should_build_tracks')
      action_data.append(None)
      serializations.append(self.decision_serialize(player, 'SHOULD_BUILD_TRACKS'))
    # skip turn (only makes sense with hoarders near end of game...)
    if len(action_types) == 0:
      action_types.append('do_nothing')
      action_data.append(None)
      serializations.append(self.decision_serialize(player, 'DO_NOTHING'))
    # concatenate each serialization
    serializations = player.append_serializations(serializations)

    # calculate each non-zero q score
    q_scores = self.calculate_q_scores(serializations, player)

    # make a decision based on both
    # some player/game configs go into the algorithm
    decision_index = self.decide_based_on_q_scores(q_scores, player)

    # record history and then return decision and related data
    player.record_history(serializations[decision_index])

    # return action
    return action_data[decision_index], action_types[decision_index]

  def decide_on_tracks(self, player, possible_tracks):
    action_types = []
    action_data = []
    serializations = []
    for track, cost_list in possible_tracks:
      for color_id in range(2):
        for cost in cost_list[color_id]:
          action_types.append('build_tracks')
          data = {'color_id': color_id, 'index': track['index'], 'cost': cost}
          action_data.append(data)
          serializations.append(self.decision_serialize(player, 'BUILD_TRACKS', data))
    # concatenate each serialization
    serializations = player.append_serializations(serializations)
    # calculate q-scores
    q_scores = self.calculate_q_scores(serializations, player)

    # make a decision based on both
    # some player/game configs go into the algorithm
    decision_index = self.decide_based_on_q_scores(q_scores, player)

    # record history and then return decision and related data
    player.record_history(serializations[decision_index])

    # return action
    return action_data[decision_index]


  def decide_on_tickets(self, player, tickets, min_tickets):
    if player.game.debug:
      pass
      # print 'length and min number of tickets'
      # print len(tickets)
      # print min_tickets
    action_type = 'decide_on_tickets'
    # second arg doesn't do anything in this case since tickets are handled differently
    decision_serialization = [self.decision_serialize(player, 'ticket_selection')]
    serializations = player.append_serializations(decision_serialization)
    ticket_serializations = player.self_ticket_serialization(tickets, min_tickets)
    # calculate q-scores
    q_scores = self.calculate_q_scores(serializations, player)

    # make a decision based on both
    # some player/game configs go into the algorithm
    decision_index = self.decide_based_on_q_scores(q_scores, player)

    # record history (with new ticket state)
    player.record_history(serializations[0], ticket_serialization=ticket_serializations[decision_index][1])
    # return ticket set
    return ticket_serializations[decision_index][0]

  def decide_on_train(self, player, move=0):
    action_type = 'decide_on_train'
    options = set(['top'])
    for color in player.game.face_up_trains:
      options.add(color)
    options = list(options)
    decision_serializations = [self.decision_serialize(
        player, 'CHOOSE_TRAINS', {'color': color, 'move': move}) for color in options]

    # concatenate each serialization
    serializations = player.append_serializations(decision_serializations)
    # calc probability & q-score
    q_scores = self.calculate_q_scores(serializations, player)

    # make a decision based on both
    # some player/game configs go into the algorithm
    decision_index = self.decide_based_on_q_scores(q_scores, player)

    # record history and then return decision and related data
    player.record_history(serializations[decision_index])
    return options[decision_index]

  def decision_serialize(self, player, mode, selection_data=None):
    """
    selection_data contains data necessary to describe a decision

    SHOULD_CHOOSE_TRAIN: length 1 (binary)
    SHOULD_CHOOSE_TICKETS: length 1 (binary)
    SHOULD_BUILD_TRACKS: length 1 (binary)
    CHOOSE TRAINS: length 10, colors and take from top
    BUILD_TRACKS: length 2*n_tracks, 2 for each track
    BUILD_TRACKS_COST: length 9, colors
    #TICKET_SELECTION: length: (n_cities + 1 (last element is points))*3; there are 3 tickets

    TOTAL: 1 + 1 + 1 + 10 + 2*n_tracks + 9 + 3*n_cities + 3 + 1 = 24 + 2*n_tracks 
    """
    serialization = copy(self.decision_template)
    if mode == 'SHOULD_CHOOSE_TRAIN':
      serialization[0] = 1
    elif mode == 'SHOULD_CHOOSE_TICKETS':
      serialization[1] = 1
    elif mode == 'SHOULD_BUILD_TRACKS':
      serialization[2] = 1
    elif mode == 'CHOOSE_TRAINS':
      if selection_data['move'] == 1:
        # a train has already been selected flag put at end of decision
        serialization[22 + 2 * N_TRACKS] = 1
      color = selection_data['color']
      if color == 'top':
        serialization[3 + 9] = 1
      else:
        serialization[3 + DECK_COLOR_INDEXES[color]] += 1
    elif mode == 'BUILD_TRACKS':
      track_index = selection_data['index']
      color_id = selection_data['color_id']
      cost = selection_data['cost']
      serialization[13 + track_index * 2 + color_id] = 1
      for color, amount in cost.items():
        serialization[13 + 2 * N_TRACKS + DECK_COLOR_INDEXES[color]] = amount
    return serialization

  # main serialization - memory serializations - ticket serializations
  def calculate_q_scores(self, serializations, player, ticket_combos=None):
    """
    will iterate over ticket combo serializations if sspecified, otherwise
    will use default player ticket serializations
    """
    if not ticket_combos:
      ticket_serialization = player.base_ticket_serialization[0]
      n_serializations = len(serializations)
      model_inputs = [np.vstack(serializations)] \
          + vertical_repeat(player.memories, n_serializations) \
          + vertical_repeat(ticket_serialization, n_serializations)
    else:
      n_combos = len(ticket_combos)
      #player.game.s = serializations
      #player.game.m = player.memories
      #player.game.tc = ticket_combos
      #print [np.asarray(ticket_combos)]
      model_inputs = vertical_repeat(serializations, n_combos) + \
          vertical_repeat(player.memories, n_combos) + restack(ticket_combos)
    all_predictions = []
    for i, delay_value in enumerate(self.delay_values):
      if self.discount_values[i] == 0:
        all_predictions.append([])
        continue
      preds = self.q_models[i].predict(model_inputs)
      all_predictions.append(preds[:, 0])
    return all_predictions

  # decides based on scores
  def decide_based_on_q_scores(self, q_scores, player, epsilon=EPSILON):
    """
    uses temperature and discount parameters
    """
    temperature = player.game.config['temperature']
    # filters out weights that are zero (and the corresponding preds)
    weights, preds = list(zip(*[(w, p) for w, p in zip(self.discount_values, q_scores) if w > 0]))
    # multiplies weights and preds together for weighted sums
    weighted_sums = np.matmul(weights, preds)
    weighted_sums = weighted_sums - np.max(weighted_sums)
    # note that in a normal boltzmann distribution, the exponential is negative; in this
    # case maximization is desired so the sign is inverted (and the max is subtracted to avoid issues with huge values)
    if temperature == 0:
      return np.argmax(weighted_sums)
    loadings = np.exp(weighted_sums / temperature) + epsilon
    # print new_probs
    try:
      return np.random.choice(len(loadings), p=loadings / np.sum(loadings))
    except ValueError:
      warnings.warn("probs didn't sum to 1; defaulting to uniform probabilities")
      return np.random.choice(len(loadings))


def vertical_repeat(vec, n):
  """
  given a list of numpy vectors, will repeat each element of the list vertically n times
  this is used when serialized vectors are combined, and some need to be repeated for each combination of the other
  """
  return [v.reshape((1, len(v))).repeat(n, axis=0) for v in vec]


def restack(combos):
  """
  ticket combinations are originally [ [array1, array2, array3], [array1, array2, array3], ...]
  it needs to be [ rbind_all(array1), rbind_all(array2), ...]
  """
  return [np.vstack(c) for c in zip(*combos)]


def min_normalize(x, epsilon=EPSILON):
  return (x - np.min(x)) / (epsilon + np.max(x) - np.min(x))


def normalize(x, temperature, epsilon=EPSILON):
  if temperature < epsilon:
    selected = x + epsilon >= np.max(x) * 1
    return (selected + epsilon) / np.sum(selected + epsilon)
  else:
    x = x + epsilon
    selected = x**(1. / (temperature)**0.5)
    return (selected) / np.sum(selected)


def select_ith_of_zipped(zipped, idx):
  return [x for i, x in enumerate(zipped) if i == idx][0]


# junk code
'''
		elif mode=='TICKET_SELECTION':
			#contains up to 3 tickets; "empty" tickets will be indicated by "None"
			tickets = selection_data['tickets']
			starting_index = 12 * 2*N_TRACKS + 9
			ticket_index_difference = N_CITIES + 1
			for i, ticket in enumerate(tickets):
				if ticket is None:
					continue
				serialization[i*ticket_index_difference + starting_index + CITY_INDEXES[ticket['city1']]] = 1
				serialization[i*ticket_index_difference + starting_index + CITY_INDEXES[ticket['city2']]] = 1
				serialization[i*ticket_index_difference + starting_index + N_CITIES] = ticket['points']/10.\
		'''
