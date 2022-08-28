import csv
from collections import defaultdict
import numpy as np 

"""
in case this is helpful...

VERNACULAR:

 * tickets - objectives that each player needs to achieve for points; in the form of a dict
 * tracks - the segments between each pair of adjacent cities; "double" tracks are counted as one, and
            each track keeps track of both parts of double tracks, even if only one exists
 * cars - the 45 pieces that each player starts with to build tracks
 * trains - the currency used to build tracks with cars; there are 8 colors and a locomotive "rainbow"/wild card

"""

#filename sources
TRACK_DATA_FILENAME = 'tracks.csv'
BASE_TICKETS_FILENAME = 'tickets.csv'
BIGCITIES_TICKETS_FILENAME = 'tickets_bigcities.csv'

#main data
CITIES = ['vancouver',
		  'calgary',
		  'winnipeg',
		  'sault st. marie',
		  'montreal',
		  'seattle',
		  'helena',
		  'duluth',
		  'toronto',
		  'boston',
		  'portland',
		  'salt lake city',
		  'omaha',
		  'chicago',
		  'pittsburgh',
		  'new york',
		  'san francisco',
		  'los angeles',
		  'las vegas',
		  'phoenix',
		  'el paso',
		  'santa fe',
		  'denver',
		  'kansas city',
		  'oklahoma city',
		  'dallas',
		  'houston',
		  'new orleans',
		  'little rock',
		  'nashville',
		  'atlanta',
		  'miami',
		  'charleston',
		  'raleigh',
		  'washington',
		  'saint louis']

CITIES.sort()

CITY_INDEXES = {city:i for i, city in enumerate(CITIES)}

N_CITIES = len(CITIES)

#each city contains a list of other cities that it is connected to
CITY_CITY_CONNECTIONS = defaultdict(dict)
#the full representation of each edge/track
TRACKS = []

with open(TRACK_DATA_FILENAME, 'r') as f:
	reader = csv.reader(f)
	next(reader)
	for i, row in enumerate(reader):
		city1 = row[0]
		city2 = row[1]
		color1 = row[2]
		color2 = row[3]
		length = int(row[4])
		double = len(color2) > 0
		TRACKS.append({'cities':frozenset([city1,city2]),
			'color1':color1,'color2':color2,
			'length':length,
			'double':double,
			'index':i})
		CITY_CITY_CONNECTIONS[city1].update({city2:[i, length]})
		CITY_CITY_CONNECTIONS[city2].update({city1:[i, length]})

N_TRACKS = len(TRACKS)

#print CITY_CITY_CONNECTIONS
#print len(CITIES)
#print len(CITY_CITY_CONNECTIONS)
#print len(TRACKS)

TRACK_DECK_LIST = {'green':12,
              'red':12,
              'yellow':12,
              'black':12,
              'white':12,
              'orange':12,
              'pink':12,
              'blue':12,
              'locomotive':14}

TRACK_DECK = reduce(list.__add__, [[color]*n for color, n in TRACK_DECK_LIST.iteritems()])

DECK_COLORS = TRACK_DECK_LIST.keys()
DECK_COLORS.sort()

#for fast lookup; excludes locomotive for BASE_DECK* so that color costs of tracks can be represented
DECK_COLOR_INDEXES = {color:i for i, color in enumerate(DECK_COLORS)}

BASE_COLORS = ['green','red','yellow','black','white','orange','pink','blue','gray']
BASE_COLORS.sort()
BASE_DECK_COLOR_INDEXES = {color:i for i, color in enumerate(BASE_COLORS)}

N_BASE_COLORS = len(BASE_DECK_COLOR_INDEXES)#9

BASE_TICKETS = []
BIGCITIES_TICKETS = []

#all of the tickets can be serialized in advance to save time and code complexity
def serialize_ticket(ticket):
		"""
		serializes a selection of tickets to describe a player's own game state
		"""
		city1, city2 = [city for city in ticket['cities']]
		serialization = np.zeros((N_CITIES + 1))
		serialization[CITY_INDEXES[city1]] = 1
		serialization[CITY_INDEXES[city2]] = 1
		serialization[N_CITIES] = ticket['points']/10.
		return serialization

with open(BASE_TICKETS_FILENAME, 'r') as f:
	reader = csv.reader(f)
	next(reader)
	for i, row in enumerate(reader):
		city1 = row[0]
		city2 = row[1]
		points = int(row[2])
		BASE_TICKETS.append({'cities':frozenset([city1, city2]),
			                 'points':points,
			                 'index':i})
		BASE_TICKETS[-1]['serialization'] = serialize_ticket(BASE_TICKETS[-1])

	MAX_BASE_INDEX = i

with open(BIGCITIES_TICKETS_FILENAME, 'r') as f:
	reader = csv.reader(f)
	next(reader)
	for i, row in enumerate(reader):
		city1 = row[0]
		city2 = row[1]
		points = int(row[2])
		BIGCITIES_TICKETS.append({'cities':frozenset([city1, city2]),
			                 'points':points,
			                 'index':i + 1 + MAX_BASE_INDEX})

#print BASE_TICKETS
#print BIGCITIES_TICKETS

"""
ticket_versions - 'base'/'big_cities'/'both' - what versions of ticket cards are being used?
memory - [integer] - how many previous turns' states should be remembered?
discount - [0., 1.] - how much weight is put on victory (1) vs. short-term points?
temperature - [0., 1. (or infinity)] - float representing how random decisions should be based on 
excess_ticket_limit - [integer] - What is the largest number of unfinished tickets a player can have?
                      this is very important, as it limits the architecture of the network under 
                      certain circumstances
big_ticket_limit - [integer]/None - If an integer, players will initially be given a random set of 
                   tickets that includes a ticket of at least this value; tickets of this value or 
                   greater will not be drawable during the game
"""

#the model config has a list of tuples of the form (delay, weight), allowing influence of varying amounts
#the default is 60% delay=1, 25% delay=2, 15% delay=3
#the network should always have the same ones (or at the very least the same amount) and same order when training, even if the raw
#weights are different
DEFAULT_MODEL_DISCOUNT_CONFIG = [(1, 0.6), (2, 0.25), (3, 0.15)]

DEFAULT_GAME_CONFIG = {'ticket_versions':'base',
                       'memory':0,
                       'discount':0.8,
                       'temperature':1.,
                       'excess_ticket_limit':7,
                       'big_ticket_limit':None,
                       'model_discount_config':DEFAULT_MODEL_DISCOUNT_CONFIG}

#number of points by segment length
SEGMENT_POINTS = [0, 1, 2, 4, 7, 10, 15]