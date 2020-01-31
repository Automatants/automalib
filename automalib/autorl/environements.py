'''
Personalised environements based on Gym and PyGame
'''

from gym import Env, spaces
import numpy as np
import pygame
import os


# Trivial Nim environement


class _NimGame():
    def __init__(self, players=("Player 1", "Player 2"), actions=(1,2,3), nb_sticks=10):
        self.players = players
        self.losers = np.zeros(len(players))
        self.actions = actions
        self.max_sticks = nb_sticks
        self.sticks_left = nb_sticks
        self.turn = 0
    
    def is_legal(self, sticks):
        """ Vérifie que l'action est authorisée """
        return sticks in self.actions

    def remove(self, sticks):
        """ Enlève les batons aux batons restants """
        self.sticks_left -= sticks
        if self.sticks_left < 1:
          self.losers[self.turn] = 1
        self.turn = (self.turn + 1) % len(self.players)

    def won(self, player_id):
        """ Si il n'y a plus de baton et que c'est votre tour
        alors vous avez gagnez """
        return self.losers[1-player_id]==1

    def get_observation(self):
        """ Return the game state """
        return int(max(self.sticks_left, 0))
    
    def render(self):
        """ Print the sticks """
        sticks = np.zeros(self.max_sticks)
        sticks[:self.sticks_left+1] = 1
        print(sticks, print(self.sticks_left))
        return


class NimEnv(Env):

    def __init__(self, players=("Agent", "Environement"), actions=(1,2,3), nb_sticks=20, is_optimal=False):
        self.players = players
        self.is_optimal = is_optimal
        self.nb_sticks = nb_sticks
        self.game = _NimGame(players=players, actions=actions, nb_sticks=nb_sticks)
        self.actions = actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(nb_sticks + 1)

    def step(self, action):
        reward = 0
        done = False

        # Agent turn
        sticks_to_remove_by_agent = self.actions[action]

        if self.game.is_legal(sticks_to_remove_by_agent):
            if self.game.won(0):
                reward = 1
                done = True
            else:
                self.game.remove(sticks_to_remove_by_agent)
        else:
            print("Warning ! {} is an illegal move ! Played random instead ...".format(sticks_to_remove_by_agent))
            reward = -1
            self.game.remove(np.random.choice(self.actions))

        # Environement turn
        def env_policy(state, actions, random=True):
            # choice = rd.choice(actions)
            if (state - 1)%4==0 or random:
              choice = np.random.choice(actions)
            else:
              choice = (state - 1)%4
            return choice
        
        sticks_to_remove_by_env = env_policy(self.game.get_observation(), self.actions, not self.is_optimal)
        if self.game.won(1) and not done:
            reward = -1
            done = True
        else:
            self.game.remove(sticks_to_remove_by_env)
        
        observation = self.game.get_observation()

        return observation, reward, done, {}

    def reset(self):
        self.game = _NimGame(players=self.players, actions=self.actions,
                            nb_sticks=self.nb_sticks)
        return self.game.get_observation()

    def render(self, mode='human'):
        self.game.render()

    def close(self):
        pass


# Santorini environement


class SantoriniGym():

    def __init__(self, players=("Player 1", "Player 2"), graphics=True, board_shape=(5, 5), nb_pawns=2):
        """
        Initialise the game with board an pawns
        :param players: the names of all players
        :param graphics: should we display the game ?
        :param board_shape: the shape of the board
        :param nb_pawns: the number of builder per player
        """
        self.board = np.zeros(board_shape, dtype=int)
        self.pawns = np.ones((len(players), nb_pawns, 2), dtype=int)*-1
        self.losers = np.zeros(len(players))
        self.players = players
        self.graphics = graphics
        self.board_shape = board_shape
        self.nb_pawns = nb_pawns
        self.pygame_init = False
        self.actions = []

        for pawn in range(nb_pawns):
            for move in range(8):
                for build in range(8):
                    self.actions.append((pawn, move, build))

        for pawn_id in range(nb_pawns):
            for player_id in range(len(self.players)):
                self._place(player_id, pawn_id)

    def _place(self, player_id, pawn_id):
        """
        Place the pawn number "pawn_id" of the player "player" if possible
        :param player: the number of the player
        :param pawn_id: the ID of the pawn
        :param coordinates: aimed coordinates
        """
        coordinates = (-1, -1)

        # Take the list of all pawns positions
        k = len(self.pawns)*len(self.pawns[0])
        list_coordinates_pawns = np.reshape(self.pawns, (k, 2))
        list_coordinates_pawns = [tuple(list_coordinates_pawns[i, :]) for i in range(k)]

        # Build the list of authorised placements
        authorised_placement = []
        for x in range(self.board_shape[0]):
            for y in range(self.board_shape[1]):
                if (x, y) not in list_coordinates_pawns:
                    authorised_placement.append((x, y))

        # Validate the placement
        coordinates = (np.random.randint(0, self.board_shape[0]), np.random.randint(0, self.board_shape[1]))
        while coordinates not in authorised_placement:
            coordinates = (np.random.randint(0, self.board_shape[0]), np.random.randint(0, self.board_shape[1]))

        # Place the pawn
        self.pawns[player_id, pawn_id] = coordinates
        return

    def move(self, player_id, pawn_id, direction):
        """
        Move the pawn number "pawn_id" of the player "player" if possible
        :param player: the number of the player
        :param pawn_id: the ID of the pawn
        :param coordinates: aimed coordinates
        """
        dir_list = [(-1, -1), (0, -1), (1, -1), (1, 0),
                    (1, 1), (0, 1), (-1, 1), (-1, 0)]

        x, y = self.pawns[player_id, pawn_id, 0], self.pawns[player_id, pawn_id, 1]
        coordinates = (x + dir_list[direction][0], y + dir_list[direction][1])

        can_do, authorised_moves, blocked = self.get_authorised(self.pawns[player_id, pawn_id], direction, "move")

        # print("move", (x, y), coordinates, authorised_moves, can_do, blocked, player_id)

        # If the player is blocked, he loses
        if blocked:
            self.losers[player_id] = 1
            return

        # Move the pawn (he will move back if not valid)
        if self.losers[player_id] == 0:
            self.pawns[player_id, pawn_id] = coordinates
            if can_do:
                if self.board[coordinates[0], coordinates[1]] == 3:
                    for player in range(len(self.players)):
                        if player != player_id:
                            self.losers[player] = 1

    def move_back(self, player_id, pawn_id, direction):
        """
        Move the pawn number "pawn_id" of the player "player" anyway
        :param player: the number of the player
        :param pawn_id: the ID of the pawn
        :param coordinates: aimed coordinates
        """
        dir_list = [(-1, -1), (0, -1), (1, -1), (1, 0),
                    (1, 1), (0, 1), (-1, 1), (-1, 0)]

        x, y = self.pawns[player_id, pawn_id, 0], self.pawns[player_id, pawn_id, 1]
        coordinates = (x - dir_list[direction][0], y - dir_list[direction][1])
        # print("move_back", (x, y), coordinates, player_id)

        # Move the pawn back
        if self.losers[player_id] == 0:
            self.pawns[player_id, pawn_id] = coordinates

    def build(self, player_id, pawn_id, direction):
        """
        Build around the pawn number "pawn_id" of the player "player" if possible
        :param player: the number of the player
        :param pawn_id: the ID of the pawn
        :param coordinates: aimed coordinates
        """
        dir_list = [(-1, -1), (0, -1), (1, -1), (1, 0),
                    (1, 1), (0, 1), (-1, 1), (-1, 0)]

        x, y = self.pawns[player_id, pawn_id, 0], self.pawns[player_id, pawn_id, 1]

        can_do, authorised_build, blocked = self.get_authorised(self.pawns[player_id, pawn_id], direction, "build")
        # print("build", authorised_build, blocked, player_id)

        # If the player is blocked, he loses
        if blocked:
            self.losers[player_id] = 1
            return

        # If the action in authorised, the player plays
        if can_do:
            # Build at the given coordinates if not dead
            coordinates = (x + dir_list[direction][0], y + dir_list[direction][1])
            if not self.check_dead(player_id):
                self.board[coordinates[0], coordinates[1]] += 1

    def get_authorised(self, pawn_coordinates, direction, action_type):

        dir_list = [(-1, -1), (0, -1), (1, -1), (1, 0),
                    (1, 1), (0, 1), (-1, 1), (-1, 0)]

        x, y = pawn_coordinates[0], pawn_coordinates[1]

        # Build pawn positions list
        pawn_list = []
        for every_player in range(len(self.pawns)):
            for every_pawn in range(len(self.pawns[0])):
                pawn_list.append(tuple(self.pawns[every_player, every_pawn]))

        # Build authorised build list
        authorised = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (i, j) != (0, 0) \
                        and x+i >= 0 and y+j >= 0 \
                        and x+i < self.board_shape[0] and y+j < self.board_shape[1] \
                        and self.board[x+i, y+j] < 4:

                    close_enough = True
                    if action_type == "move":
                        close_enough = self.board[x+i, y+j] - self.board[x, y] < 2

                    if close_enough and (x+i, y+j) not in pawn_list:
                        authorised.append((x+i, y+j))
        coordinates = (x + dir_list[direction][0], y + dir_list[direction][1])
        return (coordinates in authorised), authorised, len(authorised) == 0

    def get_authorised_tuples(self, player_id):
        dir_list = [(-1, -1), (0, -1), (1, -1), (1, 0),
                    (1, 1), (0, 1), (-1, 1), (-1, 0)]

        authorised_tuples = []

        for pawn_id in range(self.nb_pawns):
            for move_direction in range(8):
                if self.get_authorised(self.pawns[player_id, pawn_id], move_direction, "move")[0]:
                    pawn_coordinates = [self.pawns[player_id, pawn_id][0] + dir_list[move_direction][0],
                                        self.pawns[player_id, pawn_id][1] + dir_list[move_direction][1]]
                    self.move(player_id, pawn_id, move_direction)
                    for build_direction in range(8):
                        if self.get_authorised(pawn_coordinates, build_direction, "build")[0]:
                            authorised_tuples.append([pawn_id, move_direction, build_direction])
                    self.move_back(player_id, pawn_id, move_direction)

        # print("\n{}/{} authorised_moves".format(len(authorised_tuples), 64*self.nb_pawns))
        return authorised_tuples

    def check_dead(self, player_id):
        return self.losers[player_id] == 1

    def check_win(self, player_id):
        for player in range(len(self.players)):
            if player == player_id:
                if self.losers[player] == 1:
                    return False
            else:
                if self.losers[player] == 0:
                    return False
        return True

    def get_observations(self, player_id):
        board_one_hot = np.ones(self.board_shape + (5,))
        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1]):
                for k in range(self.board[i, j]):
                    board_one_hot[i, j, k] = 0

        pawns_one_hot = np.zeros(self.board_shape + (len(self.players)*self.nb_pawns,))
        for player in range(len(self.players)):
            for pawn in range(self.nb_pawns):
                pawns_one_hot[self.pawns[player, pawn, 0], self.pawns[player, pawn, pawn], player*self.nb_pawns + pawn] = 1
        for pawn in range(self.nb_pawns):
            pawns_one_hot[:, :, [pawn, player_id*self.nb_pawns + pawn]] = pawns_one_hot[:, :, [player_id*self.nb_pawns + pawn, pawn]]

        player_signature = np.ones(self.board_shape + (1,))*player_id

        game_state = np.concatenate([board_one_hot, pawns_one_hot, player_signature], axis=-1)
        authorised_tuples = self.get_authorised_tuples(player_id)
        authorised_actions = [not action in authorised_tuples for action in self.actions]

        return game_state, authorised_actions

    def render(self):
        if self.graphics and not self.pygame_init:
            # pygame.init()
            self.pygame_init = True
            
            # Open Pygame window
            self.window = pygame.display.set_mode(tuple(np.array(self.board_shape)*90))

            # Load images
            path = os.path.join('automalib', 'autorl', 'santorini_data')
            self.floors = [pygame.image.load(os.path.join(path, "floor_{}.png".format(i))).convert() for i in range(5)]
            self.builders = [pygame.image.load(os.path.join(path, "builder_{}.png".format(i))).convert_alpha() for i in range(1, 7)]
            

        if self.graphics:
            for i in range(self.board_shape[0]):
                for j in range(self.board_shape[1]):
                    self.window.blit(self.floors[self.board[i, j]], tuple(np.array((i, j))*90))
            for player_id in range(len(self.players)):
                for pawn in range(self.nb_pawns):
                    self.window.blit(self.builders[player_id], tuple(np.array(self.pawns[player_id, pawn]*90)))
                    pygame.display.flip()
            pygame.time.wait(1)


class SantoriniEnv(Env):

    def __init__(self, players=("Player 1", "Player 2"), graphics=True, board_shape=(5, 5), nb_pawns=2):

        self.players = players
        self.graphics = graphics
        self.board_shape = board_shape
        self.nb_pawns = nb_pawns
        self.game = SantoriniGym(self.players, self.graphics, self.board_shape, self.nb_pawns)

        self.actions = []
        for pawn in range(nb_pawns):
            for move in range(8):
                for build in range(8):
                    self.actions.append((pawn, move, build))
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 1, board_shape + (5 + len(players)*nb_pawns + 1,), dtype=int),
            spaces.Box(0, 1, (nb_pawns*64,), dtype=int)
        ))

    def step(self, action):
        reward = 0
        dir_list = [(-1, -1), (0, -1), (1, -1), (1, 0),
                    (1, 1), (0, 1), (-1, 1), (-1, 0)]

        # Agent turn
        pawn, move, build = self.actions[action]

        # If move is valid
        pass_turn = False
        authorised_tuples = self.game.get_authorised_tuples(0)
        if not [pawn, move, build] in authorised_tuples:
            pass_turn = True

        # Move_back if turn is passed, else build
        if not pass_turn:
            # Agent actions
            self.game.move(0, pawn, move)
            self.game.build(0, pawn, build)

            # Random turn
            pawn = np.random.randint(0, 2)
            move = np.random.randint(0, 8)
            i = 0
            while not self.game.get_authorised(self.game.pawns[1, pawn], move, "move")[0] and i < 128:
                pawn = np.random.randint(0, 2)
                move = np.random.randint(0, 8)
                i += 1
            self.game.move(1, pawn, move)

            build = np.random.randint(0, 8)
            i = 0
            while not self.game.get_authorised(self.game.pawns[1, pawn], build, "build")[0] and i < 128:
                build = np.random.randint(0, 8)
                i += 1
            self.game.build(1, pawn, build)

        player_0_lost = self.game.check_dead(0)
        player_0_win = self.game.check_win(0)
        player_1_lost = self.game.check_dead(1)
        player_1_win = self.game.check_win(1)

        # Reward
        if player_0_lost or player_1_win:
            reward -= 1

        if player_0_win or player_1_lost:
            reward += 1

        done = player_0_lost or player_1_lost or player_0_win or player_1_win
        observation = self.game.get_observations(0)
        self.game.render()
        return observation, reward, done, {}

    def reset(self):
        self.game = SantoriniGym(self.players, self.graphics, self.board_shape, self.nb_pawns)
        return self.game.get_observations(0)

    def render(self, mode='human'):
        self.game.render()


if __name__ == "__main__":
    env = SantoriniEnv(graphics=True)
    observation = env.reset()
    done = False
    G = 0
    while not done:
        action = np.random.randint(0, env.action_space.n)
        observation, reward, done, _ = env.step(action)
        G += reward
    print(reward)