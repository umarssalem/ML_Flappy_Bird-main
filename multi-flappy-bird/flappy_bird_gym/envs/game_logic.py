# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Implements the logic of the Flappy Bird game.

Some of the code in this module is an adaption of the code in the `FlapPyBird`
GitHub repository by `sourahbhv` (https://github.com/sourabhv/FlapPyBird),
released under the MIT license.
"""


import random
from enum import IntEnum
from itertools import cycle
from typing import Dict, Tuple, Union

import pygame

############################ Speed and Acceleration ############################
PIPE_VEL_X = -4

PLAYER_MAX_VEL_Y = 10  # max vel along Y, max descend speed
PLAYER_MIN_VEL_Y = -8  # min vel along Y, max ascend speed

PLAYER_ACC_Y = 1       # players downward acceleration
PLAYER_VEL_ROT = 3     # angular speed

PLAYER_FLAP_ACC = -9   # players speed on flapping
################################################################################


################################## Dimensions ##################################
PLAYER_WIDTH = 34
PLAYER_HEIGHT = 24

PIPE_WIDTH = 52
PIPE_HEIGHT = 320

BASE_WIDTH = 336
BASE_HEIGHT = 112

BACKGROUND_WIDTH = 288
BACKGROUND_HEIGHT = 512
################################################################################
class Bird:
    """ Defines the state of any bird in the game.

    Args:
        screen_size (Tuple[int, int]): Tuple with the screen's width and height.

    Attributes:
        player_x (int): The player's x position.
        player_y (int): The player's y position.
        score (int): Current score of the player.
        alive (bool): If the player is alive or not
        player_vel_y (int): The player's vertical velocity.
        player_rot (int): The player's rotation angle.
        last_action (Optional[FlappyBirdLogic.Actions]): The last action taken
            by the player. If `None`, the player hasn't taken any action yet.
        player_idx (int): Current index of the bird's animation cycle.
    """
    def __init__(self, 
        screen_size: Tuple[int, int]) -> None:
        self.player_x = int(screen_size[0] * 0.2)
        self.player_y = int((screen_size[1] - PLAYER_HEIGHT) / 2)

        self.score = 0
        self.alive = True

        # Player's info:
        self.player_vel_y = -9  # player"s velocity along Y
        self.player_rot = 45  # player"s rotation

        self.last_action = None

        self._player_flapped = False
        self.player_idx = 0
        self._player_idx_gen = cycle([0, 1, 2, 1])
        self._loop_iter = 0

class FlappyBirdLogic:
    """ Handles the logic of the Flappy Bird game.

    The implementation of this class is decoupled from the implementation of the
    game's graphics. This class implements the logical portion of the game.

    Args:
        screen_size (Tuple[int, int]): Tuple with the screen's width and height.
        pipe_gap_size (int): Space between a lower and an upper pipe.

    Attributes:
        base_x (int): The base/ground's x position.
        base_y (int): The base/ground's y position.
        max_score (int): The current maximum score of all the players (or score of the best performing player).
        upper_pipes (List[Dict[str, int]): List with the upper pipes. Each pipe
            is represented by a dictionary containing two keys: "x" (the pipe's
            x position) and "y" (the pipe's y position).
        lower_pipes (List[Dict[str, int]): List with the lower pipes. Each pipe
            is represented by a dictionary containing two keys: "x" (the pipe's
            x position) and "y" (the pipe's y position).
        sound_cache (Optional[str]): Stores the name of the next sound to be
            played. If `None`, then no sound should be played.
    """

    def __init__(self,
                 screen_size: Tuple[int, int],
                 pipe_gap_size: int = 100,
                 nr_of_birds: int = 1) -> None:
        self._screen_width = screen_size[0]
        self._screen_height = screen_size[1]
        self.nr_of_birds = nr_of_birds

        self.birds = [Bird(screen_size) for k in range(self.nr_of_birds)]

        self.base_x = 0
        self.base_y = self._screen_height * 0.79
        self._base_shift = BASE_WIDTH - BACKGROUND_WIDTH

        self.max_score = 0
        self._pipe_gap_size = pipe_gap_size

        # Generate 2 new pipes to add to upper_pipes and lower_pipes lists
        new_pipe1 = self._get_random_pipe()
        new_pipe2 = self._get_random_pipe()

        # List of upper pipes:
        self.upper_pipes = [
            {"x": self._screen_width - 50,
             "y": new_pipe1[0]["y"]},
            {"x": self._screen_width -50 + (self._screen_width / 2),
             "y": new_pipe2[0]["y"]},
        ]

        # List of lower pipes:
        self.lower_pipes = [
            {"x": self._screen_width - 50,
             "y": new_pipe1[1]["y"]},
            {"x": self._screen_width -50 + (self._screen_width / 2),
             "y": new_pipe2[1]["y"]},
        ]

        self.sound_cache = None


    class Actions(IntEnum):
        """ Possible actions for the player to take. """
        IDLE, FLAP = 0, 1

    def _get_random_pipe(self) -> Dict[str, int]:
        """ Returns a randomly generated pipe. """
        # y of gap between upper and lower pipe
        gap_y = random.randrange(0,
                                 int(self.base_y * 0.6 - self._pipe_gap_size))
        gap_y += int(self.base_y * 0.2)

        pipe_x = self._screen_width + 10
        return [
            {"x": pipe_x, "y": gap_y - PIPE_HEIGHT},          # upper pipe
            {"x": pipe_x, "y": gap_y + self._pipe_gap_size},  # lower pipe
        ]

    def check_crash(self, bird) -> bool:
        """ Returns True if the bird passed as argument collides with the ground (base) or a pipe.
        """
        # if player crashes into ground
        if bird.player_y + PLAYER_HEIGHT >= self.base_y - 1:
            return True
        # if player hits the ceiling  
        elif bird.player_y < 0:
            return True
        else:
            player_rect = pygame.Rect(bird.player_x, bird.player_y,
                                      PLAYER_WIDTH, PLAYER_HEIGHT)

            for up_pipe, low_pipe in zip(self.upper_pipes, self.lower_pipes):
                # upper and lower pipe rects
                up_pipe_rect = pygame.Rect(up_pipe['x'], up_pipe['y'],
                                           PIPE_WIDTH, PIPE_HEIGHT)
                low_pipe_rect = pygame.Rect(low_pipe['x'], low_pipe['y'],
                                            PIPE_WIDTH, PIPE_HEIGHT)

                # check collision
                up_collide = player_rect.colliderect(up_pipe_rect)
                low_collide = player_rect.colliderect(low_pipe_rect)

                if up_collide or low_collide:
                    return True

        return False

    def update_state(self, actions: Union[Actions, int]) -> bool:
        """ Given a list of actions taken, updates the game's state.

        Args:
            actions (list of Union[FlappyBirdLogic.Actions, int]): The actions taken by
                the player.

        Returns:
            list of booleans that indicate for each bird individually if it is alive (`True`) or not (`False`).
        """
        to_return = [bird.alive for bird in self.birds]

        self.base_x = -((-self.base_x + 100) % self._base_shift)

        max_points = 0
        for i in range(self.nr_of_birds):
            bird = self.birds[i]
            if bird.alive:
                self.sound_cache = None
                if actions[i] == FlappyBirdLogic.Actions.FLAP:
                    if bird.player_y > -2 * PLAYER_HEIGHT:
                        bird.player_vel_y = PLAYER_FLAP_ACC
                        bird._player_flapped = True
                        # self.sound_cache = "wing"

                bird.last_action = actions[i]
                if self.check_crash(bird):
                    # self.sound_cache = "hit"
                    bird.alive = False
                    to_return[i] = False

                # check for score
                player_mid_pos = bird.player_x + PLAYER_WIDTH / 2
                for pipe in self.upper_pipes:
                    pipe_mid_pos = pipe['x'] + PIPE_WIDTH / 2
                    if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                        bird.score += 1
                        max_points = max(max_points, bird.score)
                        # self.sound_cache = "point"

                # player_index base_x change
                if (bird._loop_iter + 1) % 3 == 0:
                    bird.player_idx = next(bird._player_idx_gen)

                bird._loop_iter = (bird._loop_iter + 1) % 30

                # rotate the player
                if bird.player_rot > -90:
                    bird.player_rot -= PLAYER_VEL_ROT

                # player's movement
                if bird.player_vel_y < PLAYER_MAX_VEL_Y and not bird._player_flapped:
                    bird.player_vel_y += PLAYER_ACC_Y

                if bird._player_flapped:
                    bird._player_flapped = False

                    # more rotation to cover the threshold
                    # (calculated in visible rotation)
                    bird.player_rot = 45

                bird.player_y += min(bird.player_vel_y,
                                    self.base_y - bird.player_y - PLAYER_HEIGHT)

        self.max_score = max(self.max_score,max_points)

        # move pipes to left
        for up_pipe, low_pipe in zip(self.upper_pipes, self.lower_pipes):
            up_pipe['x'] += PIPE_VEL_X
            low_pipe['x'] += PIPE_VEL_X

        # add new pipe when first pipe is about to touch left of screen
        if len(self.upper_pipes) > 0 and 0 < self.upper_pipes[0]['x'] < 5:
            new_pipe = self._get_random_pipe()
            self.upper_pipes.append(new_pipe[0])
            self.lower_pipes.append(new_pipe[1])

        # remove first pipe if its out of the screen
        if (len(self.upper_pipes) > 0 and
                self.upper_pipes[0]['x'] < -PIPE_WIDTH):
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        return to_return
