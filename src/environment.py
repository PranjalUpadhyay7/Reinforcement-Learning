import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from src.config import Q1Config
from src.config import Q3Config

random.seed(0)
class MDPEnvironment(ABC):
    def __init__(self, config: Q1Config):
        self.config = config
        self.grid_size = config.grid_size
        # self.obstacles = self._setup_obstacles()
        self.states = self._define_state_space()
        self.actions = self._define_action_space()
        self.current_state = self.config.start_state

    @abstractmethod
    def _define_state_space(self):
        pass

    @abstractmethod
    def _define_action_space(self):
        pass

    @abstractmethod
    def get_transition_dynamics(self, state, action):
        pass

    def reset(self):
        self.current_state = self.config.start_state
        return self.current_state

    def step(self, action):
        transitions = self.get_transition_dynamics(self.current_state, action)
        probs = [t[0] for t in transitions]
        indices = range(len(transitions))
        chosen_idx = random.choices(indices, weights=probs, k=1)[0]
        prob, next_state, reward, done = transitions[chosen_idx]
        
        if next_state is not None:
            self.current_state = next_state
            
        return next_state, reward, done

    
class DifferentialDriveRobot(MDPEnvironment):
    def __init__(self, config: Q1Config):
        super().__init__(config)
        self.obstacles = self._setup_obstacles()

    def _define_state_space(self):
        states = []
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for theta in [0, 1, 2, 3]: 
                    states.append((x, y, theta))
        return states

    def _define_action_space(self):
        return ["TurnLeft", "TurnRight", "Forward"]
    
    def _setup_obstacles(self):
        if self.config.obstacles is not None:
            return self.config.obstacles
        else:
            return self._generate_random_obstacles()

    def _generate_random_obstacles(self):
        all_coordinates = [
            (r, c) 
            for r in range(self.config.grid_size[0]) 
            for c in range(self.config.grid_size[1])
        ]
        start_pos = (self.config.start_state[0], self.config.start_state[1])
        goal_pos = self.config.goal_state
        
        candidates = [
            pos for pos in all_coordinates 
            if pos != start_pos and pos != goal_pos
        ]
        random.seed(0)
        return random.sample(candidates, self.config.num_obstacles)

    def is_valid(self, x, y):
        if x < 0 or x >= self.grid_size[0] or y < 0 or y >= self.grid_size[1]:
            return False
        if (x, y) in self.obstacles:
            return False
        return True


    def get_transition_dynamics(self, state, action):
        x, y, theta = state
        transitions = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        if action == "TurnLeft":
            new_theta = (theta - 1) % 4
            transitions.append((1.0, (x, y, new_theta), self.config.reward_step, False))

        elif action == "TurnRight":
            new_theta = (theta + 1) % 4
            transitions.append((1.0, (x, y, new_theta), self.config.reward_step, False))

        elif action == "Forward":
            outcomes = [
                (theta, self.config.prob_forward),           
                ((theta - 1) % 4, self.config.prob_slip),   
                ((theta + 1) % 4, self.config.prob_slip)     
            ]

            for move_dir_idx, prob in outcomes:
                dx, dy = directions[move_dir_idx]
                next_x, next_y = x + dx, y + dy
                
                if (next_x, next_y) == self.config.goal_state:
                     transitions.append((prob, (next_x, next_y, theta), self.config.reward_goal, True))

                elif not self.is_valid(next_x, next_y):
                    transitions.append((prob, state, self.config.reward_collision, True))
                
                else:
                    transitions.append((prob, (next_x, next_y, theta), self.config.reward_step, False))
        
        return transitions


class BatteryAwareRobot(MDPEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.charging_stations = self._setup_charging_stations()
        
    def _setup_charging_stations(self):
        if getattr(self.config, 'charging_stations', None) is not None:
            return self.config.charging_stations
        else:
            return self._generate_random_charging_stations()

    def _generate_random_charging_stations(self):
        all_coordinates = [
            (r, c) 
            for r in range(self.config.grid_size[0]) 
            for c in range(self.config.grid_size[1])
        ]
        start_pos = (self.config.start_state[0], self.config.start_state[1])
        goal_pos = self.config.goal_state
        
        candidates = [
            pos for pos in all_coordinates 
            if pos != start_pos and pos != goal_pos
        ]
        random.seed(9)
        return random.sample(candidates, self.config.num_stations)

    def is_valid(self, x, y):
        if x < 0 or x >= self.grid_size[0] or y < 0 or y >= self.grid_size[1]:
            return False
        return True
        
    def is_charging_station(self, x, y):
        return (x, y) in self.charging_stations
    
    def _define_state_space(self):
        states = []
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for battery in range(self.config.max_battery + 1): 
                    states.append((x, y, battery))
        return states

    def _define_action_space(self):
        return ["left", "right", "up", "down", "charge"]
    
    def get_transition_dynamics(self, state, action):
        x, y, battery = state
        
        if battery == 0 and not self.is_charging_station(x, y):
            return [(1.0, (x, y, 0), 0.0, True)]
        if (x, y) == self.config.goal_state:
            return [(1.0, (x, y, battery), 0.0, True)]

        transitions = []
        directions = {"charge": (0, 0), "right": (1, 0), "down": (0, 1), "left": (-1, 0), "up": (0, -1)}

        dx, dy = directions[action]
        next_x, next_y = x + dx, y + dy

        is_collision = not self.is_valid(next_x, next_y)
        if is_collision:
            next_x, next_y = x, y 

        if action == "charge":
            if self.is_charging_station(x, y):
                new_battery = self.config.max_battery
                transitions.append((1.0, (x, y, new_battery), self.config.reward_recharge, False))
                return transitions
            else:
                new_battery = battery - 1 
        else:
            new_battery = battery - 1
            
        # Guarantee battery is never negative in the state tuple
        final_battery = max(0, new_battery)
            
        if is_collision:
            reward = getattr(self.config, 'reward_collision', -100.0)
            transitions.append((1.0, (next_x, next_y, final_battery), reward, True))
        elif (next_x, next_y) == self.config.goal_state:
            transitions.append((1.0, (next_x, next_y, final_battery), self.config.reward_goal, True))
        elif new_battery <= 0 and not self.is_charging_station(next_x, next_y):
            reward = getattr(self.config, 'reward_failure', -100.0)
            transitions.append((1.0, (next_x, next_y, 0), reward, True))
        else:
            transitions.append((1.0, (next_x, next_y, final_battery), self.config.reward_step, False))

        return transitions
    


import random
import numpy as np
from .config import Q3Config
from .environment import MDPEnvironment

class RiskSensitiveRobot(MDPEnvironment):
    def __init__(self, config: Q3Config):
        super().__init__(config)
        self.hazards = self._setup_hazards()

    def _define_state_space(self):
        return [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]

    def _define_action_space(self):
        return ["up", "down", "left", "right"]

    def _setup_hazards(self):
        if self.config.hazards is not None:
            return self.config.hazards
        all_coords = [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]
        candidates = [c for c in all_coords if c != self.config.start_state and c != self.config.goal_state]
        random.seed(42)
        return random.sample(candidates, self.config.num_hazards)

    def is_valid(self, x, y):
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]

    def get_transition_dynamics(self, state, action):
        if state == self.config.goal_state:
            return [(1.0, state, 0.0, True)]

        x, y = state
        moves = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        dx, dy = moves[action]
        next_x, next_y = x + dx, y + dy

        if not self.is_valid(next_x, next_y):
            next_x, next_y = x, y

        next_state = (next_x, next_y)
        transitions = []

        if next_state in self.hazards:
            transitions.append((self.config.prob_slip_hazard, next_state, self.config.reward_failure, True))
            transitions.append((1.0 - self.config.prob_slip_hazard, next_state, self.config.reward_step, False))
        else:
            is_goal = (next_state == self.config.goal_state)
            reward = self.config.reward_goal if is_goal else self.config.reward_step
            transitions.append((1.0, next_state, reward, is_goal))

        return transitions