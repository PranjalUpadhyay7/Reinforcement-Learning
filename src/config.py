from dataclasses import dataclass, field
from typing import Tuple, List, Optional

@dataclass
class Q1Config:
    grid_size: Tuple[int, int] = (10, 10)
    start_state: Tuple[int, int, int] = (0, 0, 0) 
    goal_state: Tuple[int, int] = (9, 9)
    num_obstacles: int = 5 
    obstacles: Optional[List[Tuple[int, int]]] = None # [(1,2), (3,5)]
    prob_forward: float = 0.8  
    prob_slip: float = 0.1     
    reward_step: float = -1.0       
    reward_collision: float = -100.0 
    reward_goal: float = 50.0    
    gamma: float = 0.99
    threshold: float = 1e-4


@dataclass
class Q2Config:
    grid_size: Tuple[int, int] = (10, 10)
    start_state: Tuple[int, int, int] = (0, 0, 5) 
    goal_state: Tuple[int, int] = (9, 9)
    max_battery: int = 5
    num_stations: int = 10
    charging_stations: Optional[List[Tuple[int, int]]] = None
    reward_step: float = -1.0     
    reward_recharge: float = -2.0   
    reward_failure: float = -100.0 
    reward_collision: float = -50.0
    reward_goal: float = 100.0      
    gamma: float = 0.99 
    threshold: float = 1e-5


@dataclass
class Q3Config:
    grid_size: Tuple[int, int] = (10, 10)
    
    start_state: Tuple[int, int, int] = (0, 0, 0) 
    goal_state: Tuple[int, int] = (9, 9)
    
    num_hazards: int = 15 
    hazards: Optional[List[Tuple[int, int]]] = None
    
    prob_success_normal: float = 1.0  
    prob_slip_hazard: float = 0.1  
    
    reward_step: float = -1.0       
    reward_failure: float = -200.0 
    reward_goal: float = 50.0       
    
    gamma: float = 0.99
    threshold: float = 1e-4

@dataclass
class Q3Config:
    grid_size: Tuple[int, int] = (10, 10)
    start_state: Tuple[int, int] = (0, 0) 
    goal_state: Tuple[int, int] = (9, 9)
    num_hazards: int = 15 
    hazards: Optional[List[Tuple[int, int]]] = None
    prob_slip_hazard: float = 0.2
    reward_step: float = -1.0 
    reward_failure: float = -200.0
    reward_goal: float = 100.0 
    gamma: float = 0.99
    threshold: float = 1e-4
    mc_episodes: int = 5000