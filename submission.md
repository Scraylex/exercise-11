# Submission Exercise 11

Github Fork: https://github.com/Scraylex/exercise-11

## Task 1)

To design the states, actions, and rewards for the illuminance controller agent using Q-learning, we have to consider as for any model free control:

### States

The states represent the current conditions of the system that the agent can observe. In this case, the states should include the indoor illuminance levels in both workstation zones and the outdoor illuminance level. Each workstation zone can be represented by a rank value (0 to 3) indicating the current indoor illuminance level. Additionally, the outdoor illuminance level can also be represented by a rank value (0 to 3). Therefore, a state can be defined as a tuple (rank_ws1, rank_ws2, rank_outdoor).

### Actions

The actions represent the decisions or control actions that the agent can take. In this case, the agent can control the ceiling lights and the blinds. The actions can be defined as a combination of turning the lights on/off and raising/lowering the blinds. Therefore, the action space can be defined as a tuple (ceiling_lights_action, blinds_action) where each action can take values from a predefined set (e.g., {0, 1} for lights off/on and {0, 1} for blinds lowered/raised).

### Rewards

In this case, the rewards should be designed to encourage the agent to achieve the desired illuminance levels in both workstation zones. A reward can be defined based on the proximity of the current illuminance levels to the desired levels. For example, a positive reward can be given when the illuminance levels match the desired ranks in both workstations, and a negative reward can be given for large deviations from the desired ranks. Further we could factor the required energy into the reward function to also add the dimension of energy efficiency of solutions.


This would leave us with something like:

| State | Lights on | Lights off | Blind lowered | Blinds Raised |
| ----- | --------- | ---------- | ------------- | ------------- |
| 0,0,0 | x1        | x2         | x3            | x4            |
| ...   | ...       | ...        | ...           | ...           |
| 3,3,3 | y1        | y2         | y3            | y4            |



## Argument for using reinforcement learning:

- Complex environment: The smart factory environment described involves multiple devices, sensor measurements, and the need to balance indoor illuminance levels. Reinforcement learning provides a framework to learn optimal actions in such complex environments.
- Unknown optimal actions: The agent does not know the optimal combination of blinds and lights to achieve the desired illuminance levels. Reinforcement learning allows the agent to explore and learn the best actions through trial and error.
- Dynamic nature: The illuminance requirements may change over time due to different tasks assigned to the workstations. Reinforcement learning can adapt to dynamic environments and learn to optimize the illuminance levels based on the current requirements.

## Argument against using reinforcement learning:

- Simpler alternatives: If the environment and the set of actions are relatively simple, rule-based or deterministic approaches may be sufficient to achieve the desired illuminance levels.

In this toy example reinforment learning is overkill in real world examples of significant scale the training and modelling requirements provide such complexity that it is almost impossible to bring into production.

## Handling additional constraint

To handle the additional constraint that the blinds should be closed when the outdoor light level is of Rank 3, we can modify the action space and rewards:

- Action space: We can add a special action for closing the blinds when the outdoor light level is of Rank 3. The agent can choose this action only when the outdoor illuminance level is Rank 3.

- Rewards: If the agent takes the action to close the blinds when the outdoor light level is of Rank 3, it should receive a positive reward. This encourages the agent to satisfy the constraint. If the agent takes any other action or fails to close the blinds when required, it should receive a negative reward.

## Task 2)

see qtables.json for the qtables