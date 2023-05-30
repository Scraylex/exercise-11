//illuminance controller agent

/*
* The URL of the W3C Web of Things Thing Description (WoT TD) of a lab environment
* Simulated lab WoT TD: "https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl"
* Real lab WoT TD: "https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab-real.ttl"
*/

/* Initial beliefs and rules */

// the agent has a belief about the location of the W3C Web of Thing (WoT) Thing Description (TD)
// that describes a lab environment to be learnt
learning_lab_environment("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl").

// the agent believes that the task that takes place in the 1st workstation requires an indoor illuminance
// level of Rank 2, and the task that takes place in the 2nd workstation requires an indoor illumincance 
// level of Rank 3. Modify the belief so that the agent can learn to handle different goals.
task_requirements([2,3]).

/* Initial goals */
!start. // the agent has the goal to start

/* 
 * Plan for reacting to the addition of the goal !start
 * Triggering event: addition of goal !start
 * Context: the agent believes that there is a WoT TD of a lab environment located at Url, and that 
 * the tasks taking place in the workstations require indoor illuminance levels of Rank Z1Level and Z2Level
 * respectively
 * Body: (currently) creates a QLearnerArtifact and a ThingArtifact for learning and acting on the lab environment.
*/
@start
+!start : learning_lab_environment(Url) 
  & task_requirements([Z1Level, Z2Level]) <-

  .print("Hello world");
  .print("I want to achieve Z1Level=", Z1Level, " and Z2Level=",Z2Level);

  // creates a QLearner artifact for learning the lab Thing described by the W3C WoT TD located at URL
  makeArtifact("qlearner", "tools.QLearner", [Url], QLArtId);

  // creates a ThingArtifact artifact for reading and acting on the state of the lab Thing
  makeArtifact("lab", "wot.ThingArtifact", [Url], LabArtId);
  // goalDescription, episodes, alphaObj, gamma, epsilon, reward
  calculateQ([2,3], 50, 0.2, 0.8, 0.2, 100)[artifact_id(QLArtId)];
  getCurrentZLevels(CurrentZLevels)[artifact_id(QLArtId)] ;
  +current_zlevels(CurrentZLevels);
  getCurrentLabState(CurrentLabState);
  !take_action(CurrentLabState).

@take_action_plan
+!take_action(CurrentLabState): current_zlevels([CurrentZ1,CurrentZ2]) 
& task_requirements([TargetZ1,TargetZ2]) & CurrentZ1 == TargetZ1 & CurrentZ2 == TargetZ2 <-
  .print("Target achieved").

@take_action_loop_plan
+!take_action(CurrentLabState): true <-
  .print("current Full State: ", CurrentLabState);
  getActionFromState([2,1], CurrentLabState, ActionTag, PayloadTags, Payload);
  invokeAction(ActionTag, PayloadTags, Payload);
  !get_new_state.

@get_new_state_plan
+!get_new_state: true <-
  .abolish(current_zlevels(_));
  getCurrentZLevels(NewZLevels)[artifact_id(QLArtId)] ;
  +current_zlevels(NewZLevels);
  .print("New zlevels: ", NewZLevels);
  getCurrentLabState(NewLabState);
  .wait(1000);
  !take_action(NewLabState).