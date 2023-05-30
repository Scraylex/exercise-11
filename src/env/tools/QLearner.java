package tools;

import cartago.Artifact;
import cartago.OPERATION;
import cartago.OpFeedbackParam;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.Logger;

public class QLearner extends Artifact {

    private Lab lab; // the lab environment that will be learnt
    private int stateCount; // the number of possible states in the lab environment
    private int actionCount; // the number of possible actions in the lab environment
    //    private HashMap<Integer, double[][]> qTables; // a map for storing the qTables computed for different goals
    private HashMap<String, double[][]> qTables; // a map for storing the qTables computed for different goals
    private final Random random = new Random();
    private static final int ITERATIONS = 10;
    private static final String FILENAME = "qtables.json";
    private static final Logger LOGGER = Logger.getLogger(QLearner.class.getName());

    @SuppressWarnings("unused")
    public void init(String environmentURL) {
        // the URL of the W3C Thing Description of the lab Thing
        this.lab = new Lab(environmentURL);

        this.stateCount = lab.getStateCount();
        LOGGER.info("Initialized with a state space of n=" + stateCount);

        this.actionCount = lab.getActionCount();
        LOGGER.info("Initialized with an action space of m=" + actionCount);

        shuffle();
    }

    private void shuffle() {
        int currentState = lab.readCurrentState();

        for (int i = 0; i < ITERATIONS; i++) {
            List<Integer> possibleActions = lab.getApplicableActions(currentState);
            int randomAction = possibleActions.get(random.nextInt(possibleActions.size()));
            lab.performAction(randomAction);
        }

        currentState = lab.readCurrentState();
        LOGGER.info("Shuffled to state: " + currentState);
    }

    /**
     * Computes a Q matrix for the state space and action space of the lab, and against
     * a goal description. For example, the goal description can be of the form [z1level, z2Level],
     * where z1Level is the desired value of the light level in Zone 1 of the lab,
     * and z2Level is the desired value of the light level in Zone 2 of the lab.
     * For exercise 11, the possible goal descriptions are:
     * [0,0], [0,1], [0,2], [0,3],
     * [1,0], [1,1], [1,2], [1,3],
     * [2,0], [2,1], [2,2], [2,3],
     * [3,0], [3,1], [3,2], [3,3].
     *
     * <p>
     * HINT: Use the methods of {@link LearningEnvironment} (implemented in {@link Lab})
     * to interact with the learning environment (here, the lab), e.g., to retrieve the
     * applicable actions, perform an action at the lab during learning etc.
     * </p>
     *
     * @param goalDescription the desired goal against the which the Q matrix is calculated (e.g., [2,3])
     * @param episodesObj     the number of episodes used for calculating the Q matrix
     * @param alphaObj        the learning rate with range [0,1].
     * @param gammaObj        the discount factor [0,1]
     * @param epsilonObj      the exploration probability [0,1]
     * @param rewardObj       the reward assigned when reaching the goal state
     **/
    @OPERATION
    @SuppressWarnings("unused")
    public void calculateQ(Object[] goalDescription,
                           Object episodesObj,
                           Object alphaObj,
                           Object gammaObj,
                           Object epsilonObj,
                           Object rewardObj) {

        // ensure that the right datatypes are used
        int episodes = Integer.parseInt(episodesObj.toString());
        double alpha = Double.parseDouble(alphaObj.toString());
        double gamma = Double.parseDouble(gammaObj.toString());
        double epsilon = Double.parseDouble(epsilonObj.toString());
        int reward = Integer.parseInt(rewardObj.toString());

        HyperParams params = HyperParams.create(alpha, gamma, epsilon, reward, episodes);

        Integer z1 = Integer.valueOf(goalDescription[0].toString());
        Integer z2 = Integer.valueOf(goalDescription[1].toString());

        qTables = readOrInitializeQTablesFromFile();

        String newKey = String.format("[%d,%d]", z1, z2);

        if (qTables.containsKey(newKey)) {
            LOGGER.info("Already know: " + newKey);
        } else {
            qTables.put(newKey, initializeQTable());
            double[][] singleQTable = qTables.get(newKey);
            int currentState = lab.readCurrentState();
            for (int i = 0; i < params.getEpisodes(); i++) {
                LOGGER.info("It's the next episode - Dr. Dre");
                // intialize S
                initS(currentState);
                currentState = performActions(params, z1, z2, singleQTable, currentState);
                LOGGER.info("State after actions: " + currentState);
            }
            LOGGER.info("Tune in next time for the chronic");
        }
        writeQTablesToFile(qTables);
    }

    private int performActions(HyperParams params,
                               Integer z1,
                               Integer z2,
                               double[][] singleQTable,
                               int currentState) {
        while (true) {
            List<Integer> possibleActions = lab.getApplicableActions(currentState);
            double randomNumber = random.nextDouble();
            int chosenAction = possibleActions.get(random.nextInt(possibleActions.size()));
            if (randomNumber > params.getEpsilon()) {
                chosenAction = getMaxValueIndex(singleQTable[currentState], lab.getApplicableActions(currentState));
            }
            lab.performAction(chosenAction);
            sleep(50);
            int newState = lab.readCurrentState();
            double maxqsda = getMaxQSA(newState, singleQTable);
            double currentQsa = singleQTable[currentState][chosenAction];
            int calculatedReward = checkReward(params.getReward(), z1, z2);
            double newValue = currentQsa + params.getAlpha() * ((calculatedReward + params.getGamma() * maxqsda) - currentQsa);
            singleQTable[currentState][chosenAction] = newValue;
            currentState = newState;
            if (calculatedReward == params.getReward()) {
                break;
            }
        }
        return currentState;
    }

    private void initS(int currentState) {
        for (int j = 0; j < 1000; j++) {
            List<Integer> possibleActions = lab.getApplicableActions(currentState);
            int randomAction = possibleActions.get(random.nextInt(possibleActions.size()));
            lab.performAction(randomAction);
            sleep(3);
        }
    }

    private static void sleep(int millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    @OPERATION
    @SuppressWarnings("unused")
    public void getCurrentZLevels(OpFeedbackParam<Integer[]> currentStateTag) {
        Integer[] zlevels = {
                lab.getCurrentState().get(0),
                lab.getCurrentState().get(1)
        };
        currentStateTag.set(zlevels);
    }

    @OPERATION
    @SuppressWarnings("unused")
    public void getCurrentLabState(OpFeedbackParam<Object[]> currentStateTag) {
        Object[] t = {
                lab.getCurrentState().get(0),
                lab.getCurrentState().get(1),
                lab.getCurrentState().get(2).equals(1),
                lab.getCurrentState().get(3).equals(1),
                lab.getCurrentState().get(4).equals(1),
                lab.getCurrentState().get(5).equals(1),
                lab.getCurrentState().get(6)
        };
        currentStateTag.set(t);
    }

    private static void writeQTablesToFile(HashMap<String, double[][]> qTables) {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(qTables);
        try (FileWriter writer = new FileWriter(FILENAME)) {
            writer.write(json);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static HashMap<String, double[][]> readOrInitializeQTablesFromFile() {
        try {
            String json = new String(Files.readAllBytes(Paths.get(FILENAME)));
            Gson gson = new Gson();
            return gson.fromJson(json, new TypeToken<HashMap<String, double[][]>>() {
            }.getType());
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new HashMap<>();
    }

    private int checkReward(int reward, Integer z1, Integer z2) {
        if (Objects.equals(z1, lab.getCurrentState().get(0))
                && Objects.equals(z2, lab.getCurrentState().get(1))) {
            LOGGER.info("Got the chronic");
            return reward;
        } else {
            return 0;
        }
    }

    private double getMaxQSA(int currentState, double[][] qTable) {
        List<Integer> possibleActions = lab.getApplicableActions(currentState);
        double max = 0.0;
        for (int item : possibleActions) {
            double possibleMax = qTable[currentState][item];
            if (possibleMax > max) {
                max = possibleMax;
            }
        }
        return max;
    }

    /**
     * Returns information about the next best action based on a provided state and the QTable for
     * a goal description. The returned information can be used by agents to invoke an action
     * using a ThingArtifact.
     *
     * @param goalDescription           the desired goal against the which the Q matrix is calculated (e.g., [2,3])
     * @param currentStateDescription   the current state e.g. [2,2,true,false,true,true,2]
     * @param nextBestActionTag         the (returned) semantic annotation of the next best action, e.g. "http://example.org/was#SetZ1Light"
     * @param nextBestActionPayloadTags the (returned) semantic annotations of the payload of the next best action, e.g. [Z1Light]
     * @param nextBestActionPayload     the (returned) payload of the next best action, e.g. true
     **/
    @OPERATION
    @SuppressWarnings("unused")
    public void getActionFromState(Object[] goalDescription,
                                   Object[] currentStateDescription,
                                   OpFeedbackParam<String> nextBestActionTag,
                                   OpFeedbackParam<Object[]> nextBestActionPayloadTags,
                                   OpFeedbackParam<Object[]> nextBestActionPayload) {
        int z1 = Integer.parseInt(goalDescription[0].toString());
        int z2 = Integer.parseInt(goalDescription[1].toString());

        String targetState = String.format("%d,%d", z1, z2);

        double[][] singleQTable = qTables.get(targetState);
        int currentIndex = lab.readCurrentState();
        System.out.println("current Index: " + currentIndex);
        double randomNumber = random.nextDouble();
        double epsilon = 0.9;
        double[] possibleActions = singleQTable[currentIndex];
        List<Integer> applicableActions = lab.getApplicableActions(currentIndex);

        int nextAction = applicableActions.get(random.nextInt(applicableActions.size()));
        if (randomNumber > epsilon) {
            nextAction = getMaxValueIndex(possibleActions, applicableActions);
        }
        ActionHandler.handleAction(nextAction, nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
    }

    private int getMaxValueIndex(double[] possibleActions, List<Integer> applicableActions) {
        int maxIndex = 0;

        double maxValue = possibleActions[applicableActions.get(0)];

        for (Integer applicableAction : applicableActions) {
            if (possibleActions[applicableAction] > maxValue) {
                maxValue = possibleActions[applicableAction];
                maxIndex = applicableAction;
            }
        }
        if (maxValue == 0.0) {
            int maxIndexHelper = random.nextInt(applicableActions.size());
            maxIndex = applicableActions.get(maxIndexHelper);
        }
        return maxIndex;
    }

    /**
     * Initialize a Q matrix
     *
     * @return the Q matrix
     */
    private double[][] initializeQTable() {
        double[][] qTable = new double[stateCount][actionCount];
        for (int i = 0; i < stateCount; i++) {
            Arrays.fill(qTable[i], 0.0);
        }
        return qTable;
    }

    private static class HyperParams {
        private final double alpha;
        private final double gamma;
        private final double epsilon;
        private final int reward;
        private final int episodes;

        private HyperParams(double alpha, double gamma, double epsilon, int reward, int episodes) {
            this.alpha = alpha;
            this.gamma = gamma;
            this.epsilon = epsilon;
            this.reward = reward;
            this.episodes = episodes;
        }

        public static HyperParams create(double alpha, double gamma, double epsilon, int reward, int episodes) {
            return new HyperParams(alpha, gamma, epsilon, reward, episodes);
        }

        public double getAlpha() {
            return alpha;
        }

        public double getGamma() {
            return gamma;
        }

        public double getEpsilon() {
            return epsilon;
        }

        public int getReward() {
            return reward;
        }

        public int getEpisodes() {
            return episodes;
        }
    }

    private static class ActionHandler {

        static void handleAction(int nextAction, OpFeedbackParam<String> nextBestActionTag,
                                 OpFeedbackParam<Object[]> nextBestActionPayloadTags,
                                 OpFeedbackParam<Object[]> nextBestActionPayload) {
            switch (nextAction) {
                case 0:
                    setZ1LightFalse(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
                    break;
                case 1:
                    setZ1LightTrue(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
                    break;
                case 2:
                    setZ2LightFalse(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
                    break;
                case 3:
                    setZ2LightTrue(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
                    break;
                case 4:
                    setZ1BlindsFalse(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
                    break;
                case 5:
                    setZ1BlindsTrue(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
                    break;
                case 6:
                    setZ2BlindsFalse(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
                    break;
                case 7:
                    setZ2BlindsTrue(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
                    break;
                default:
                    // Uh ohhhhhhh
                    System.err.println("U done goofed: " + nextAction);
                    break;
            }
        }
        private static void setZ1LightTrue(OpFeedbackParam<String> nextBestActionTag,
                                           OpFeedbackParam<Object[]> nextBestActionPayloadTags,
                                           OpFeedbackParam<Object[]> nextBestActionPayload) {
            // sets the semantic annotation of the next best action to be returned
            nextBestActionTag.set("http://example.org/was#SetZ1Light");

            // sets the semantic annotation of the payload of the next best action to be
            // returned
            Object[] payloadTags = {"Z1Light"};
            nextBestActionPayloadTags.set(payloadTags);

            // sets the payload of the next best action to be returned
            Object[] payload = {true};
            nextBestActionPayload.set(payload);
        }

        private static void setZ1LightFalse(OpFeedbackParam<String> nextBestActionTag,
                                            OpFeedbackParam<Object[]> nextBestActionPayloadTags,
                                            OpFeedbackParam<Object[]> nextBestActionPayload) {
            // sets the semantic annotation of the next best action to be returned
            nextBestActionTag.set("http://example.org/was#SetZ1Light");

            // sets the semantic annotation of the payload of the next best action to be
            // returned
            Object[] payloadTags = {"Z1Light"};
            nextBestActionPayloadTags.set(payloadTags);

            // sets the payload of the next best action to be returned
            Object[] payload = {false};
            nextBestActionPayload.set(payload);
        }

        private static void setZ2LightFalse(OpFeedbackParam<String> nextBestActionTag,
                                            OpFeedbackParam<Object[]> nextBestActionPayloadTags,
                                            OpFeedbackParam<Object[]> nextBestActionPayload) {
            // sets the semantic annotation of the next best action to be returned
            nextBestActionTag.set("http://example.org/was#SetZ2Light");

            // sets the semantic annotation of the payload of the next best action to be
            // returned
            Object[] payloadTags = {"Z2Light"};
            nextBestActionPayloadTags.set(payloadTags);

            // sets the payload of the next best action to be returned
            Object[] payload = {false};
            nextBestActionPayload.set(payload);
        }

        private static void setZ2LightTrue(OpFeedbackParam<String> nextBestActionTag,
                                           OpFeedbackParam<Object[]> nextBestActionPayloadTags,
                                           OpFeedbackParam<Object[]> nextBestActionPayload) {
            // sets the semantic annotation of the next best action to be returned
            nextBestActionTag.set("http://example.org/was#SetZ2Light");

            // sets the semantic annotation of the payload of the next best action to be
            // returned
            Object[] payloadTags = {"Z2Light"};
            nextBestActionPayloadTags.set(payloadTags);

            // sets the payload of the next best action to be returned
            Object[] payload = {true};
            nextBestActionPayload.set(payload);
        }

        private static void setZ1BlindsTrue(OpFeedbackParam<String> nextBestActionTag,
                                            OpFeedbackParam<Object[]> nextBestActionPayloadTags,
                                            OpFeedbackParam<Object[]> nextBestActionPayload) {
            // sets the semantic annotation of the next best action to be returned
            nextBestActionTag.set("http://example.org/was#SetZ1Blinds");

            // sets the semantic annotation of the payload of the next best action to be
            // returned
            Object[] payloadTags = {"Z1Blinds"};
            nextBestActionPayloadTags.set(payloadTags);

            // sets the payload of the next best action to be returned
            Object[] payload = {true};
            nextBestActionPayload.set(payload);
        }

        private static void setZ1BlindsFalse(OpFeedbackParam<String> nextBestActionTag,
                                             OpFeedbackParam<Object[]> nextBestActionPayloadTags,
                                             OpFeedbackParam<Object[]> nextBestActionPayload) {
            // sets the semantic annotation of the next best action to be returned
            nextBestActionTag.set("http://example.org/was#SetZ1Blinds");

            // sets the semantic annotation of the payload of the next best action to be
            // returned
            Object[] payloadTags = {"Z1Blinds"};
            nextBestActionPayloadTags.set(payloadTags);

            // sets the payload of the next best action to be returned
            Object[] payload = {false};
            nextBestActionPayload.set(payload);
        }

        private static void setZ2BlindsFalse(OpFeedbackParam<String> nextBestActionTag,
                                             OpFeedbackParam<Object[]> nextBestActionPayloadTags,
                                             OpFeedbackParam<Object[]> nextBestActionPayload) {
            // sets the semantic annotation of the next best action to be returned
            nextBestActionTag.set("http://example.org/was#SetZ2Blinds");

            // sets the semantic annotation of the payload of the next best action to be
            // returned
            Object[] payloadTags = {"Z2Blinds"};
            nextBestActionPayloadTags.set(payloadTags);

            // sets the payload of the next best action to be returned
            Object[] payload = {false};
            nextBestActionPayload.set(payload);
        }

        private static void setZ2BlindsTrue(OpFeedbackParam<String> nextBestActionTag,
                                            OpFeedbackParam<Object[]> nextBestActionPayloadTags,
                                            OpFeedbackParam<Object[]> nextBestActionPayload) {
            // sets the semantic annotation of the next best action to be returned
            nextBestActionTag.set("http://example.org/was#SetZ2Blinds");

            // sets the semantic annotation of the payload of the next best action to be
            // returned
            Object[] payloadTags = {"Z2Blinds"};
            nextBestActionPayloadTags.set(payloadTags);

            // sets the payload of the next best action to be returned
            Object[] payload = {true};
            nextBestActionPayload.set(payload);
        }
    }
}
