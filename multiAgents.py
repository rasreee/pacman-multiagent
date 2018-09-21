# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        def distance(x, y):
            return abs(x[0] - y[0]) + abs(x[1] - y[1])


        min_distance = float('inf')
        for ghost in newGhostStates:
            distance_to_ghost = distance(newPos,ghost.getPosition())
            if distance_to_ghost < min_distance:
                min_distance = distance_to_ghost
        min_distance_tofood = float('inf')

        for i in range(1,newFood.width):
            for j in range(1,newFood.height):
                if newFood[i][j]:
                    if distance(newPos,(i,j)) < min_distance_tofood:
                        min_distance_tofood = distance(newPos,(i,j))

        if min_distance <= 2 and (currentGameState.getNumFood() > successorGameState.getNumFood()):
            return 30+10*min_distance - 5*min_distance_tofood
        if min_distance <= 2:
            return 10*min_distance - 5*min_distance_tofood
        if successorGameState.getNumFood() == 0 and currentGameState.getNumFood() == 1:
            return 1000 + 5*min_distance
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            # print(min_distance_tofood)
            return 1000000-10*min_distance_tofood + 5*min_distance
        if action == 'Stop':
            return 100-100*min_distance_tofood + 5*min_distance
        else:
            return 100-10*min_distance_tofood + 5*min_distance

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        num_ghost = gameState.getNumAgents() - 1
        def maximizer(gameStates,depth):
            if gameStates.isLose() or gameStates.isWin():
                return self.evaluationFunction(gameStates)
            actions = gameStates.getLegalActions(0)
            max_score = float("-inf")
            max_action = "Stop"
            for i in actions:
                self.num_ghost = gameStates.getNumAgents()-1
                score = minimizer(gameStates.generateSuccessor(0,i),depth,1)
                if score*1.0 > max_score:
                    max_score = score
                    max_action = i

            if depth == 0:
                return max_action
            else:
                return max_score

        def minimizer(gameStates, depth, num_ghosts):
            if gameStates.isLose() or gameStates.isWin():
                return self.evaluationFunction(gameStates)
            actions = gameStates.getLegalActions(num_ghosts)
            best_score = float('inf')

            for i in actions:
                if num_ghosts == gameState.getNumAgents() - 1:
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(gameStates.generateSuccessor(num_ghosts, i))
                    else:
                        score = min(best_score, maximizer(gameStates.generateSuccessor(num_ghosts, i), depth + 1))
                else:
                    self.num_ghost = num_ghosts - 1
                    score = min(best_score,
                                minimizer(gameStates.generateSuccessor(num_ghosts, i), depth, num_ghosts + 1))
                if best_score > score * 1.0:
                    best_score = score
            return best_score

        return maximizer(gameState, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()

        def reset(agent, depth):
            agent = 0
            depth += 1

        def alphaBetaPrune(gameState, depth, agent, alpha, beta):
            if agent >= numAgents:
                agent = 0
                depth += 1
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            elif agent == 0:
                return maximizer(gameState, depth, agent, alpha, beta)
            return minimizer(gameState, depth, agent, alpha, beta)

        def maximizer(gameState, depth, agent, alpha, beta):
            currentNode = (None, -math.inf)
            potentialActions = gameState.getLegalActions(agent)

            if potentialActions == []:
                print("MAXIMIZER: No more actions left!!! Calling getScore()")
                return self.evaluationFunction(gameState)

            for action in potentialActions:
                successorState = gameState.generateSuccessor(agent, action)
                successorValue = alphaBetaPrune(successorState, depth, agent + 1, alpha, beta)

                tempValue = successorValue
                if type(successorValue) is tuple:
                    tempValue = successorValue[1]

                if tempValue > currentNode[1]:
                    currentNode = (action, tempValue)
                if tempValue > beta:
                    return (action, tempValue)

                alpha = max(alpha, tempValue)

            return currentNode

        def minimizer(state, depth, agent, alpha, beta):
            currentNode = (None, math.inf)
            potentialActions = state.getLegalActions(agent)

            if potentialActions == []:
                print("MINIMIZER: No more actions left!!! Calling getScore()")
                return self.evaluationFunction(state)

            for action in potentialActions:
                successorState = state.generateSuccessor(agent, action)
                successorValue = alphaBetaPrune(successorState, depth, agent + 1, alpha, beta)

                tempValue = successorValue
                if type(successorValue) is tuple:
                    tempValue = successorValue[1]

                if tempValue < currentNode[1]:
                    currentNode = (action, tempValue)
                if tempValue < alpha:
                    return (action, tempValue)

                beta = min(beta, tempValue)

            return currentNode

        return alphaBetaPrune(gameState, 0, 0, -math.inf, math.inf)[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, depth, agent):
            if agent >= state.getNumAgents():  # Reset back to maximizing node, and iterate depth by 1
                agent = 0
                depth += 1
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            elif agent == 0:
                return maximizer(state, depth, agent)
            return chance(state, depth, agent)

        def maximizer(state, depth, agent):
            currentNode = (None, -math.inf)
            potentialActions = state.getLegalActions(agent)

            if potentialActions == []:
                return self.evaluationFunction(gameState)

            for action in potentialActions:
                successorState = state.generateSuccessor(agent, action)
                successorValue = expectimax(successorState, depth, agent + 1)

                tempValue = successorValue
                if type(successorValue) is tuple:
                    tempValue = successorValue[1]

                if tempValue > currentNode[1]:
                    currentNode = (action, tempValue)

            return currentNode

        def chance(state, depth, agent):
            currentNode = (None, 0)
            potentialActions = state.getLegalActions(agent)

            if potentialActions == []:
                return self.evaluationFunction(state)

            # ASSUMPTION: Chance of each action is equally likely
            chance = 1.0/len(potentialActions)

            for action in potentialActions:

                successorState = state.generateSuccessor(agent, action)
                successorValue = expectimax(successorState, depth, agent + 1)

                tempValue = successorValue
                if type(successorValue) is tuple:
                    tempValue = successorValue[1]

                currentNode = (currentNode[0], tempValue * chance + currentNode[1])
            return currentNode

        return expectimax(gameState, 0, 0)[0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    First I check if the current game state is already a win or a loss.
    Then, I check if there are no food items left. If there isn't anymore food, I just return the current score of the given state.
    I ran the autograder at this point, and it gave me a very
    """
    locations = currentGameState.getFood().asList()
    currentPosition = currentGameState.getPacmanPosition()
    currentValue = currentGameState.getScore()

    minDistance = math.inf
    if locations == []:
        return currentValue
    for foodCoords in locations:
        distanceToFood = util.manhattanDistance(foodCoords, currentPosition)
        minDistance = min(minDistance, distanceToFood)

    return currentValue - minDistance

# Abbreviation
better = betterEvaluationFunction
