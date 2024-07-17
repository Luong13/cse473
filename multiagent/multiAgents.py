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
import random, util

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

        "*** YOUR CODE HERE ***"
        minFoodDistance = float("inf")
        for foodPos in newFood.asList():
            foodDistance = util.manhattanDistance(newPos, foodPos)
            minFoodDistance = min(minFoodDistance, foodDistance)

        ghostDistanceSum = 1
        inDanger = 0
        for ghostPos in successorGameState.getGhostPositions():
            ghostDistance = util.manhattanDistance(newPos, ghostPos)
            ghostDistanceSum += ghostDistance
            if ghostDistance == 1:
                inDanger = 1

        return successorGameState.getScore() + 1/minFoodDistance - 1/ghostDistanceSum - inDanger

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
        "*** YOUR CODE HERE ***"
        def minimax(state, agent, depth):
            if (state.isWin() or state.isLose() or depth == self.depth):
                return self.evaluationFunction(state)
            nextAgent = agent + 1
            if agent == self.index:
                maximum = float("-inf")
                for legalAction in state.getLegalActions(agent):
                    maximum = max(maximum, minimax(state.generateSuccessor(agent, legalAction), nextAgent, depth))
                return maximum
            else:
                if nextAgent == state.getNumAgents():
                    nextAgent = self.index
                    depth += 1
                minimum = float("inf")
                for legalAction in state.getLegalActions(agent):
                    minimum = min(minimum, minimax(state.generateSuccessor(agent, legalAction), nextAgent, depth))
                return minimum

        maxUtility = float("-inf")
        action = None
        for legalAction in gameState.getLegalActions(self.index):
            utility = minimax(gameState.generateSuccessor(self.index, legalAction), self.index + 1, 0)
            if utility > maxUtility:
                maxUtility = utility
                action = legalAction
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBetaPrune(state, agent, depth, a, b):
            if (state.isWin() or state.isLose() or depth == self.depth):
                return self.evaluationFunction(state)
            if agent == self.index:
                return maxValue(state, agent, depth, a, b)
            else:
                return minValue(state, agent, depth, a, b)

        def maxValue(state, agent, depth, a, b):
            value = float("-inf")

            nextAgent = agent + 1
            if nextAgent == state.getNumAgents():
                nextAgent = self.index
                depth += 1

            for legalAction in state.getLegalActions(agent):
                value = max(value, alphaBetaPrune(state.generateSuccessor(agent, legalAction), nextAgent, depth, a, b))
                if value > b:
                    return value
                a = max(a, value)
            return value

        def minValue(state, agent, depth, a, b):
            value = float("inf")

            nextAgent = agent + 1
            if nextAgent == state.getNumAgents():
                nextAgent = self.index
                depth += 1

            for legalAction in state.getLegalActions(agent):
                value = min(value, alphaBetaPrune(state.generateSuccessor(agent, legalAction), nextAgent, depth, a, b))
                if value < a:
                    return value
                b = min(b, value)
            return value

        utility = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        action = None
        for legalAction in gameState.getLegalActions(self.index):
            value = alphaBetaPrune(gameState.generateSuccessor(self.index, legalAction), self.index + 1, 0, alpha, beta)
            if value > utility:
                utility = value
                action = legalAction
            if utility > beta:
                return utility
            alpha = max(alpha, utility)

        return action

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
        "*** YOUR CODE HERE ***"
        def expectimax(state, agent, depth):
            if (state.isWin() or state.isLose() or depth == self.depth):
                return self.evaluationFunction(state)
            nextAgent = agent + 1
            if agent == self.index:
                maximum = float("-inf")
                for legalAction in state.getLegalActions():
                    maximum = max(maximum, expectimax(state.generateSuccessor(agent, legalAction), nextAgent, depth))
                return maximum
            else:
                if nextAgent == state.getNumAgents():
                    nextAgent = self.index
                    depth += 1
                total = 0
                for legalAction in state.getLegalActions(agent):
                    total += expectimax(state.generateSuccessor(agent, legalAction), nextAgent, depth)
                return total / len(state.getLegalActions(agent))

        maxUtility = float("-inf")
        action = None
        for legalAction in gameState.getLegalActions(self.index):
            utility = expectimax(gameState.generateSuccessor(self.index, legalAction), self.index + 1, 0)
            if utility > maxUtility:
                maxUtility = utility
                action = legalAction

        return action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: My better evaluation is similar to the evaluation in my reflex agent.
    It also takes into account the distance to the closest food, the sum of distances
    to all ghosts, and whether or not there is danger (a ghost right next to pacman).
    In addition, it also takes into account the sum of distances to all capsules and
    the number of capsules remaining. To get a higher evaluation score, we want lesser
    values for distance to closest food, danger, sum of distances to all capsules, and
    number of capsules remaining. And we want a greater value of sum of distances to all
    ghosts
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    minFoodDistance = float("inf")
    for food in currentGameState.getFood().asList():
        minFoodDistance = min(minFoodDistance, util.manhattanDistance(pos, food))

    ghostDistanceSum = 1
    inDanger = 0

    for ghostPos in currentGameState.getGhostPositions():
        ghostDistance = util.manhattanDistance(pos, ghostPos)
        ghostDistanceSum += ghostDistance
        if ghostDistance == 1:
            inDanger = 1

    capsuleDistanceSum = 1
    for capsule in currentGameState.getCapsules():
        capsuleDistanceSum += util.manhattanDistance(pos, capsule)

    numCapsules = len(currentGameState.getCapsules())
    
    return currentGameState.getScore() + 1/minFoodDistance - 1/ghostDistanceSum - inDanger + 1/capsuleDistanceSum - numCapsules

# Abbreviation
better = betterEvaluationFunction
