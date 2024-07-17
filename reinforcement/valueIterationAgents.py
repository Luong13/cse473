# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        while self.iterations > 0:
            tempValues = self.values.copy()
            states = self.mdp.getStates()
            for state in states:
                possibleActions = self.mdp.getPossibleActions(state)
                actionValues = []
                for action in possibleActions:
                    nextStates = self.mdp.getTransitionStatesAndProbs(state, action)
                    actionValue = 0
                    for nextState, prob in nextStates:
                        actionValue += (prob * (self.mdp.getReward(state, action, nextState) + (self.discount * tempValues[nextState])))
                    actionValues.append(actionValue)
                if len(actionValues) != 0:
                    self.values[state] = max(actionValues)
            self.iterations -= 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        nextStates = self.mdp.getTransitionStatesAndProbs(state, action)
        actionValue = 0
        for nextState, prob in nextStates:
            actionValue += (prob * (self.mdp.getReward(state, action, nextState) + (self.discount * self.values[nextState])))

        return actionValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        possibleActions = self.mdp.getPossibleActions(state)
        finalAction = ""
        maxSum = float("-inf")
        for action in possibleActions:
            actionValue = self.computeQValueFromValues(state, action)
            if actionValue > maxSum:
                finalAction = action
                maxSum = actionValue

        return finalAction
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        iterIndex = 0
        for iter in range(0, self.iterations):
            if iterIndex == len(states):
                iterIndex = 0
            targetState = states[iterIndex]
            iterIndex += 1
            if self.mdp.isTerminal(targetState):
                continue
            bestAction = self.computeActionFromValues(targetState)
            qValue = self.computeQValueFromValues(targetState, bestAction)
            self.values[targetState] = qValue

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        self.queue = util.PriorityQueue()
        self.predecessors = util.Counter()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                self.predecessors[state] = set()

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue

            possibleActions = self.mdp.getPossibleActions(state)
            for action in possibleActions:
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob != 0 and not self.mdp.isTerminal(nextState):
                        self.predecessors[nextState].add(state)

            currentValue = self.values[state]
            bestAction = self.computeActionFromValues(state)
            highestQValue = self.computeQValueFromValues(state, bestAction)
            diff = abs(currentValue - highestQValue)
            self.queue.push(state, -diff)

        for iteration in range(0, self.iterations):
            if self.queue.isEmpty():
                return

            state = self.queue.pop()

            bestAction = self.computeActionFromValues(state)
            self.values[state] = self.computeQValueFromValues(state, bestAction)

            for predecessor in self.predecessors[state]:
                currentValue = self.values[predecessor]
                bestAction = self.computeActionFromValues(predecessor)
                highestQValue = self.computeQValueFromValues(predecessor, bestAction)
                diff = abs(currentValue - highestQValue)
                if diff > self.theta:
                    self.queue.update(predecessor, -diff)
