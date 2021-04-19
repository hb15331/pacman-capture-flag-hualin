# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
from util import nearestPoint
from util import manhattanDistance
from util import nearestPoint
from game import Directions
from game import Actions
from game import Grid

import distanceCalculator
import random, time, util
import game


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

## ---------------------------
## |   defense               |
## ---------------------------

class DefensiveBaseAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print ('eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    # pick randomly from the best actions
    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class DefensiveAgent(DefensiveBaseAgent):

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)

    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      # features['invaderDistance'] = min(dists)

      # if they are more than 5 away we know we have a noisy reading
      if min(dists) > 5:
        # we can see where the food has disappeared
        beforeState = self.getPreviousObservation()
        nowState = self.getCurrentObservation()
        missingFood = set(self.getFood(nowState).asList()) \
                    - set(self.getFood(beforeState).asList())

        # get distance to the missing food
        for food in list(missingFood):
            dists.append(self.getMazeDistance(myPos, food))

        # essentially, move towards the missing food
        features['invaderDistance'] = min(dists)

      else:
          # otherwise if they are within 5, the reading is already accurate
          features['invaderDistance'] = min(dists)


    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
      return {'numInvaders': -5000, 'onDefense': 100, 'invaderDistance': -2000, 'stop': -500, 'reverse': -100}



## ---------------------------
## |   offense               |
## ---------------------------


class OffensiveBaseAgent(CaptureAgent):
  """
  handles the initial setup of the agent to populate useful fields
  """
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

    self.initialPos = gameState.getInitialAgentPosition(self.index)

    # initialize an agent's beliefs over opponent positions given the observations
    # key: opponent index, value: belief over opponent positions
    self.beliefs = util.Counter()
    self.opponents = self.getOpponents(gameState)

    for opponent in self.opponents:
      # each opponent's belief is also a dict that maps possible positions to belief values
      self.beliefs[opponent] = util.Counter()
      # all agents must be at their initial positions when game starts
      self.beliefs[opponent][gameState.getInitialAgentPosition(opponent)] = 1.0

    # use asList method to retrieve all possible positions for agent
    self.allPositions = []
    # exclude the wall positions
    for pos in gameState.getWalls().asList(False):
      if pos[1] > 1:
        self.allPositions.append(pos)

    return


  def observeUpdate(self, noisyDistances, gameState, opponent):
    """
    Update agent beliefs based on distance observation and pacman's position
    """
    newBelief = util.Counter()
    currPos = gameState.getAgentPosition(self.index)
    midWidth = gameState.data.layout.width/2


    noisyDistance = noisyDistances[opponent]

    for opponentPos in self.allPositions:
      trueDistance = util.manhattanDistance(currPos, opponentPos)
      # get sensor model P(e_t | x_t)
      observationProb = gameState.getDistanceProb(trueDistance, noisyDistance)

      if self.red and opponentPos[0] < midWidth:
        # from red agent's view, the blue agent is a pacman only if it is on the board's left half
        pacmanCheck = True
      elif (not self.red) and opponentPos[0] > midWidth:
        pacmanCheck = True
      else:
        pacmanCheck = False

      opponentType = gameState.getAgentState(opponent).isPacman
      # the opponent type does not match the possible positions it is allowed to be
      if opponentType != pacmanCheck:
        newBelief[opponentPos] = 0
      else:
        newBelief[opponentPos] = observationProb * self.beliefs[opponent][opponentPos]

      newBelief.normalize()
      self.beliefs[opponent] = newBelief

    return


  def getObservationProb(self, noisyDistance, trueDistance, currPos, opponentPos):
    """
    Return the sensor model P(noisyDistance | trueDistance)
    """
    return



  def elapseTime(self, gameState, opponent):
    """
    Predict beliefs in response to a time step passing from the current state
    """
    newBelief = util.Counter()

    for oldPos in self.allPositions:
      # determine transition model P(newPos | oldPos)
      newPosDistr = self.getPositionDistribution(gameState, oldPos)

      for newPos in newPosDistr.keys():
        # get belief value over old position for a particular opponent
        oldPosProb = self.beliefs[opponent][oldPos]
        newBelief[newPos] += newPosDistr[newPos] * oldPosProb

      newBelief.normalize()
      self.beliefs[opponent] = newBelief

    return



  def getPositionDistribution(self, gameState, oldPos, index = None, agent = None):
    """
    Compute oldPos x -> newPos belief (y, P(y))
    """

    newPositions = []
    newPositions.append((oldPos[0] + 1, oldPos[1]))
    newPositions.append((oldPos[0] - 1, oldPos[1]))
    newPositions.append((oldPos[0], oldPos[1] + 1))
    newPositions.append((oldPos[0], oldPos[1] - 1))

    newPosDistr = util.Counter()

    for newPos in newPositions:
      if newPos in self.allPositions:
        # agent is equally likely to move to any next position
        newPosDistr[newPos] = 1.0

    newPosDistr.normalize()
    return newPosDistr


  def updatePosState(self, opponent):
    mostProbPos = self.beliefs[opponent].argMax()
    return game.Configuration(mostProbPos, Directions.STOP)


  def updateBelief(self, gameState):
    """
    Update the belief over each opponent position given noisy reading
    and current position of calling agent, using the time elapse and
    observation technique
    """
    noisyDistances = gameState.getAgentDistances()

    for opponent in self.opponents:
      opponentPos = gameState.getAgentPosition(opponent)
      # if the opponent is unobservable, infer the current state
      if not opponentPos:
        self.elapseTime(gameState, opponent)
        self.observeUpdate(noisyDistances, gameState, opponent)
      else:
        newBelief = util.Counter()
        newBelief[opponentPos] = 1.0
        self.beliefs[opponent] = newBelief

    return



  def chooseAction(self, gameState):
    """
    Return the expectiMax action using self.depth and self.evaluation function
    """
    start = time.time()
    self.updateBelief(gameState)
    stateCopy = gameState.deepCopy()

    for opponent in self.opponents:
      posConfig = self.updatePosState(opponent)
      stateCopy.data.agentStates[opponent] = game.AgentState(posConfig, gameState.getAgentState(opponent).isPacman)

    optAction = self.maxNode(stateCopy, 1)[1]
    return optAction


  def maxNode(self, gameState, depth):
    if gameState.isOver() or depth == 0:
      return self.evaluationFunction(gameState), Directions.STOP

    maxValue = -999999
    values = []
    actions = gameState.getLegalActions(self.index)
    actions.remove(Directions.STOP)

    for action in actions:
      successor = gameState.generateSuccessor(self.index, action)
      expectiValue = self.expectiNode(self.opponents[0], successor, depth)[0]
      values.append(expectiValue)

      if expectiValue > maxValue:
        maxValue = expectiValue

    # could exist multiple actions that yield same best result
    optActions = [i for i in range(len(values)) if values[i] == maxValue]
    optAction = actions[random.choice(optActions)]

    # print(optAction)
    return maxValue, optAction



  def expectiNode(self, opponent, gameState, depth):
    if gameState.isOver() or depth == 0:
      expectedUtil = self.evaluationFuntion(gameState)
      return expectedUtil, 0

    totalValue = 0
    actions = gameState.getLegalActions(opponent)
    # actions.remove(Directions.STOP)

    for action in actions:
      successor = gameState.generateSuccessor(opponent, action)
      # continue expanding the next opponent's state
      if opponent == self.opponents[0]:
        nextOpponent = opponent + 2
        expectiValue = self.expectiNode(nextOpponent, successor, depth)[0]
        totalValue += expectiValue
      else:
        totalValue += self.maxNode(gameState, depth - 1)[0]

    return totalValue / len(actions), 0



class OffensiveAgent(OffensiveBaseAgent):

  def registerInitialState(self, gameState):
    OffensiveBaseAgent.registerInitialState(self, gameState)
    self.attack = True



  def chooseAction(self, gameState):
    # agents consider following criteria to decide when to return home and garner the points collected
    # 1. proximity to opponent
    # 2. amount of food it is holding

    ghostDistances = []
    currPos = gameState.getAgentPosition(self.index)

    for opponent in self.opponents:
      if not gameState.getAgentState(opponent).isPacman:
        ghostDist = self.distancer.getDistance(currPos, self.beliefs[opponent].argMax())
        ghostDistances.append(ghostDist)

    minGhostDist = min(ghostDistances)

    agentState = gameState.getAgentState(self.index)

    scaredTimers = []
    for opponent in self.opponents:
      scaredTimers.append(gameState.getAgentState(opponent).scaredTimer)
    minScaredTime = min(scaredTimers)

    #print("minScaredTime: " + str(minScaredTime))

    if minGhostDist < 6 and minScaredTime < 5:
      # if ghost is too close and not scared, always return no matter how many foods have been eaten
      self.attack = False
    else:
      # if ghost is far away and agent has eaten enough foods, returns and secures the score
      if agentState.numCarrying > 4:
        self.attack = False
      else:
        self.attack = True

    if len(self.getFood(gameState).asList()) < 3:
      self.attack = False

    #print("carrying: " + str(agentState.numCarrying))
    #print("minGhostDist: " + str(minGhostDist))
    #print(self.attack)

    return OffensiveBaseAgent.chooseAction(self, gameState)



  def evaluationFunction(self, gameState):
    """
    Use heuristic evaluation function to estimate utilities for non-terminal states
    The game state is evaluated differently based on the operation mode of agents
    """
    currScore = self.getScore(gameState)
    currPos = gameState.getAgentPosition(self.index)

    # compute the distance between agent and closest food
    distancesToFoods = []
    foodList = self.getFood(gameState).asList()
    for foodPos in foodList:
      foodDistance = self.distancer.getDistance(currPos, foodPos)
      distancesToFoods.append(foodDistance)

    distanceToClosestFood = 0
    if len(distancesToFoods) != 0:
      distanceToClosestFood = min(distancesToFoods)

    # compute the distance between agent and closest ghost
    distancesToGhosts = []
    for opponent in self.opponents:
      if not gameState.getAgentState(opponent).isPacman:
        opponentPos = gameState.getAgentPosition(opponent)

        if opponentPos != None:
          currDist = self.distancer.getDistance(currPos, opponentPos)
          distancesToGhosts.append(currDist)

    distanceToClosestGhost = 0
    if len(distancesToGhosts) != 0:
      distanceToClosestGhost = min(distancesToGhosts)

    # the offensive agent is actively seeking for foods
    if self.attack:
      foodList = self.getFood(gameState).asList()
      numOfFoods = len(foodList)

      # agent can keep offensive when close to capsules
      distancesToCapsule = []
      if self.red:
        capsules = gameState.getBlueCapsules()
      else:
        capsules = gameState.getRedCapsules()

      for capsule in capsules:
        capsuleDistance = self.distancer.getDistance(currPos, capsule)
        distancesToCapsule.append(capsuleDistance)

      minCapsuleDist = 0
      if len(distancesToCapsule) != 0:
        minCapsuleDist = min(distancesToCapsule)

      if distanceToClosestGhost > 5:
        distanceToClosestGhost = 0

      return currScore - 2 * distanceToClosestFood - 400 * numOfFoods - 10 * minCapsuleDist + 100 * distanceToClosestGhost

    # offensive agents decide to return home and secure the score
    else:
      # home distance = mazeDistance(currPos, any point on board's central axis)
      boardHeight = gameState.data.layout.height
      midWidth = gameState.data.layout.width/2

      homeDistances = []
      for y in range(boardHeight):
        # for all positions reachable except wall
        if (midWidth, y) in self.allPositions:
          homeDistance = self.distancer.getDistance(currPos, (midWidth, y))
          homeDistances.append(homeDistance)

      minHomeDistance = min(homeDistances)

      return 500 * distanceToClosestGhost - 5 * minHomeDistance
