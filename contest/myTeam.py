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
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'OffensiveAgent'):
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

# class DummyAgent(CaptureAgent):
#   """
#   A Dummy agent to serve as an example of the necessary agent structure.
#   You should look at baselineTeam.py for more details about how to
#   create an agent as this is the bare minimum.
#   """
#
#   def registerInitialState(self, gameState):
#     """
#     This method handles the initial setup of the
#     agent to populate useful fields (such as what team
#     we're on).
#
#     A distanceCalculator instance caches the maze distances
#     between each pair of positions, so your agents can use:
#     self.distancer.getDistance(p1, p2)
#
#     IMPORTANT: This method may run for at most 15 seconds.
#     """
#
#     '''
#     Make sure you do not delete the following line. If you would like to
#     use Manhattan distances instead of maze distances in order to save
#     on initialization time, please take a look at
#     CaptureAgent.registerInitialState in captureAgents.py.
#     '''
#     CaptureAgent.registerInitialState(self, gameState)
#
#     '''
#     Your initialization code goes here, if you need any.
#     '''

  #
  # def chooseAction(self, gameState):
  #   """
  #   Picks among actions randomly.
  #   """
  #   actions = gameState.getLegalActions(self.index)
  #
  #   '''
  #   You should change this in your own agent.
  #   '''
  #
  #   return random.choice(actions)



class DummyAgent(CaptureAgent):
  """
  handles the initial setup of the agent to populate useful fields
  """
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

    # self.initialPos = gameState.getAgentPosition(self.index)
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

    self.midWidth = gameState.data.layout.width/2

    return



  # TODO:
  # def initializeBeliefs(self, opponent):
  #   return



  def observeUpdate(self, noisyDistances, gameState, opponent):
    """
    Update agent beliefs based on distance observation and pacman's position
    """
    newBelief = util.Counter()
    currPos = gameState.getAgentPosition(self.index)

    noisyDistance = noisyDistances[opponent]

    for opponentPos in self.allPositions:
      trueDistance = util.manhattanDistance(currPos, opponentPos)
      # get sensor model P(e_t | x_t)
      observationProb = gameState.getDistanceProb(trueDistance, noisyDistance)

      if self.red and opponentPos[0] < self.midWidth:
        # from red agent's view, the blue agent is a pacman only if it is on the board's left half
        pacmanCheck = True
      elif (not self.red) and opponentPos[0] > self.midWidth:
        pacmanCheck = True
      else:
        pacmanCheck = False

      opponentType = gameState.getAgentState(opponent).isPacman
      # the opponent type does not match the possible positions it is allowed to be
      if opponentType != pacmanCheck:
        newBelief[opponentPos] = 0
      elif trueDistance <= 5:
        newBelief[opponentPos] = 0
      else:
        newBelief[opponentPos] = observationProb * self.beliefs[opponent][opponentPos]

      # TODO:
      newBelief.normalize()
      self.beliefs[opponent] = newBelief

    return



  # def getObservationProb(self, noisyDistance, trueDistance, currPos, opponentPos):
  #   """
  #   Return the sensor model P(noisyDistance | trueDistance)
  #   """
  #   return



  def elapseTime(self, gameState, opponent):
    """
    Predict beliefs in response to a time step passing from the current state
    """
    # use asList method to retrieve all possible positions for agent
    # allPositions = []
    # exclude the wall positions
    # for pos in gameState.getWalls().asList(False):
    #   if pos[1] > 1:
    #     allPositions.append(pos)

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
    self.updateBelief(gameState)
    stateCopy = gameState.deepCopy()

    for opponent in self.opponents:
      mostProbPos = self.beliefs[opponent].argMax()
      config = game.Configuration(mostProbPos, Directions.STOP)
      stateCopy.data.agentStates[opponent] = game.AgentState(config, gameState.getAgentState(opponent).isPacman)

    optAction = self.maxNode(stateCopy, 2)[1]
    return optAction



  def maxNode(self, gameState, depth):
    if gameState.isOver() or depth == 0:
      return self.evaluationFunction(gameState), Directions.STOP

    maxValue = -999999
    values = []

    for action in gameState.getLegalActions(self.index):
      # TODO: remove Direction.STOP
      successor = gameState.generateSuccessor(self.index, action)
      expectiValue = self.expectiNode(self.opponents[0], successor, depth)[0]
      values.append(expectiValue)

      if expectiValue > maxValue:
        maxValue = expectiValue

    # could exist multiple actions that yield same best result
    optActions = [i for i in range(len(values)) if values[i] == maxValue]
    actions = gameState.getLegalActions(self.index)
    optAction = actions[random.choice(optActions)]

    return maxValue, optAction



  def expectiNode(self, opponent, gameState, depth):
    if gameState.isOver() or depth == 0:
      expectedUtil = self.evaluationFuntion(gameState)
      return expectedUtil, Directions.STOP

    totalValue = 0
    actions = gameState.getLegalActions(opponent)

    for action in gameState.getLegalActions(opponent):
      successor = gameState.generateSuccessor(opponent, action)
      # continue expanding the next opponent's state
      if opponent == self.opponents[0]:
        nextOpponent = opponent + 2
        expectiValue = self.expectiNode(nextOpponent, successor, depth)[0]
        totalValue += expectiValue
      else:
        totalValue += self.maxNode(gameState, depth - 1)[0]

    return totalValue / len(actions), Directions.STOP



  def evaluationFunction(self, gameState):
    # print(self)
    util.raiseNotDefined()



class OffensiveAgent(DummyAgent):

  def registerInitialState(self, gameState):
    DummyAgent.registerInitialState(self, gameState)
    self.attack = True



  def chooseAction(self, gameState):
    # agent decides when to return home and garner the points collected, based on
    # 1. how much the team score leads
    # 2. proximity to opponent
    # 3. amount of food it is holding

    # relevant data is stored in game.py
    agentState = gameState.getAgentState(self.index)
    if agentState.numCarrying > 10:
      self.attack = False


    return DummyAgent.chooseAction(self, gameState)




  def evaluationFunction(self, gameState):
    """
    Use heuristic evaluation function to estimate utilities for non-terminal states
    """
    currScore = self.getScore(gameState)
    currPos = gameState.getAgentPosition(self.index)

    distancesToFoods = []
    foodList = self.getFood(gameState).asList()
    for foodPos in foodList:
      foodDistance = self.distancer.getDistance(currPos, foodPos)
      distancesToFoods.append(foodDistance)


    distanceToClosestFood = 0
    if len(distancesToFoods) != 0:
      distanceToClosestFood = min(distancesToFoods)


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


    if self.attack:
      foodList = self.getFood(gameState).asList()
      numOfFoods = len(foodList)

      return 3 * currScore + distanceToClosestGhost - \
             2 * distanceToClosestFood - 50 * numOfFoods

    else:
      # home distance = mazeDistance(currPos, any point on board's central axis)
      boardHeight = gameState.data.layout.height
      midWidth = gameState.data.layout.width/2
      # print(boardHeight)

      homeDistances = []
      for y in range(boardHeight):
        # for all positions reachable except wall
        if (midWidth, y) in self.allPositions:
          homeDistance = self.distancer.getDistance(currPos, (midWidth, y))
          homeDistances.append(homeDistance)

      minHomeDistance = min(homeDistances)

      return 300 * distanceToClosestGhost - 5 * minHomeDistance




