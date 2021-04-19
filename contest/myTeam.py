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
    self.updateBelief(gameState)
    stateCopy = gameState.deepCopy()

    for opponent in self.opponents:
      posConfig = self.updatePosState(opponent)
      stateCopy.data.agentStates[opponent] = game.AgentState(posConfig, gameState.getAgentState(opponent).isPacman)

    optAction = self.maxNode(stateCopy, 2)[1]
    return optAction


  def maxNode(self, gameState, depth):
    if gameState.isOver() or depth == 0:
      return self.evaluationFunction(gameState), Directions.STOP

    maxValue = -999999
    values = []

    for action in gameState.getLegalActions(self.index):
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
      return expectedUtil, 0

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

    print("minScaredTime: " + str(minScaredTime))

    if minGhostDist < 4 and minScaredTime < 5:
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

    print("carrying: " + str(agentState.numCarrying))
    print("minGhostDist: " + str(minGhostDist))
    print(self.attack)

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

      return currScore - 3 * distanceToClosestFood - 300 * numOfFoods - 10 * minCapsuleDist + 80 * distanceToClosestGhost

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

      return 400 * distanceToClosestGhost - 4 * minHomeDistance

