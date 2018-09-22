# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    stack = util.Stack()
    stack.push( (problem.getStartState(), []) )
    visited = set()
    while True:
        currentNode = stack.pop()
        currentNodeCoords = currentNode[0]
        currentNodeAction = currentNode[1]
        if problem.isGoalState(currentNodeCoords):
            return currentNodeAction
        if currentNodeCoords not in visited:
            visited.add(currentNodeCoords)
            successors = problem.getSuccessors(currentNodeCoords)
            for index in range(len(successors)):  # Left-most successor to right-most successor
                actionsList = currentNodeAction + [successors[index][1]]
                stack.push( (successors[index][0], actionsList) )

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    visited = []
    queue.push( (problem.getStartState(), []) )
    visited.append(problem.getStartState())
    while not queue.isEmpty():
        nextNode = queue.pop()
        if problem.isGoalState(nextNode[0]):
            return nextNode[1]
        successors = problem.getSuccessors(nextNode[0])  # Expanded this node
        for index in range(len(successors)):
            if successors[index][0] not in visited:
                visited.append(successors[index][0])
                actionsList = nextNode[1] + [successors[index][1]]
                queue.push( (successors[index][0], actionsList) )


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    priorityQueue = util.PriorityQueue()
    priorityQueue.push((problem.getStartState(), []), problem.getCostOfActions([]))
    visited = set()
    while True:
        if priorityQueue.isEmpty():
            break
        currentNode = priorityQueue.pop()
        currentNodeCoords = currentNode[0]
        currentNodeAction = currentNode[1]
        if problem.isGoalState(currentNodeCoords):
            return currentNodeAction
        if not currentNodeCoords in visited:
            visited.add(currentNodeCoords)
            successors = problem.getSuccessors(currentNodeCoords)
            for index in range(len(successors)):
                actionsList = currentNodeAction + [successors[index][1]]
                priorityQueue.push( (successors[index][0], actionsList), problem.getCostOfActions(actionsList) )

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    priorityQueue = util.PriorityQueue()
    totalCost = problem.getCostOfActions([]) + heuristic(problem.getStartState(), problem)
    priorityQueue.push((problem.getStartState(), []), totalCost)
    visited = []
    while True:
        if priorityQueue.isEmpty():  # Again, this should never happen.
            return
        currentNode = priorityQueue.pop()
        currentNodeCoords = currentNode[0]
        currentNodeAction = currentNode[1]
        if problem.isGoalState(currentNodeCoords):
            return currentNodeAction
        if currentNodeCoords not in visited:
            visited.append(currentNodeCoords)
            successors = problem.getSuccessors(currentNodeCoords)
            for index in range(len(successors)):
                actionsList = currentNodeAction + [successors[index][1]]
                totalCost = problem.getCostOfActions(actionsList) + heuristic(successors[index][0], problem)
                priorityQueue.push((successors[index][0], actionsList), totalCost)

# Helper functions that we may or may not use
def noMoreSuccessors(successors, visited):
    "Returns whether there are any more successors not already visited"
    if visited == [] and successors != []:
        return True
    for successor in successors:
        if successor not in visited:
            return False
    return True

def visitRemainingNodes(currentNode, successors, visited, stack, dfsOutput, parents):
    "Mutatively visits a successor if it hasn't already been visited"
    if not noMoreSuccessors(successors, visited):
        for successor in successors:
            if successor not in visited:
                visited.append(successor)
                stack.push(successor)
                dfsOutput.push(successor)
                if successor not in parents:
                    parents[successor] = currentNode
                return

def returnActionsList(goal, parents):
    "Returns Direction objects' path list from start to goal"
    actionsStack = util.Stack()
    currentNode = goal
    actionsStack.push(currentNode)
    while parents[parents[currentNode]] is not None:
        actionsStack.push(parents[currentNode])
        currentNode = parents[currentNode]
    result = []
    while not actionsStack.isEmpty():
        result.append(actionsStack.pop()[1])
    return result

def containsGoalNode(successors, problem):
    "Returns whether the goal node is one of the successors"
    if getGoalNode(successors, problem) is None:
        return False
    return True

def getGoalNode(successors, problem, currentNode, parents):
    "Returns the goal node from the list of successors, and marks its parent as currentNode"
    for successor in successors:
        if problem.isGoalState(successor):
            if successor in parents:
                parents[successor] = currentNode
            return successor
    return None

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
