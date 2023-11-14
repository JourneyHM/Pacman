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

class Node:
    "A node in a search tree"
    def __init__(self, state, parent=None, action=None, path_cost=0, goal=None):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost, goal=goal)
    
    def __repr__(self): return '<{}>'.format(self.state)
    def __len__(self): return 0 if self.parent is None else (1+len(self.parent))
    def __lt__(self, other): return self.path_cost < other.path_cost

def expand(problem, node):
    "Expand a node, generating the children nodes."

    s = node.state
    for sNext, action, cost in problem.getSuccessors(s):
        yield Node(sNext, node, action, cost+node.path_cost)

def path_actions(node):
    "The sequence of actions to get to this node."
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]

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
    """

    node = Node(problem.getStartState())
    frontier = util.Stack()
    frontier.push(node)
    reached = {problem.getStartState(): node}
    actions = []

    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node.state):
            actions = path_actions(node)
        for child in expand(problem, node):
            s = child.state
            if s not in reached:
                reached[s] = child
                frontier.push(child)
    return actions


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    node = Node(problem.getStartState())
    frontier = util.Queue()
    frontier.push(node)
    reached = {problem.getStartState(): node}
    actions = []

    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node.state):
            actions = path_actions(node)
        for child in expand(problem, node):
            s = child.state
            if s not in reached:
                reached[s] = child
                frontier.push(child)
    return actions

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def f(n):
    return util.manhattanDistance(n.goal, n.state)

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    node = Node(problem.getStartState())
    node.goal = problem.goal
    frontier = util.PriorityQueueWithFunction(f)
    frontier.push(node)
    reached = {problem.getStartState(): node}
    actions = []

    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node.state):
            actions = path_actions(node)
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                child.goal = problem.goal
                frontier.push(child)
    return actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
