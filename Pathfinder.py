import pygame
import math
import sys
import queue as Q
from queue import PriorityQueue
import tkinter as tk
from tkinter import *
from tkinter import simpledialog
from tkinter import messagebox

"""

PATHFINDING ALGORITHM PROGRAM
- Your first click will be the position of the starting node
- Your second click will be the position of the ending node
- Press 'Space' to start the algorithm search
- Press 'c' to clear the board
After the algorithm is done finding a shortest path, you may add in more obstacles and press 'Space' to run the search again.

Supported Algorithms:
- A*
- BFS
- DFS
- Custom
    The custom algorithm is used when you are given the start and end positions. It determines the quadrant of the end node and only searches nodes that are in that quadrant.

"""

application_window = tk.Tk()

# Choose Algorithm
algorithm = 0
algorithms = ["A* Algorithm", "BFS Algorithm", "DFS Algorithm", "Custom"]

flag = True
while flag:
    messagebox.showinfo("Algorithms", "0 - A*, 1 - BFS, 2 - DFS, 3 - Custom")
    algorithm = simpledialog.askinteger("Input", "Pick an Algorithm from 0 to 3", parent = application_window, initialvalue = 0)
    if algorithm is None:
        sys.exit("Program Cancelled at Algorithm Input")
    flag = False


# Set values for the program
flag = True
while flag:
    width = simpledialog.askinteger("Input", "Enter Grid Dimension (Default to 800 if value not entered or less than 200)", parent = application_window, minvalue = 200, initialvalue = 800)
    if width is None:
        sys.exit("Program Cancelled at Width Input")        
    WIDTH = width

    rows = simpledialog.askinteger("Input", "Enter # of Rows (Default to 50 is value is not entered or less than 10)", parent = application_window, minvalue = 10, initialvalue = 50)
    if rows is None:
        sys.exit("Program Cancelled at Row Input")
    ROWS = rows

    WIN = pygame.display.set_mode((WIDTH, WIDTH))
    flag = False

pygame.display.set_caption(algorithms[algorithm])

# Define Colours
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)

class Node:
    def __init__(self, row, col, width, total_rows): # n x n grid, therefore total_rows = total_columns 
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbours = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def getSearched(self):
        return self.color == RED

    def getSearching(self):
        return self.color == GREEN

    def getObstacle(self):
        return self.color == BLACK

    def getStart(self):
        return self.color == ORANGE

    def getEnd(self):
        return self.color == BLUE

    def reset(self):
        self.color = WHITE

    def setStart(self):
        self.color = ORANGE

    def setSearched(self):
        self.color = RED

    def setSearching(self):
        self.color = GREEN

    def setObstacle(self):
        self.color = BLACK

    def setEnd(self):
        self.color = BLUE

    def setPath(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    # This function checks each direction of the current node and sees if we can move in that direction
    def update_neighbours(self, grid):
        self.neighbours = []
        if self.row < self.total_rows - 1: # Checks if lower bound 
            if not grid[self.row + 1][self.col].getObstacle(): # CHECKS DOWN
                self.neighbours.append(grid[self.row + 1][self.col])

        if self.row > 0: # Checks upper bound 
            if not grid[self.row - 1][self.col].getObstacle(): # CHECKS UP
                self.neighbours.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1: # Checks right bound
            if not grid[self.row][self.col + 1].getObstacle(): # CHECKS RIGHT
                self.neighbours.append(grid[self.row][self.col + 1])

        if self.col > 0: # Checks left bound
            if not grid[self.row][self.col - 1].getObstacle(): # CHECKS LEFT
                self.neighbours.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False

# Function H(n) using Manhattan Distance
def heuristic(p1, p2):
    currentx = p1[0]
    currenty = p1[1]
    goalx = p2[0]
    goaly = p2[1]
    dist = abs(goalx - currentx) + abs(goaly - currenty)
    return dist


def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.setPath()
        draw()
   

# A* Algorithm
def astar(draw, grid, start, end):
    count = 0
    
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}

    # Function G(n)
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0

    # Function F(n)
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = heuristic(start.get_pos(), end.get_pos()) # Heuristic Estimate

    open_set_hash = {start} # Keeps track of items are in the priority queue

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2] # Current will be the start node
        open_set_hash.remove(current)

        if current == end: # Checks if we found end node
            reconstruct_path(came_from, end, draw)
            end.setEnd()
            return True

        for neighbour in current.neighbours:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = temp_g_score + heuristic(neighbour.get_pos(), end.get_pos())
                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.setSearching()

        draw()

        if current != start:
            current.setSearched()

    return False


# BFS Algorithm
def bfs(draw, grid, start, end):
    # Keep track of all visited nodes
    explored = []
    came_from = {}
    
    # Keep track of nodes to be checked
    queue = [[start]]

    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        path = queue.pop(0)
        current = path[-1]
        
        if current not in explored:
            if current is not start:
                current.setSearching()

            for neighbour in current.neighbours:
                came_from[neighbour] = current
                newPath = list(path)
                newPath.append(neighbour)
                queue.append(newPath)
                if neighbour == end:
                    #reconstruct_path(came_from, end, draw)
                    end.setEnd()
                    return True

            explored.append(current)

        draw()

        if current != start:
            current.setSearched()

    return False

# DFS Algorithm
def dfs(draw, grid, start, end):
    # Keep track of all visited nodes
    visited = []
    came_from = {}

    
    # Keep track of nodes to be checked
    queue = [[start]]

    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        path = queue.pop(0)
        current = path[-1]

        if current not in visited:
            if current is not start:
                current.setSearching()

            newList = (list(list(set(current.neighbours)-set(visited))+list(set(visited)-set(current.neighbours))))
            for neighbour in newList:
                newPath = list(path)
                newPath.append(neighbour)
                queue.append(newPath)
                if neighbour == end:
                    end.setEnd()
                    return True

            visited.append(current)

        draw()

        if current != start:
            current.setSearched()

    return False

# Custom Algorithm
def custom(draw, grid, start, end):
    explored = []
    came_from = {}
    startx = start.get_pos()[0]
    starty = start.get_pos()[1]
    endx = end.get_pos()[0]
    endy = end.get_pos()[1]

    tl = False # Top Left
    tr = False # Top Right
    bl = False # Bottom Left
    br = False # Bottom Right
    
    # Find which direction the end point is
    if startx < endx:
        if starty < endy:
            br = True
        else:
            tr = True
    else:
        if starty < endy:
            bl = True
        else:
            tl = True

    queue = [[start]]
    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        path = queue.pop(0)
        current = path[-1]
                
        if current not in explored:
            if current is not start:
                current.setSearching()

            # neighbours = [bottom,top,right,left]
            for neighbour in current.neighbours:
                neighbourx = neighbour.get_pos()[0]
                neighboury = neighbour.get_pos()[1]
                
                if tl is True:
                    if neighbourx >= endx and neighboury >= endy and neighbourx <= startx and neighboury <= starty:
                        #came_from[neighbour] = current
                        newPath = list(path)
                        newPath.append(neighbour)
                        queue.append(newPath)
                        if neighbour == end:
                            reconstruct_path(came_from, end, draw)
                            end.setEnd()
                            return True
                    
                elif bl is True:
                    if neighbourx >= endx and neighboury <= endy and neighbourx <= startx and neighboury >= starty:
                        #came_from[neighbour] = current
                        newPath = list(path)
                        newPath.append(neighbour)
                        queue.append(newPath)
                        if neighbour == end:
                            reconstruct_path(came_from, end, draw)
                            end.setEnd()
                            return True
                    
                elif tr is True:
                    if neighbourx <= endx and neighboury >= endy and neighbourx >= startx and neighboury <= starty:
                        #came_from[neighbour] = current
                        newPath = list(path)
                        newPath.append(neighbour)
                        queue.append(newPath)
                        if neighbour == end:
                            reconstruct_path(came_from, end, draw)
                            end.setEnd()
                            return True
                    
                elif br is True:
                    if neighbourx <= endx and neighboury <= endy and neighbourx >= startx and neighboury >= starty:
                        #came_from[neighbour] = current
                        newPath = list(path)
                        newPath.append(neighbour)
                        queue.append(newPath)
                        if neighbour == end:
                            reconstruct_path(came_from, end, draw)
                            end.setEnd()
                            return True

            explored.append(current)

        draw()

        if current != start:
            current.setSearched()
        
    return False
        

# Djikstra's Algorithm
#def djikstra(draw, grid, start, end):


def makeGrid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)

    return grid


def drawGrid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i*gap), (width, i*gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j*gap, 0), (j*gap, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for node in row:
            node.draw(win)

    drawGrid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col


def main(win, width):
    grid = makeGrid(ROWS, width)

    start = None
    end = None
    reached = False

    run = True
    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]: # Set Nodes
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                if not start and node != end:
                    start = node
                    start.setStart()

                elif not end and node != start:
                    end = node
                    end.setEnd()

                elif node != end and node != start:
                    node.setObstacle()

            elif pygame.mouse.get_pressed()[2]: # Delete Nodes
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                node.reset()
                if node == start:
                    start = None
                elif node == end:
                    end = None

            if event.type == pygame.KEYDOWN: # Start Algorithm
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbours(grid)

                    # Pick Algorithm
                    if algorithm == 0:
                        astar(lambda: draw(win, grid, ROWS, width), grid, start, end)

                        # Shortest path check for A*
                        reached = astar
                        if reached is False:
                            messagebox.showinfo("Information","No shortest path found.")
                        else:
                            print(reached)
                            messagebox.showinfo("Information","Shortest path found.")
                            
                    elif algorithm == 1:
                        bfs(lambda: draw(win, grid, ROWS, width), grid, start, end)

                        # Shortest path check for BFS
                        reached = bfs
                        if reached is False:
                            messagebox.showinfo("Information","No shortest path found.")
                        else:
                            messagebox.showinfo("Information","Shortest path found.")
                            
                    
                    elif algorithm == 2:
                        dfs(lambda: draw(win, grid, ROWS, width), grid, start, end)

                        # Shortest path check for DFS
                        reached = dfs
                        if reached is False:
                            messagebox.showinfo("Information","No shortest path found.")
                        else:
                            messagebox.showinfo("Information","Shortest path found.")

                            
                    elif algorithm == 3:
                        custom(lambda: draw(win, grid, ROWS, width), grid, start, end)

                        # Shortest path check for DFS
                        reached = custom
                        if reached is False:
                            messagebox.showinfo("Information","No shortest path found.")
                        else:
                            messagebox.showinfo("Information","Shortest path found.")
                            
                    '''
                    elif algorithm == 4:
                        djikstra(lambda: draw(win, grid, ROWS, width), grid, start, end)
                    '''
                
                    # Shortest path check
                    '''
                    reached = astar
                    if reached is False:
                        messagebox.showinfo("Information","No shortest path found.")
                    else:
                        messagebox.showinfo("Information","Shortest path found.")
                    '''

                if event.key == pygame.K_c: # Clear The Board
                    start = None
                    end = None
                    grid = makeGrid(ROWS, width)

    pygame.quit()

main(WIN, WIDTH)
