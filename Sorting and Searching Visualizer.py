import pygame
import math
import random
from queue import Queue
from queue import PriorityQueue
from queue import LifoQueue
pygame.init()
pygame.font.init()

#define a width as frame is a square
WIDTH = 800
WIN = pygame.display.set_mode((WIDTH,WIDTH))

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128 , 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class button():
	def __init__(self, color, x, y, width, height, text = ''):
		self.color = color
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.text = text

	def drawButton(self, win, outline = None):
		if outline:
			pygame.draw.rect(win, outline, (self.x-3, self.y-3, self.width+6, self.height+6))
		pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height))

		if self.text != '':
			font = pygame.font.SysFont('comicsans', 40)
			text = font.render(self.text, 1, BLACK)
			win.blit(text, (self.x + (self.width/2 - text.get_width()/2), self.y + (self.height/2 - text.get_height()/2)))

	def isOver(self, pos):
		if pos[0]>self.x and pos[0]<self.x+self.width:
			if pos[1]>self.y and pos[1]<self.y+self.height:
				return True
		return False


class Cell:
	def __init__(self, row, col, width, total_rows):
		self.row = row
		self.col = col
		self.x = row * width
		self.y = col * width
		self.color = WHITE
		self.neighbors = []
		self.width =  width
		self.total_rows = total_rows

	def get_pos(self):
		return self.row, self.col

	#checking of red squares (already considered)
	def is_closed(self):
		return self.color == RED

	def is_open(self):
		return self.color == GREEN

	def is_barrier(self):
		return self.color == BLACK

	def is_start(self):
		return self.color == ORANGE

	def is_end(self):
		return self.color == BLUE

	def reset(self):
		self.color = WHITE

	def make_start(self):
		self.color = ORANGE

	def make_closed(self):
		self.color = RED

	def make_open(self):
		self.color = GREEN

	def make_barrier(self):
		self.color = BLACK

	def make_end(self):
		self.color = BLUE

	def make_path(self):
		self.color = PURPLE

	#also passing where to draw? It's the window of line 8
	def draw(self, win):
		pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

	# basically the maintains the graph properly
	def update_neighbors(self, grid):
		self.neighbors = []
		#checking the cell adjacent to self

		#down
		if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): 
			self.neighbors.append(grid[self.row + 1][self.col])
		#up
		if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): 
			self.neighbors.append(grid[self.row - 1][self.col])
		#left
		if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): 
			self.neighbors.append(grid[self.row][self.col - 1])
		#right
		if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): 
			self.neighbors.append(grid[self.row][self.col + 1])

	# less than (compares 2 cells)
	def __lt__(self, other):
		return False


#defining a good heuristic function for a*, manhattan here
def h(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw):
	while current in came_from:
		current = came_from[current]
		current.make_path()
		draw()

#need for a DS to hold the grid
def make_grid(rows, width):
	grid = []
	gap = width // rows
	for i in range(rows):
		grid.append([])
		for j in range(rows):
			cell = Cell(i, j, gap, rows)
			grid[i].append(cell)

	return grid

#Now we need to define some grid lines
def draw_grid(win, rows,width):
	gap = width // rows
	for i in range(rows):
		pygame.draw.line(win, GREY, (0, i*gap), (width, i*gap))
		for j in range(rows):
			pygame.draw.line(win, GREY, (j*gap, 0), (j*gap, width))

# main draw function

def draw(win, grid, rows, width):
	win.fill(WHITE)

	for row in grid:
		for cell in row:
			cell.draw(win)

	draw_grid(win, rows, width)
	# tells pygame that whatever has been drawn update it to display
	pygame.display.update()

#takes mouse position and figures out the cell
def get_clicked_pos(pos, rows, width):
	gap = width // rows
	y, x = pos

	row = y // gap
	col = x // gap

	return row, col


def drawWindow(win, Button):
	Button.drawButton(win, BLACK)

def Astar(draw, grid, start, end):
	count = 0
	cur_dat = PriorityQueue()
	# zero here is the f-score
	cur_dat.put((0, count, start))
	came_from = {}
	#python hacks to initialise
	g_score = {spot: float("inf") for row in grid for spot in row}
	g_score[start] = 0
	f_score = {spot: float("inf") for row in grid for spot in row}
	f_score[start] = h(start.get_pos(), end.get_pos())

	#this keeps track of things in/out of priority_queue
	seen = {start}

	while not cur_dat.empty():
		#as this algo would be running ,if we want someone
		#to exit it we need to keep the following check
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

		current = cur_dat.get()[2] #the node
		cur_dat
		seen.remove(current)

		if current == end:
			reconstruct_path(came_from, end, draw)
			end.make_end()
			return True

		for neighbor in current.neighbors:
			temp_g_score = g_score[current] + 1

			if temp_g_score < g_score[neighbor]:
				came_from[neighbor] = current
				g_score[neighbor] = temp_g_score
				f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())

				if neighbor not in seen:
					count += 1
					cur_dat.put((f_score[neighbor], count, neighbor))
					seen.add(neighbor)
					neighbor.make_open() #inserted
		draw()

		if current != start:
			current.make_closed() #already considered

	return False

def BFS(draw, grid, start, end):
	cur_dat = Queue()
	cur_dat.put(start)
	came_from = {}
	#python hacks to initialise
	dist = {spot: float("inf") for row in grid for spot in row}
	dist[start] = 0;

	while not cur_dat.empty():
		#as this algo would be running ,if we want someone
		#to exit it we need to keep the following check
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

		current = cur_dat.get() #the node

		if current == end:
			reconstruct_path(came_from, end, draw)
			end.make_end()
			return True

		for neighbor in current.neighbors:
			temp_dist = dist[current] + 1

			if temp_dist < dist[neighbor]:
				came_from[neighbor] = current
				dist[neighbor] = temp_dist
				cur_dat.put(neighbor)
				neighbor.make_open()
		draw()

		if current != start:
			current.make_closed() #already considered

	return False



def DFS(draw, grid, start, end):
	cur_dat = LifoQueue()
	cur_dat.put(start)
	came_from = {}
	#python hacks to initialise
	vis = {spot: 0 for row in grid for spot in row}
	vis[start] = 1;

	while not cur_dat.empty():
		#as this algo would be running ,if we want someone
		#to exit it we need to keep the following check
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

		current = cur_dat.get() #the node

		for neighbor in current.neighbors:
			if not vis[neighbor]:
				came_from[neighbor] = current
				vis[neighbor] = 1
				if neighbor == end:
					reconstruct_path(came_from, end, draw)
					end.make_end()
					return True
				cur_dat.put(neighbor)
				neighbor.make_open()
		draw()

		if current != start:
			current.make_closed() #already considered

	return False



def doAlgo(text, win, width):
	ROWS = 50
	grid = make_grid(ROWS, width)
	start = None
	end = None

	run = True
	while run:
		draw(win ,grid, ROWS, width)
		for event in pygame.event.get(): #an event is mouse click or timer went off
			if event.type == pygame.QUIT:
				run = False

			if pygame.mouse.get_pressed()[0]: # 0 index is for left, 2 for right, 1 for middle
				pos = pygame.mouse.get_pos()
				row, col = get_clicked_pos(pos, ROWS, width)
				cell = grid[row][col]
				if not start and cell != end:
					start = cell
					start.make_start()

				elif not end and cell != start: 
					end = cell
					end.make_end()

				elif cell != end and cell != start:
					cell.make_barrier()

			elif pygame.mouse.get_pressed()[2]:
				pos = pygame.mouse.get_pos()
				row, col = get_clicked_pos(pos, ROWS, width)
				cell = grid[row][col]
				cell.reset()
				if cell == start:
					start = None
				elif cell == end:
					end = None
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE and start and end:
					for row in grid:
						for cell in row:
							cell.update_neighbors(grid)

					#A-star function
					if text == 'Astar':
						Astar(lambda: draw(win ,grid, ROWS, width), grid, start, end)
					elif text == 'Dfs':
						DFS(lambda: draw(win, grid, ROWS, width), grid, start, end)
					elif text == 'Bfs':
						BFS(lambda: draw(win, grid, ROWS, width), grid, start, end)

				if event.key == pygame.K_c:
					start = None
					end = None
					grid = make_grid(ROWS, width)

				if event.key == pygame.K_x:
					return


def searchAlgos(win, width):
	run = True
	Astar = button(TURQUOISE, 250, 300, 250, 100,'A-Star')
	Bfs = button(TURQUOISE, 250, 150, 250, 100, 'Bfs')
	Dfs = button(TURQUOISE, 250, 450, 250, 100, 'Dfs')
	while run:
		win.fill(WHITE)
		drawWindow(win, Astar)
		drawWindow(win, Dfs)
		drawWindow(win, Bfs)
		pygame.display.update()

		for event in pygame.event.get():
			pos = pygame.mouse.get_pos()

			if event.type == pygame.QUIT:
				run = False
				return
			if event.type == pygame.MOUSEBUTTONDOWN:
				if Astar.isOver(pos):
					doAlgo('Astar', win, width)
				elif Dfs.isOver(pos):
					doAlgo('Dfs', win, width)
				elif Bfs.isOver(pos):
					doAlgo('Bfs', win, width)
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_x:
					return
			if event.type == pygame.MOUSEMOTION:
				if Astar.isOver(pos):
					Astar.color = GREEN
					Dfs.color = TURQUOISE
					Bfs.color = TURQUOISE
				elif Dfs.isOver(pos):
					Dfs.color = GREEN
					Astar.color = TURQUOISE
					Bfs.color = TURQUOISE
				elif Bfs.isOver(pos):
					Dfs.color = TURQUOISE
					Astar.color = TURQUOISE
					Bfs.color = GREEN
				else:
					Astar.color = TURQUOISE
					Dfs.color = TURQUOISE
					Bfs.color = TURQUOISE

class Block:
	def __init__(self, side, base, gap, maxHeight):
		self.gap = gap
		self.base = base #upfill
		self.side = side #left emptly
		self.color = GREEN
		self.border = BLACK
		self.height = random.randint(5,maxHeight) 

	def is_sel(self):
		return self.color == RED
	def is_cur(self):
		return self.color == PURPLE
	def is_norm(self):
		return self.color == GREEN

	def draw(self, win, cur):
		pygame.draw.rect(win, self.border, (self.side+cur*self.gap, self.base-self.height, self.gap, self.height))
		pygame.draw.rect(win, self.color, (self.side+cur*self.gap+2, self.base-self.height+2, self.gap-4, self.height-4))

	def make_sel(self):
		self.color = RED
	def make_cur(self):
		self.color = PURPLE
	def make_mid(self):
		self.color = BLUE
	def make_sweep(self):
		self.color = ORANGE
	def make_norm(self):
		self.color = GREEN



#generates a random array
def make_arr(width, arrSize, maxHeight):
	arr = []
	gap = width // (1.5*arrSize)
	base = 500
	side = 120
	for i in range(arrSize):
		block = Block(side, base, gap, maxHeight)
		arr.append(block)

	return arr

def drawArr(win, arr, arrSize):
	for i in range(arrSize):
		arr[i].draw(win, i)

	pygame.display.update()

def bubbleSort(win, drawArr, arr, arrSize):
	for curSize in reversed(range(arrSize)):
		for i in range(curSize):
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
			arr[i].make_sel()
			arr[i+1].make_cur()
			win.fill(WHITE)
			drawArr()
			pygame.time.delay(70)
			if arr[i].height>arr[i+1].height:
				arr[i],arr[i+1] = arr[i+1],arr[i]
			win.fill(WHITE)
			drawArr()
			pygame.time.delay(70)
			arr[i].make_norm()
		arr[curSize].make_norm()

def insertionSort(win, drawArr, arr, arrSize):
	for i in range(1, arrSize):
		for j in reversed(range(1,i+1)):
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
			if arr[j].height>arr[j-1].height:
				break
			arr[j].make_sel()
			arr[j-1].make_cur()
			win.fill(WHITE)
			drawArr()
			pygame.time.delay(70)
			arr[j],arr[j-1] = arr[j-1],arr[j]
			win.fill(WHITE)
			drawArr()
			pygame.time.delay(70)
			arr[j].make_norm()
			arr[j-1].make_norm()

# mergesort visual utility
def setArr(arr, tarr, l, mid, r, i, j, arrSize):
	darr = []
	for x in range (0,l):
		darr.append(arr[x])
	for x in tarr:
		darr.append(x)
	for x in range(i,mid+1):
		darr.append(arr[x])
	for x in range(j,r+1):
		darr.append(arr[x])
	for x in range(r+1,arrSize):
		darr.append(arr[x])
	return darr;

def mergeSort(win, arr, arrSize, l, r):
	if l==r:
		return
	mid = (l+r)//2
	mergeSort(win, arr, arrSize, l, mid)
	mergeSort(win, arr, arrSize, mid+1, r)
	tarr = []
	i = l
	j = mid+1
	while i<=mid and j<=r:
		for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
		arr[i].make_sel()
		arr[j].make_cur()
		darr = setArr(arr, tarr, l, mid, r, i, j, arrSize)
		darr[mid].make_mid()
		win.fill(WHITE)
		drawArr(win, darr, arrSize)
		arr[i].make_norm()
		arr[j].make_norm()
		darr[mid].make_norm()
		pygame.time.delay(70)
		if arr[i].height<=arr[j].height:
			tarr.append(arr[i])
			i+=1
		else:
			tarr.append(arr[j])
			j+=1
		darr = setArr(arr, tarr, l, mid, r, i, j, arrSize)
		darr[mid].make_mid()
		win.fill(WHITE)
		drawArr(win, darr, arrSize)
		darr[mid].make_norm()
		pygame.time.delay(70)

	while i<=mid:
		for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
		darr = setArr(arr, tarr, l, mid, r, i, j, arrSize)
		darr[mid].make_mid()
		darr[i].make_sel()
		win.fill(WHITE)
		drawArr(win, darr, arrSize)
		pygame.time.delay(70)
		darr[mid].make_norm()
		darr[i].make_norm()
		tarr.append(arr[i])
		i+=1
		darr = setArr(arr, tarr, l, mid, r, i, j, arrSize)
		darr[mid].make_mid()
		win.fill(WHITE)
		drawArr(win, darr, arrSize)
		darr[mid].make_norm()
		pygame.time.delay(70)

	while j<=r:
		for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
		darr = setArr(arr, tarr, l, mid, r, i, j, arrSize)
		darr[mid].make_mid()
		darr[j].make_sel()
		win.fill(WHITE)
		drawArr(win, darr, arrSize)
		pygame.time.delay(70)
		darr[mid].make_norm()
		darr[j].make_norm()
		tarr.append(arr[j])
		j+=1
		darr = setArr(arr, tarr, l, mid, r, i, j, arrSize)
		darr[mid].make_mid()
		win.fill(WHITE)
		drawArr(win, darr, arrSize)
		darr[mid].make_norm()
		pygame.time.delay(70)

	for i in range(l,r+1):
		arr[i] = tarr[i-l]
	arr[mid].make_mid()
	for i in range(l,r+1):
		for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
		arr[i].make_sweep()
		drawArr(win, arr, arrSize)
		arr[i].make_norm()
		pygame.time.delay(100)
		if i == mid:
			arr[i].make_mid()
	arr[mid].make_norm()

def drawWindow(win, Button):
	Button.drawButton(win, BLACK)

def doSort(win, arr, arrSize, text):
	if text == 'BubbleSort':
		bubbleSort(win, lambda: drawArr(win, arr, arrSize), arr, arrSize)
	elif text == 'InsertionSort':
		insertionSort(win, lambda: drawArr(win, arr, arrSize), arr, arrSize)
	elif text == 'MergeSort':
		mergeSort(win, arr, arrSize, 0, arrSize-1)
def sortAlgos(win, width):
	arrSize = 30
	maxHeight = 400
	arr = make_arr(width, arrSize, maxHeight)

	run = True
	BubbleSort = button(TURQUOISE, 20, 550, 200, 100, 'Bubble Sort')
	Shuffle = button(TURQUOISE , 280, 675, 200, 100, 'Shuffle')
	InSort = button(TURQUOISE,540, 550, 200, 100, 'Ins. Sort')
	MgSort = button(TURQUOISE,280, 550, 200, 100, 'Merge-Sort')
	while run:
		win.fill(WHITE)
		drawWindow(win, BubbleSort)
		drawWindow(win, Shuffle)
		drawWindow(win, InSort)
		drawWindow(win, MgSort)
		drawArr(win, arr, arrSize)
		pygame.display.update()
		
		for event in pygame.event.get():
			pos = pygame.mouse.get_pos()
			if event.type == pygame.QUIT:
				run = False
				return
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_x:
					return
			if event.type == pygame.MOUSEBUTTONDOWN:
				if BubbleSort.isOver(pos):
					doSort(win, arr, arrSize, 'BubbleSort')
				if Shuffle.isOver(pos):
					random.shuffle(arr)
				if InSort.isOver(pos):
					doSort(win, arr, arrSize, 'InsertionSort')
				if MgSort.isOver(pos):
					doSort(win, arr, arrSize, 'MergeSort')
			if event.type == pygame.MOUSEMOTION:
				if BubbleSort.isOver(pos):
					BubbleSort.color = GREY
				elif Shuffle.isOver(pos):
					Shuffle.color = GREY
				elif InSort.isOver(pos):
					InSort.color = GREY
				elif MgSort.isOver(pos):
					MgSort.color = GREY
				else:
					BubbleSort.color = TURQUOISE
					Shuffle.color = TURQUOISE
					MgSort.color = TURQUOISE
					InSort.color = TURQUOISE


def main(win, width):
	run = True
	sortButton = button(TURQUOISE, 250, 450, 250, 100,'Sorting')
	searchButton = button(TURQUOISE, 250, 300, 250, 100, 'Searching')
	while run:
		win.fill(WHITE)
		drawWindow(win, sortButton)
		drawWindow(win, searchButton)
		pygame.display.update()

		for event in pygame.event.get():
			pos = pygame.mouse.get_pos()

			if event.type == pygame.QUIT:
				run = False
				pygame.quit()
				quit()
			if event.type == pygame.MOUSEBUTTONDOWN:
				if sortButton.isOver(pos):
					sortAlgos(win, width)
				if searchButton.isOver(pos):
					searchAlgos(win, width)
			if event.type == pygame.MOUSEMOTION:
				if sortButton.isOver(pos):
					sortButton.color = GREEN
					searchButton.color = TURQUOISE
				elif searchButton.isOver(pos):
					searchButton.color = GREEN
					sortButton.color = TURQUOISE
				else:
					sortButton.color = TURQUOISE
					searchButton.color = TURQUOISE


main(WIN, WIDTH)