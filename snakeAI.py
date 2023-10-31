import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import numpy as np

# Constants for the directions and grid items
DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]]
EMPTY, FOOD, WALL = 0, 1, 2


class Game:
    def __init__(self, size=10):
        self.size = size 
        self.grid = np.full((size, size), WALL)
        self.grid[1:-1, 1:-1] = 0
        self.snakes = []
    
    def step(self, snake):
        snake.think(self.grid)
        status = snake.move(self.grid, self.size)
        if status == "dead":
            self.remove_snake(snake)
        if status == "ate":
            self.add_food()
        return status

    def add_food(self, pos=None):
        pos = pos if pos else self.get_empty_pos()
        self.grid[pos] = FOOD
    
    def add_snake(self, pos=None, brain=None):
        pos = pos if pos else self.get_empty_pos()
        id = len(self.snakes) + 3
        self.grid[pos] = id
        self.snakes.append(Snake(pos, id, brain))
    
    def get_empty_pos(self):
        zeros = np.argwhere(self.grid == 0)
        return tuple(zeros[np.random.randint(len(zeros))])

    def remove_snake(self, snake):
        for pos in snake.poses:
            self.grid[pos] = 0
        self.snakes.remove(snake)


class Snake:
    def __init__(self, pos, id, brain=None):
        self.id = id
        self.poses = [pos]
        self.dir = 0
        self.brain = brain 

    def move(self, grid, size):
        hy, hx = self.poses[0]
        dy, dx = DIRS[self.dir]
        ny, nx = hy + dy, hx + dx

        if grid[ny, nx] > FOOD or not(0 <= ny < size-1 and 0 <= nx < size-1):
            return "dead"

        if grid[ny, nx] == FOOD:
            grid[ny, nx] = self.id
            self.poses = [(ny, nx)] + self.poses
            return "ate"
        else:
            grid[self.poses[-1]] = 0
            grid[ny, nx] = self.id
            self.poses = [(ny, nx)] + self.poses[:-1]
            return ""
    
    def think(self, grid):
        if self.brain is None:
            self.dir = (self.dir + np.random.choice([1,0,-1])) % 4
            return

        x = [self.look(grid, DIRS[self.dir]), 
             self.look(grid, DIRS[(self.dir + 1)%4]), 
             self.look(grid, DIRS[(self.dir - 1)%4])
        ]
        
        weights = self.brain[0]
        biases = self.brain[1]

        for w, b in list(zip(weights,biases))[:-1]:
            x = np.dot(x, w)
            x += b
            x = np.maximum(0, x)
        x = np.dot(x, weights[-1])
        x += biases[-1]

        choice = np.argmax(x) - 1
        self.dir = (self.dir + choice) % 4 
    
    def look(self, grid, dir):
        hy, hx = self.poses[0]
        dist = 1
        while grid[hy + dist * dir[0], hx + dist * dir[1]] == 0:
            dist += 1
        if grid[hy + dist * dir[0], hx + dist * dir[1]] == FOOD:
            return 1 / dist
        else:
            return -1 / dist


class GeneticAlorithm:
    def __init__(self, initial_population_size=1000):
        self.population_size = initial_population_size
        self.population = [self.create_random_brain((3, 5, 3)) for _ in range(initial_population_size)]

    def find_fittest(self, population, grid_size=10, food=1, select=100):
        fitnesses = []
        for brain in population:
            fitness = 0
            game = Game(grid_size)
            game.add_snake((5,5), brain)
            for _ in range(food):
                game.add_food()

            move_limit = 100
            while move_limit > 0:
                status = game.step(game.snakes[0])
                if status == "dead":
                    break
                elif status == "ate":
                    fitness += 100
                    move_limit = 100
                else:
                    fitness -= 1            
                move_limit -= 1
            fitnesses.append(fitness)
        
        sorted_indices = np.argsort(fitnesses)[::-1]
        ordered_population = [population[i] for i in sorted_indices]
        print(f"Peak Fitnesses: {fitnesses[sorted_indices[0]]}")
        return ordered_population[:select]

    def make_new_population(self, fittest_brains, propagate=1, mutate=1, crossover=1, output_size=1000):
        out_pop = fittest_brains * propagate
        out_pop += [self.mutate(x) for x in fittest_brains for _ in range(mutate)]
        for _ in range(crossover):
            out_pop += [self.crossover(fittest_brains[np.random.randint(len(fittest_brains))], fittest_brains[np.random.randint(len(fittest_brains))])]
        out_pop += [self.create_random_brain((3, 5, 3)) for _ in range(output_size - len(out_pop))]
        return out_pop

    def crossover(self, brain1, brain2):
        new_weights = [(w1 + w2)/2 for w1, w2 in zip(brain1[0], brain2[0])]
        new_biases = [(b1 + b2)/2 for b1, b2 in zip(brain1[1], brain2[1])]
        return (new_weights, new_biases)

    def mutate(self, brain):
        new_weights = [w + np.random.randn(w.shape[0], w.shape[1])*0.01 for w in brain[0]]
        new_biases = [b + np.random.randn(b.shape[0])*0.01 for b in brain[1]]
        return (new_weights, new_biases)

    def create_random_brain(self, layer_sizes):
        n_layers = len(layer_sizes)
        weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(n_layers - 1)]
        biases = [np.random.randn(layer_sizes[i+1])*0.01 for i in range(n_layers - 1)]
        return (weights, biases)


ga = GeneticAlorithm(initial_population_size=1000)
for i in range(20):
    print(f"Running Gen: {i}")
    fittest = ga.find_fittest(ga.population, grid_size=15, food=5, select=100)
    ga.population = ga.make_new_population(fittest_brains=fittest, propagate=2, mutate=2, crossover=100, output_size=100)

def save_anim(anim, filename):
    matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\Users\\2175\\Downloads\\ffmpeg"
    writer = animation.FFMpegWriter(fps=10, metadata=dict(artist="CodersLegacy"), bitrate=4000)
    anim.save(filename, writer=writer, dpi=300) 

grid_size = 50
n_snakes = 50
n_food = 50
game = Game(size=grid_size)

for i in range(n_snakes):
    game.add_snake(brain=ga.population[np.random.randint(3)])
for _ in range(n_food):
    game.add_food()

fig, axs = plt.subplots()
cmap = ListedColormap(['black', 'green', 'white'] + ["red" for _ in range(n_snakes)])
plot = axs.matshow(game.grid, cmap=cmap)

def update(frame):
    for snake in game.snakes:
        game.step(snake)
    plot.set_data(game.grid)

ani = animation.FuncAnimation(fig, update, interval=1, frames=200)
#save_anim(ani, "snake_anim4.gif")
plt.tight_layout()
plt.show()
