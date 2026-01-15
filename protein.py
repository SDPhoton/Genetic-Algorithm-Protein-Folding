import random
import numpy as np
import matplotlib.pyplot as plt
import copy

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================

# HP Sequence to Fold (H=Hydrophobic, P=Polar)
# A classic benchmark sequence (Length 20)
PROTEIN_SEQUENCE = "HPHPPHHPHPPHPHHPPHPH" 

# Genetic Algorithm Parameters
POPULATION_SIZE = 200
GENERATIONS = 500
MUTATION_RATE = 0.05
ELITISM_COUNT = 5  # Number of best individuals to carry over unchanged
TOURNAMENT_SIZE = 5

# Direction Encodings (Relative Moves)
# 0: Forward, 1: Left, 2: Right
MOVES = [0, 1, 2]

# ==========================================
# CORE LOGIC: COORDINATES & ENERGY
# ==========================================

def get_coordinates(moves):
    """
    Converts a sequence of relative moves into 2D (x, y) coordinates.
    Starts at (0,0), first bond is always Up (0,1).
    """
    x, y = 0, 0
    coords = [(x, y)]
    
    # Initial direction vector (Up)
    dx, dy = 0, 1
    
    # First node is at (0,0), second is always fixed at (0,1) to remove rotational redundancy
    x += dx
    y += dy
    coords.append((x, y))
    
    for move in moves:
        if move == 1:   # Left Turn relative to current direction
            dx, dy = -dy, dx
        elif move == 2: # Right Turn relative to current direction
            dx, dy = dy, -dx
        # If move == 0 (Forward), dx, dy remain unchanged
        
        x += dx
        y += dy
        coords.append((x, y))
        
    return coords

def is_valid_fold(coords):
    """Checks if the protein folds into itself (collision detection)."""
    return len(coords) == len(set(coords))

def calculate_energy(coords, sequence):
    """
    Calculates the energy of the fold based on the HP model.
    Energy = -1 for every non-covalent topological contact between two H's.
    """
    if not is_valid_fold(coords):
        return 1000  # High penalty for invalid folds (collisions)
    
    energy = 0
    n = len(coords)
    
    # coordinate to index map for fast lookup
    coord_map = {pos: i for i, pos in enumerate(coords)}
    
    for i in range(n):
        if sequence[i] != 'H':
            continue
            
        x, y = coords[i]
        
        # Check all 4 neighbors (Up, Down, Left, Right)
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        for nx, ny in neighbors:
            if (nx, ny) in coord_map:
                j = coord_map[(nx, ny)]
                
                # Check conditions for a valid H-H contact:
                # 1. The neighbor must be an H
                # 2. The neighbor must NOT be connected in the sequence (i and j are not adjacent indices)
                # 3. Avoid double counting (i < j)
                if sequence[j] == 'H' and abs(i - j) > 1 and i < j:
                    energy -= 1
                    
    return energy

# ==========================================
# GENETIC ALGORITHM CLASS
# ==========================================

class GeneticAlgorithm:
    def __init__(self, sequence):
        self.sequence = sequence
        self.seq_len = len(sequence)
        # Genome length is len(protein) - 2 because first 2 positions are fixed
        self.genome_len = self.seq_len - 2 
        self.population = []
        
    def init_population(self):
        """Initialize population with random moves."""
        self.population = []
        for _ in range(POPULATION_SIZE):
            # Create random individual
            moves = [random.choice(MOVES) for _ in range(self.genome_len)]
            self.population.append(moves)
            
    def evaluate(self, individual):
        """Calculate fitness (Energy). Lower is better."""
        coords = get_coordinates(individual)
        return calculate_energy(coords, self.sequence)
        
    def select_parent(self):
        """Tournament Selection: Pick random subset, return the best."""
        tournament = random.sample(self.population, TOURNAMENT_SIZE)
        tournament.sort(key=self.evaluate)
        return tournament[0]
        
    def crossover(self, parent1, parent2):
        """Single Point Crossover."""
        if random.random() > 0.7: # 30% chance of no crossover
            return copy.deepcopy(parent1)
            
        point = random.randint(1, self.genome_len - 1)
        child = parent1[:point] + parent2[point:]
        return child
        
    def mutate(self, individual):
        """Randomly change one move."""
        if random.random() < MUTATION_RATE:
            idx = random.randint(0, self.genome_len - 1)
            individual[idx] = random.choice(MOVES)
        return individual
        
    def run(self):
        print(f"Starting Genetic Algorithm for sequence: {self.sequence}")
        print(f"Length: {len(self.sequence)}")
        print("-" * 40)
        
        self.init_population()
        
        best_overall_score = float('inf')
        best_overall_genome = None
        
        for generation in range(GENERATIONS):
            # Sort population by fitness (Energy)
            # Note: evaluate returns energy, smaller is better (e.g. -5 is better than -2)
            scored_pop = [(ind, self.evaluate(ind)) for ind in self.population]
            scored_pop.sort(key=lambda x: x[1])
            
            best_gen_genome = scored_pop[0][0]
            best_gen_score = scored_pop[0][1]
            
            # Track global best
            if best_gen_score < best_overall_score:
                best_overall_score = best_gen_score
                best_overall_genome = copy.deepcopy(best_gen_genome)
                # If valid solution found, print update
                if best_overall_score <= 0:
                    print(f"Gen {generation}: New Best Energy = {best_overall_score}")
            
            # Elitism: Keep the best individuals
            new_pop = [ind for ind, score in scored_pop[:ELITISM_COUNT]]
            
            # Create rest of next generation
            while len(new_pop) < POPULATION_SIZE:
                p1 = self.select_parent()
                p2 = self.select_parent()
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)
                
            self.population = new_pop
            
        return best_overall_genome, best_overall_score

# ==========================================
# VISUALIZATION
# ==========================================

def plot_protein(moves, sequence):
    coords = get_coordinates(moves)
    energy = calculate_energy(coords, sequence)
    
    if not is_valid_fold(coords):
        print("Final result is invalid (self-intersecting). Cannot plot properly.")
        return

    x_vals = [c[0] for c in coords]
    y_vals = [c[1] for c in coords]
    
    plt.figure(figsize=(8, 8))
    
    # Draw Backbone
    plt.plot(x_vals, y_vals, color='black', linewidth=2, zorder=1, label='Backbone')
    
    # Draw H-H Contacts (Energy bonds)
    # We re-calculate contacts just for drawing lines
    n = len(coords)
    coord_map = {pos: i for i, pos in enumerate(coords)}
    for i in range(n):
        if sequence[i] == 'H':
            x, y = coords[i]
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for nx, ny in neighbors:
                if (nx, ny) in coord_map:
                    j = coord_map[(nx, ny)]
                    if sequence[j] == 'H' and abs(i - j) > 1 and i < j:
                        # Draw dashed line for contact
                        plt.plot([x, nx], [y, ny], color='gold', linestyle='--', linewidth=3, zorder=1)

    # Draw Amino Acids
    for i, (x, y) in enumerate(coords):
        color = 'red' if sequence[i] == 'H' else 'skyblue'
        label = sequence[i]
        
        # Marker
        plt.scatter(x, y, s=400, color=color, zorder=2, edgecolors='black')
        # Text Label (Index)
        plt.text(x, y, str(i), color='black' if sequence[i]=='P' else 'white', 
                 ha='center', va='center', fontweight='bold', zorder=3)

    # Grid settings
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    margin = 1
    plt.xlim(min(x_vals) - margin, max(x_vals) + margin)
    plt.ylim(min(y_vals) - margin, max(y_vals) + margin)
    plt.title(f"Best Fold Found\nSequence: {sequence}\nEnergy: {energy} (H-H Contacts)", fontsize=14)
    
    # Legend trick
    plt.scatter([], [], color='red', s=200, label='Hydrophobic (H)')
    plt.scatter([], [], color='skyblue', s=200, label='Polar (P)')
    plt.plot([], [], color='gold', linestyle='--', linewidth=2, label='H-H Bond')
    plt.legend()
    
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Initialize and Run
    ga = GeneticAlgorithm(PROTEIN_SEQUENCE)
    best_genome, best_score = ga.run()
    
    print("\n" + "="*40)
    print(f"FINAL RESULT")
    print("="*40)
    
    if best_score > 0:
        print(f"Could not find a valid fold in {GENERATIONS} generations.")
        print("Try increasing population size or generations.")
    else:
        print(f"Best Energy Found: {best_score}")
        print(f"Configuration (Relative Moves): {best_genome}")
        
        # Validate one last time
        coords = get_coordinates(best_genome)
        valid = is_valid_fold(coords)
        print(f"Valid Structure: {valid}")
        
        # Plot
        plot_protein(best_genome, PROTEIN_SEQUENCE)
