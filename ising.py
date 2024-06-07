# Simulate Ising model on an nxn square lattice at inverse temperature beta


#Set the parameters that you want and run the whole file. 
#It will plot a configuration sampled from the Gibbs distribution.

read_once = False       #run the read once version
n = 50                  #size of the square lattice
beta = 0.4           #inverse temperature

# beta_critical = 0.441 critical inverse temperature


####################### DO NOT MODIFY #######################################################################################################################

import numpy as np
import time as time
import matplotlib.pyplot as plt

class Ising_Chain:
    
    def __init__(self, n, beta, spin = 1, config = None):
        self.n = n
        self.beta = beta
        if config is None:
            self.config = [[spin for _ in range(n)] for _ in range((n))]
        else:
            self.config = config.copy()
        
    def get_plus_neighbors(self, i, j):
        n = self.n
        k_plus = int(self.config[(i+1)%n][j] == 1) + int(self.config[(i-1)%n][j] == 1) + int(self.config[i][(j+1)%n] == 1) + int(self.config[i][(j-1)%n] == 1)
        return k_plus
            
    def transition(self, i, j, U):
        k, k_plus = 4, self.get_plus_neighbors(i, j)
        if U < np.exp(2 * self.beta * (2 * k_plus - k)) / (np.exp(2 * self.beta * (2 * k_plus - k)) + 1):
            self.config[i][j] = 1
        else:
            self.config[i][j] = -1
            
    def __eq__(self, other):
        if self.n != other.n: return False
        if self.beta != other.beta: return False
        for i in range(self.n):
            for j in range(self.n):
                if self.config[i][j] != other.config[i][j]: return False
        return True
    
    def copy(self):
        return Ising_Chain(self.n, self.beta, config=self.config)
    
    def reset(self, spin):
        self.config = [[spin for _ in range(self.n)] for _ in range((self.n))]
    
                
class Propp_Wilson_Ising:
    def __init__(self, n, beta = 0.1):
        self.top_chain = Ising_Chain(n, beta, 1)
        self.bottom_chain = Ising_Chain(n, beta, -1)
        self.n = n
        self.beta = beta
        
        self.U = []
        self.vertices = []
        
        self.runningtimes = []
        self.energies = []
        self.steps = []
        
    def run_from_t(self, t):
        self.top_chain.reset(1)
        self.bottom_chain.reset(-1)
        for i in range(1, t+1):
            self.top_chain.transition(self.vertices[t-i][0], self.vertices[t-i][1], self.U[t-i])
            self.bottom_chain.transition(self.vertices[t-i][0], self.vertices[t-i][1], self.U[t-i])
    
    def run(self):
        self.U = [np.random.uniform()]
        self.vertices = [(np.random.randint(0, self.n), np.random.randint(0, self.n))]
        t = 1
        self.run_from_t(t)
        counter = 1
        while not self.top_chain == self.bottom_chain:
            #generate more uniform[0,1] random vars and more uniform random vertices
            #print("hey")
            self.U.extend(np.random.uniform(0, 1, t))
            i_coords = np.random.randint(0, self.n, t)
            j_coords = np.random.randint(0, self.n, t)         
            self.vertices.extend(zip(i_coords, j_coords))
            t = t * 2
            self.run_from_t(t)
            counter +=t 
        return self.top_chain.config, counter
            
    def generate_N_samples(self, N):
        self.clear()
        energies = []
        runningtimes = []
        steps = []
        for _ in range(N):
            start_time = time.time()
            state, timesteps = self.run()
            end_time = time.time()
            energies.append(get_energy(state))
            runningtimes.append(end_time - start_time)
            steps.append(2 * timesteps) #because we run 2 chains
        self.runningtimes = np.array(runningtimes)
        self.steps = np.array(steps)
        self.energies = np.array(energies)
        return energies, steps, runningtimes
    
    def plot_config(self, top = True):
        matrix = self.top_chain.config if top else self.bottom_chain.config
        plt.imshow(matrix, cmap='binary', interpolation='nearest', vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        label = 'beta = ' + str(self.beta) + 'n = ' + str(self.n)
        plt.title(label)
        plt.show()
        
    def clear(self):
        self.U = []
        self.vertices = []
        self.runningtimes = []

    def plot_n_steps(self):
        plt.hist(self.steps, bins = 10)
        plt.title('number of steps')
        plt.show()
    
    def plot_energies(self):
        plt.hist(self.energies, bins = 10)
        plt.title('energies')
        plt.show()


def get_energy(config):
    H = 0
    for i in range(len(config) - 1):
        for j in range(len(config)):
            H += config[i][j] * config[i+1][j]
            H += config[j][i] * config[j][i+1]
    return -H    
    

class Read_Once_Ising:
    def __init__(self, n, beta = 0.1, k = 1000):
        self.n = n
        self.beta = beta
        
        self.runningtimes = []
        self.energies = []
        self.steps = []
         
        self.k = k
        self.index_U = 0
        self.U = np.random.default_rng().uniform(0, 1, k)
        self.index_vertices = 0
        self.vertices = np.random.default_rng().integers(0, n, size = (k, 2))
        
        self.last_config = None
        
    def get_U(self):
        if self.index_U >= self.k:
            self.U = np.random.default_rng().uniform(0, 1, self.k)
            self.index_U = 0
        U = self.U[self.index_U]
        self.index_U +=1
        return U
    
    def get_vertex(self):
        if self.index_vertices >= self.k:
            self.vertices = np.random.default_rng().integers(0, self.n, size = (self.k, 2))
            self.index_vertices = 0
        vertex = self.vertices[self.index_vertices]
        self.index_vertices +=1
        return vertex
    
    def twin_run1(self):
        """
        runs 2 independent couplings until both of them coalesce
        returns the output of the winner and the loser, in that order
        """
        top_chain1 = Ising_Chain(self.n, self.beta, spin = 1)
        bottom_chain1 = Ising_Chain(self.n, self.beta, spin = -1)
        top_chain2 = Ising_Chain(self.n, self.beta, spin = 1)
        bottom_chain2 = Ising_Chain(self.n, self.beta, spin = -1)

        winner = 0
        coalescence1 = False
        coalescence2 = False
        while not coalescence1  or not coalescence2:
            U1 = self.get_U()
            vertex1 = self.get_vertex()
            top_chain1.transition(vertex1[0], vertex1[1], U1)
            bottom_chain1.transition(vertex1[0], vertex1[1], U1)
            
            U2 = self.get_U()
            vertex2 = self.get_vertex()
            top_chain2.transition(vertex2[0], vertex2[1], U2)
            bottom_chain2.transition(vertex2[0], vertex2[1], U2)
            
            coalescence1 = (top_chain1 == bottom_chain1)
            coalescence2 = (top_chain2 == bottom_chain2)
                    
            if coalescence1 and coalescence2 and winner == 0:
                U = self.get_U()
                winner = 1 + int(U < 0.5) #decide a winner by a fair coin toss
            elif coalescence1 and winner == 0:
                winner = 1
            elif coalescence2 and winner == 0:
                winner = 2
                
        assert(winner in [1,2])
        if winner == 1:
            w_state, l_state = top_chain1.config, top_chain2.config
        else:
            w_state, l_state = top_chain2.config, top_chain1.config
        return w_state, l_state
        
    def twin_run2(self, last_chain_state): 
        """Runs 2 independent couplings until one of them coalesces.
           Since this run is repeated Y times after the first twin run, we need to keep track
           of the chain started from the final state of the previous chain.
           We label this state as last_chain_state. We only consider the chain from the loser
           coupling""" 
        top_chain1 = Ising_Chain(self.n, self.beta, spin = 1)
        bottom_chain1 = Ising_Chain(self.n, self.beta, spin = -1)
        top_chain2 = Ising_Chain(self.n, self.beta, spin = 1)
        bottom_chain2 = Ising_Chain(self.n, self.beta, spin = -1)
        last_chain1 = Ising_Chain(self.n, self.beta, config=last_chain_state)
        last_chain2 = Ising_Chain(self.n, self.beta, config=last_chain_state)

        winner = 0
        coalescence1 = False
        coalescence2 = False
        while not coalescence1 or not coalescence2:
            U1 = self.get_U()
            vertex1 = self.get_vertex()
            top_chain1.transition(vertex1[0], vertex1[1], U1)
            bottom_chain1.transition(vertex1[0], vertex1[1], U1)
            last_chain1.transition(vertex1[0], vertex1[1], U1)

            
            U2 = self.get_U()
            vertex2 = self.get_vertex()
            top_chain2.transition(vertex2[0], vertex2[1], U2)
            bottom_chain2.transition(vertex2[0], vertex2[1], U2)
            last_chain2.transition(vertex2[0], vertex2[1], U2)

            coalescence1 = (top_chain1 == bottom_chain1)
            coalescence2 = (top_chain2 == bottom_chain2)
                       
        if coalescence1 and coalescence2:
            U = self.get_U()
            winner = 1 + int(U < 0.5) #decide a winner by a fair coin toss
        elif coalescence1:
            winner = 1
        elif coalescence2:
            winner = 2    
                          
        if winner == 1:
            return last_chain2.config
        else:
            return last_chain1.config    
        
    def run(self):
        state = self.twin_run1()[0]
        Y = np.random.default_rng().geometric(1/2) - 1
        for _ in range(Y):
            state = self.twin_run2(state)
        self.last_config = state
        return state
    
    def plot_config(self):
        matrix = self.last_config
        plt.imshow(matrix, cmap='binary', interpolation='nearest', vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        label = 'beta = ' + str(self.beta) + ', n = ' + str(self.n)
        plt.title(label)
        plt.show()
        
pw_alg = Read_Once_Ising(n, beta) if read_once else Propp_Wilson_Ising(n, beta)
_ = pw_alg.run()
pw_alg.plot_config()
        
