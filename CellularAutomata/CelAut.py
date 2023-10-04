from .ca_functions2d import *
from .ca_functions import *
import numpy as np

class CAsimulator:
    def __init__(self):
        self.shape = (100, 100)
        self.field = np.array([np.zeros(self.shape)])

    def set_percentage(self, percentage):
        field = np.zeros(self.shape)
        locs = np.vstack(np.where(field==0)).T
        n = len(locs)
        div =  1 / (percentage / 100)
        changes_loc = np.random.permutation(locs)[:int(n//div)]
        field[changes_loc[:,0],changes_loc[:,1]]=1
        self.field = np.array([field])

    def set_neighborhood(self, nbhstring):
        options = ['Moore', 'VonNeumann']
        if nbhstring not in options:
            raise ValueError("Neigborhood \""+nbhstring+"\" not available. Choose from "+str(options))
        else:
            self.neighborhood = nbhstring

    def randomizer(self, p):
        if np.random.rand(1)[0] <= p:
            return True
        else:
            return False

    def custom_rule(self, neighbourhood, c, t):
        return self.evolve_rule(neighbourhood[1][1], np.sum(neighbourhood), c[0], c[1], self.field)

    def evolve_rule(self, state, alive_count, x, y, field):
        raise NotImplementedError

    def run(self, steps, asynch):
        if asynch == True:
            self.cellular_automaton = evolve2d(self.field, timesteps=steps, neighbourhood=self.neighborhood,
                            apply_rule=AsynchronousRule(self.custom_rule, num_cells=self.shape, randomize_each_cycle=True),
                            memoize=False)
        else:
            self.cellular_automaton = evolve2d(self.field, timesteps=steps, neighbourhood=self.neighborhood,
                                      apply_rule=self.custom_rule, memoize="True")

    def output(self, save=True):
        self.animation = plot2d_animate(self.cellular_automaton, save=save)

def main():
    def evolve_rule(self, state, alive_count, x, y, field):
        new_state = state
        if alive_count == 3 or alive_count == 2:
            new_state = 1
        else:
            new_state = 0
        return new_state
    
    CA = CAsimulator()
    CA.set_neighborhood("Moore")
    CA.set_percentage(50)

    CAsimulator.evolve_rule = evolve_rule
    CA.run(steps=60, asynch=False)
    CA.output(save=True)

if __name__ == '__main__':
    main()
