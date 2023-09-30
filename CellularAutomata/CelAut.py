from .ca_functions2d import *
import numpy as np

class CAsimulator:
    def __init__(self):
        self.field = np.array([np.zeros((100,100))])

    def set_percentage(self, percentage):
        field = np.zeros((100,100))
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
                                        apply_rule=self.custom_rule, memoize=True, asynch=asynch)
        else:
            self.cellular_automaton = evolve2d(self.field, timesteps=steps, neighbourhood=self.neighborhood,
                                      apply_rule=self.custom_rule, memoize="recursive", asynch=asynch)

    def output(self, save=True, show=True):
        if show == True:
            #from IPython.display import HTML
            #animation = plot2d_animate(self.cellular_automaton, save=save, show=False)
            #HTML(animation.to_html5_video())
            animation = plot2d_animate(self.cellular_automaton, save=save, show=show)
        else:
            animation = plot2d_animate(self.cellular_automaton, save=save, show=show)

def evolve_rule(self, state, alive_count, x, y, field):
    new_state = state
    if alive_count == 3 or alive_count == 2:
        new_state = 1
    else:
        new_state = 0

    return new_state
    #if state == 1:
    #    if alive_count - 1 < 2:
    #        return 0  # Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
    #    if alive_count - 1 == 2 or alive_count - 1 == 3:
    #        return 1  # Any live cell with two or three live neighbours lives on to the next generation.
    #    if alive_count - 1> 3:
    #        return 0  # Any live cell with more than three live neighbours dies, as if by overpopulation.
    #else:
    #    if alive_count == 3:
    #        return 1  # Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
    #    else:
    #        return 0

def main():
    CA = CAsimulator()
    CA.set_neighborhood("Moore")
    CA.set_percentage(50)

    CAsimulator.evolve_rule = evolve_rule
    CA.run(steps=60, asynch=False)
    CA.output(save=True, show=True)

if __name__ == '__main__':
    main()