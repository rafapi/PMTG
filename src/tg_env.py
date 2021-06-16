import numpy as np
import matplotlib.pyplot as plt


class BasicTG:
    def __init__(self):
        self.num_pts = 200
        self.total_time = 1.0

        self.current_index = 0

        self.x = np.zeros(self.num_pts)
        self.y = np.zeros(self.num_pts)

        # Parameterize the eight shape path
        self.a_x_start = 2.0
        self.a_y_start = 2.0

        self.a_x = self.a_x_start
        self.a_y = self.a_y_start

        self.a_x_history = np.zeros(self.num_pts)
        self.a_y_history = np.zeros(self.num_pts)

        for i in range(self.num_pts):
            nextTime = (float(i)/self.num_pts) * self.total_time
            next_x = self.a_x * np.sin(2 * np.pi * nextTime)
            next_y = (self.a_y/2.0) * (np.sin(2 * np.pi * nextTime) * np.cos(2 * np.pi * nextTime))

            self.x[i] = next_x
            self.y[i] = next_y

            self.degrade_path()

            self.a_x_history[i] = self.a_x
            self.a_y_history[i] = self.a_y

    def compute_tg_at_index(self, time, a_x, a_y):
        time = (float(time)/self.num_pts) * self.total_time
        next_x = a_x * np.sin(2 * np.pi * time)
        next_y = (a_y/2.0) * (np.sin(2 * np.pi * time) * np.cos(2 * np.pi * time))

        return next_x, next_y

    def degrade_path(self):
        '''
            Add some deformation to the infinity fig
        '''
        self.a_x = self.a_x - (2 * self.a_x / self.num_pts)/10.0
        self.a_y = self.a_y - (self.a_y / (2 * self.num_pts))/10.0

    def view_plot(self):
        plt.scatter(self.x[0:50], self.y[0:50])
        plt.show()

    def reset(self):
        self.current_index = 0

        self.a_x = self.a_x_start
        self.a_y = self.a_y_start

        return (self.current_index, self.x[self.current_index],
                self.y[self.current_index], self.a_x, self.a_y)

    def reward(self, x, y):
        optimal_x, optimal_y = (self.x[self.current_index],
                                self.y[self.current_index])

        return -np.sqrt(((x - optimal_x)**2) + ((y - optimal_y)**2))

    def step(self, new_a_x, new_a_y, nn_x, nn_y):
        # need to recompute the TG's at index value
        x_tg, y_tg = self.compute_tg_at_index(self.current_index,
                                              new_a_x, new_a_y)

        self.current_index = self.current_index + 1
        # [time, agent's_x, agent's_y, TG's a_x, TG's a_y]
        return (self.current_index, (x_tg + nn_x), (y_tg + nn_y),
                new_a_x, new_a_y)

    def is_done(self):
        if (self.current_index >= self.num_pts - 1):
            return True
        return False
