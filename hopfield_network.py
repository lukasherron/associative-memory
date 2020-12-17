import matplotlib.pyplot as plt
import numpy as np
from array2gif import write_gif
from scipy import sparse
from scipy.special import comb
from scipy.signal import savgol_filter



def overlap(state_a, state_b):
    N = len(state_a)
    C = np.multiply(state_a, state_b).sum()
    C *= 1 / N
    return (C)

def energy(net, J):
    E = -0.5 * (net + 1) / 2 @ J @ (net + 1) / 2
    return E


class HopfieldNetwork(object):

    def __init__(self, num_patterns, num_neurons):
        self.num_patterns = num_patterns
        self.num_neurons = num_neurons

    def generate_patterns(self, correlation=0):

        patterns = [[] for _ in range(self.num_patterns)]
        num_same = int(correlation * self.num_neurons)
        for i in range(self.num_patterns):
            patterns[i] = np.random.randint(0, 2, self.num_neurons)


        self.patterns = patterns

    def init_network(self):
        pattern_init = self.patterns[0]
        network = 2 * pattern_init - 1
        # for _ in range(int(0.05*self.num_neurons)):
        #     x = int(np.random.random() * self.num_neurons)
        #     network[x] = -1*(2*pattern_init - 1)[x]

        self.network = network
        return network


class TraditionalNetwork(HopfieldNetwork):

    def __init__(self, num_patterns, num_neurons):
        super().__init__(num_patterns, num_neurons)

    def get_weights(self, lam=0, negative_coupling=False, cyclic=False):

        n_n = self.num_neurons
        n_p = self.num_patterns

        J = np.zeros((n_n, n_n))
        for k in range(n_p):
            pattern = self.patterns[k]
            next_pattern = self.patterns[(k + 1) % n_p]
            xi_current = 2 * pattern - 1
            xi_next = 2 * next_pattern - 1
            J += np.outer(xi_current, xi_current)

            if (k + 1) % n_p != 0:
                J += lam * np.outer(xi_next, xi_current)
            if cyclic:
                if (k + 1) % n_p == 0:
                    if negative_coupling:
                        J += -lam * np.outer(xi_next, xi_current)
                    else:
                        J += lam * np.outer(xi_next, xi_current)

        J += (np.random.rand(n_n, n_n) * 2 - 1) / 10
        np.fill_diagonal(J, 0)
        J *= 1 /n_n
        self.J = J
        return J

    def evolve_network(self, max_iter, samp_freq=20, theta=0, asyn=True, quiet=False):
        n_n = self.num_neurons
        J = self.J
        iter = 0
        time_arr = []
        time_arr = np.append(time_arr, iter)
        if asyn:
            if not quiet:
                images = [[] for _ in range(int(max_iter / samp_freq))]
                overlap_arr = [[] for _ in range(int(max_iter))]
                energy_arr = []
                frame = 0

            while iter < max_iter:

                i = np.random.randint(0, n_n)
                self.network[i] = np.sign(np.dot(J[i][:], self.network) - theta)
                if self.network[i] == 0:
                    x = np.random.random()
                    if x > 0.5:
                        self.network[i] = 1
                    else:
                        self.network[i] = -1

                if iter % samp_freq == 0:
                    if not quiet:
                        network_img = (self.network + 1) / 2
                        network_gif = np.array((network_img.reshape((int(n_n ** 0.5), int(n_n ** 0.5)))))
                        images[frame] = np.array([network_gif.astype('uint8') * 255,
                                                  network_gif.astype('uint8') * 255,
                                                  network_gif.astype('uint8') * 255])
                        frame += 1
                time_arr = np.append(time_arr, iter)
                overlap_arr[iter] = [overlap(self.network, 2 * pattern - 1) for pattern in self.patterns]
                energy_arr.append(energy(self.network, J))
                iter += 1
        else:
            if not quiet:
                images = [[] for _ in range(int(max_iter+1))]
                overlap_arr = [[] for _ in range(int(max_iter+1))]
                energy_arr = []
                overlap_arr[iter] = [overlap(self.network, 2 * pattern - 1) for pattern in self.patterns]
                energy_arr.append(energy(self.network, J))
                frame = 0
            while iter < max_iter:
                copy_net = self.network.copy()
                for i in range(n_n):

                    copy_net[i] = np.sign(np.dot(J[i][:], self.network) - theta)

                    if copy_net[i] == 0:
                        x = np.random.random()
                        if x > 0.5:
                            copy_net[i] = 1
                        else:
                            copy_net[i] = -1
                self.network = copy_net
                if not quiet:
                    network_img = (self.network + 1) / 2
                    network_gif = np.array((network_img.reshape((int(n_n ** 0.5), int(n_n ** 0.5)))))
                    images[frame] = np.array([network_gif.astype('uint8') * 255,
                                              network_gif.astype('uint8') * 255,
                                              network_gif.astype('uint8') * 255])
                    frame += 1
                iter += 1
                overlap_arr[iter] = [overlap(self.network, 2 * pattern - 1) for pattern in self.patterns]
                energy_arr.append(energy(self.network, J))
                time_arr = np.append(time_arr, iter)


        return self.network, time_arr, np.array(overlap_arr), np.array(energy_arr)


class TimeDelayNetwork(HopfieldNetwork):

    def __init__(self, num_patterns, num_neurons):
        super().__init__(num_patterns, num_neurons)

    def get_weights(self, lam=0):

        n_n = self.num_neurons
        n_p = self.num_patterns

        J_fast = np.zeros((n_n, n_n))
        J_slow = np.zeros((n_n, n_n))

        for k in range(n_p-1):
            pattern = self.patterns[k]
            next_pattern = self.patterns[(k + 1) % n_p]
            xi_current = 2*pattern-1
            xi_next = 2*next_pattern-1
            J_fast += np.outer(xi_current, xi_current)
            J_slow += lam * np.outer(xi_next, xi_current)

        np.fill_diagonal(J_fast, 0)
        np.fill_diagonal(J_slow, 0)
        J_fast *= 1 /n_n
        J_slow *= 1/ n_n
        self.J_fast= J_fast
        self.J_slow= J_slow
        return J_fast, J_slow

    def kernel(self, kernel_type, tau, t, time_arr):
        delta_t_arr = np.ones_like(time_arr)*t - time_arr
        if kernel_type == 'step':
            w = np.zeros_like(delta_t_arr)
            for i in range(len(delta_t_arr)):
                if delta_t_arr[i] < tau:
                    w[i] = 1/tau
                else:
                    w[i] = 0

        if kernel_type == 'exp_decay':
            w = 1 / tau * np.exp(-delta_t_arr / tau)

        if kernel_type == 'time_delay':
            w = np.zeros_like(delta_t_arr)
            for i in range(len(delta_t_arr)):
                if delta_t_arr[i] == tau:
                    w[i] = 1
                else:
                    w[i] = 0

        return w


    def evolve_network(self, max_iter, tau, kernel_type = 'step', samp_freq=20, theta=0, asyn=True, quiet=False):
        n_n = self.num_neurons
        J_slow, J_fast = self.J_slow, self.J_fast
        iter = 0
        time_arr = []
        time_arr = np.append(time_arr, iter)
        if asyn:
            if not quiet:
                images = [[] for _ in range(int(max_iter / samp_freq))]
                overlap_arr = [[] for _ in range(int(max_iter))]
                energy_arr = []
                frame = 0

            while iter < max_iter:
                if iter == 0:
                    history = np.array(self.network)
                    history = history.reshape((1, len(history)))
                else:
                    history = np.vstack((history, self.network))

                (r,_) = history.shape
                idx_range = np.arange(iter + 1 - r, iter + 1)
                if r > 5000:
                    history = history[idx_range][:]

                time_arr_prime = time_arr[idx_range]
                kern = self.kernel(kernel_type, tau, iter, time_arr_prime)
                network_bar = np.trapz(np.multiply(kern, history.T).T, time_arr, axis=0)

                i = np.random.randint(0, n_n)
                self.network[i] = np.sign(np.dot(J_fast[i][:], self.network) + np.dot(J_slow[i][:], network_bar) - theta)
                if self.network[i] == 0:
                    x = np.random.random()
                    if x > 0.5:
                        self.network[i] = 1
                    else:
                        self.network[i] = -1

                if iter % samp_freq == 0:
                    if not quiet:
                        network_img = (self.network + 1) / 2
                        network_gif = np.array((network_img.reshape((int(n_n ** 0.5), int(n_n ** 0.5)))))
                        images[frame] = np.array([network_gif.astype('uint8') * 255,
                                                  network_gif.astype('uint8') * 255,
                                                  network_gif.astype('uint8') * 255])
                        frame += 1

                overlap_arr[iter] = [overlap(self.network, 2 * pattern - 1) for pattern in self.patterns]
                iter += 1
                time_arr = np.append(time_arr, iter)
        else:
            if not quiet:
                images = [[] for _ in range(int(max_iter))]
                overlap_arr = [[] for _ in range(int(max_iter))]
                frame = 0

            while iter < max_iter:
                if iter == 0:
                    history = np.array(self.network)
                    history = history.reshape((1, len(history)))
                else:
                    history = np.vstack((history, self.network))

                (r, _) = history.shape
                idx_range = np.arange(iter + 1 - r, iter + 1)
                if r > 100:
                    history = history[idx_range][:]

                time_arr_prime = time_arr[idx_range]
                kern = self.kernel(kernel_type, tau, iter, time_arr_prime)
                network_bar = np.trapz(np.multiply(kern, history.T).T, time_arr, axis=0)

                for i in range(n_n):
                    self.network[i] = np.sign(np.dot(J_fast[i][:], self.network) + np.dot(J_slow[i][:], network_bar) - theta)
                    if self.network[i] == 0:
                        x = np.random.random()
                        if x > 0.5:
                            self.network[i] = 1
                        else:
                            self.network[i] = -1

                if not quiet:
                    network_img = (self.network + 1) / 2
                    network_gif = np.array((network_img.reshape((int(n_n ** 0.5), int(n_n ** 0.5)))))
                    images[frame] = np.array([network_gif.astype('uint8') * 255,
                                              network_gif.astype('uint8') * 255,
                                              network_gif.astype('uint8') * 255])
                    frame += 1

                overlap_arr[iter] = [overlap(self.network, 2 * pattern - 1) for pattern in self.patterns]
                iter += 1
                time_arr = np.append(time_arr, iter)

        return self.network, time_arr, np.array(overlap_arr)



class DynamicalThresholdNetwork(HopfieldNetwork):

    def __init__(self, num_patterns, num_neurons):
        super().__init__(num_patterns, num_neurons)
        self.network = None
        self.J = None

    def get_weights(self, lam, negative_coupling=False, cyclic=False):

        n_n = self.num_neurons
        n_p = self.num_patterns

        J = np.zeros((n_n, n_n))
        for k in range(n_p):
            pattern = self.patterns[k]
            next_pattern = self.patterns[(k + 1) % n_p]
            xi_current = 2 * pattern - 1
            xi_next = 2 * next_pattern - 1
            J += np.outer(xi_current, xi_current)
            J += lam*np.outer(xi_next, xi_current)
            if cyclic:
                if (k + 1) % n_p == 0:
                    if negative_coupling:
                        J += -lam * np.outer(xi_next, xi_current)
                    else:
                        J += lam * np.outer(xi_next, xi_current)

        J += (np.random.rand(n_n, n_n) * 2 - 1) / 10
        np.fill_diagonal(J, 0)
        J *= 1 / n_n
        self.J = J
        return J

    def threshold(self, R_current, S_next, c):
        R_next = R_current/c + S_next

        return R_next

    def evolve_network(self, max_iter, c, g, samp_freq=20, asyn=True, quiet=False):
        n_n = self.num_neurons
        J = self.J
        iter = 0
        time_arr, theta_arr = [], []
        time_arr = np.append(time_arr, iter)
        R_current = np.zeros((n_n,1))
        b = g*(c-1)/c
        if asyn:
            if not quiet:
                images = [[] for _ in range(int(max_iter / samp_freq))]
                overlap_arr = [[] for _ in range(int(max_iter))]
                energy_arr = []
                frame = 0

            while iter < max_iter:
                i = np.random.randint(0, n_n)
                theta = b * R_current[i]
                self.network[i] = np.sign(np.dot(J[i][:], self.network) - theta)
                if self.network[i] == 0:
                    x = np.random.random()
                    if x > 0.5:
                        self.network[i] = 1
                    else:
                        self.network[i] = -1

                if iter % samp_freq == 0:
                    if not quiet:
                        network_img = (self.network + 1) / 2
                        network_gif = np.array((network_img.reshape((int(n_n ** 0.5), int(n_n ** 0.5)))))
                        images[frame] = np.array([network_gif.astype('uint8') * 255,
                                                  network_gif.astype('uint8') * 255,
                                                  network_gif.astype('uint8') * 255])
                        frame += 1

                overlap_arr[iter] = [overlap(self.network, 2 * pattern - 1) for pattern in self.patterns]
                iter += 1
                time_arr = np.append(time_arr, iter)
                theta_arr = np.append(theta_arr, theta)
                R_current[i] = self.threshold(R_current[i], self.network[i], c)

        else:
            if not quiet:
                images = [[] for _ in range(int(max_iter))]
                overlap_arr = [[] for _ in range(int(max_iter))]
                energy_arr = []
                frame = 0

            while iter < max_iter:
                copy_net = self.network.copy()
                for i in range(n_n):
                    copy_net[i] = np.sign(np.dot(J[i][:], copy_net) - b * R_current[i])
                    if copy_net[i] == 0:
                        x = np.random.random()
                        if x > 0.5:
                            copy_net[i] = 1
                        else:
                            copy_net[i] = -1
                    R_current[i] = self.threshold(R_current[i], copy_net[i], c)
                self.network = copy_net

                if not quiet:
                    network_img = (self.network + 1) / 2
                    network_gif = np.array((network_img.reshape((int(n_n ** 0.5), int(n_n ** 0.5)))))
                    images[frame] = np.array([network_gif.astype('uint8') * 255,
                                              network_gif.astype('uint8') * 255,
                                              network_gif.astype('uint8') * 255])
                    frame += 1

                overlap_arr[iter] = [overlap(self.network, 2 * pattern - 1) for pattern in self.patterns]
                iter += 1
                time_arr = np.append(time_arr, iter)


        return self.network, time_arr, np.array(overlap_arr)

