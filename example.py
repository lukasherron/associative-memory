from hopfield_network.py import *

lam = 1.25

net = TraditionalNetwork(num_patterns=3, num_neurons=900)
net.generate_patterns()
net.init_network()
J = net.get_weights(lam, negative_coupling=False, cyclic=False)
network, time_arr, overlap_arr, energy_arr = net.evolve_network(2000, asyn=True)

fig = plt.figure()
for i in range(len(overlap_arr[0,:])):
    plt.plot(time_arr, overlap_arr[:, i], label="Pattern " + str(i))
plt.legend()
plt.title("lambda = " +  str(lam));
