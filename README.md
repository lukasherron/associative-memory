# associative-memory

This repository contains:
  1. hopfield_network.py - contains definitions and class structure of Hopfield and Hopfield variant networks
  2. example.py - contains an example of how to simulate a hopfield network with stored patterns
  3. report.pdf - details the analysis of stability for pattern transitions in the Hopfield model, as well as the results of sequential pattern transitions in the Hopfield variant models.
  
 Notes:

* The hopfield_network.py file requires that the array2gif library is install. If it is not installed then all of the lines which call an array2gif function must be commented out in the evolve_network() definition.

* The network includes classes for the traditional Hopfield network (Hopfield 1982), as well as a variant with time delay synaptic connections (Sompolinsky 1986) and a variant with a dynamical threshold (Horn 1989).

* See example.py for an example of how to simulate a Hopfield or Hopfield variant network
