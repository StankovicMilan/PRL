Abstract

Designing an irreducible, aperiodic Markov chain with a given stationary distribution over a given
set, and running the chain until its state follows ’approximately’ the stationary distribution, is
at the core of Markov chain Monte Carlo algorithms. The main challenge is that, in general,
we do not know for how many steps to run the Markov chain to obtain samples that resemble
the stationary distribution well enough. In 1996, James Gary Propp and David Bruce Wilson
designed an algorithm that tackles this challenge, by providing exact samples from the stationary
distribution and determining automatically when to stop. In this paper, we give a detailed analysis
of the algorithm, prove that it terminates with probability 1 if and only if certain conditions are
met - otherwise it terminates with probability 0, and we present a classical application to the Ising
model.

See the report for more details.
