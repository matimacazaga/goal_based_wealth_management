# Dynamic Portfolio Allocation for Goal Based Wealth Management

Implementation of the algorithm proposed in *Dynamic Portfolio Allocation in Goals-Based Wealth Management* (Das et al.)

The methodology seeks to maximize investor outcomes over different goals. In the GBWM framework, risk is defined as the probability that an investor does not meet those financial goals. 

The algorithm takes a given set of portfolios (with their respective &mu; and &sigma; values). Using backward recursion, the methodology selects the portfolio that maximizes the probability of meeting investor's financial goals at time t=0.

Currently, the input portfolios are all on the efficient frontier (Modern Portfolio Theory), but any set of portfolios can be used.

This implementation uses a single goal value function:

<img src="img/eq1.svg"/>

where W<sub>i</sub>(T) is the investor's wealth at time t=T and G is the goal wealth. However, a multiple goal value function can be easily implemented.

The code makes use of the [Numba](https://numba.pydata.org/) library to speed up computations. 
