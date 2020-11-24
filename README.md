# ARCS: An order-based BN learning method
We implement Annealing Regularized Cholesky Score (ARCS) algorithm to search over topological sorts for a high-scoring Bayesian network. Our scoring function is derived from regularizing Gaussian DAG likelihood. We combine global simulated annealing over permutations with a fast proximal gradient algorithm, operating on triangular matrices of edge coefficients, to compute our scoring function. 

![alt text](https://github.com/yeqiaoling/ARCS-BN/blob/master/Poster-SoCal.pdf?raw=true)


## Getting Started
Download and install [MATLAB](https://www.mathworks.com/downloads/).

## Running the tests
Clone the package.
```
$ git clone https://github.com/yeqiaoling/ARCS-BN.git
```

Run the **test.m** file in the directory **/arcs_bn/test** for an example. 

## Publication
[Optimizing regularized Cholesky score for order-based learning of Bayesian networks. Qiaoling Ye, Arash A. Amini, and Qing Zhou. IEEE-TPAMI. 2020.](https://www.computer.org/csdl/journal/tp/5555/01/09079582/1jmV9bJGu6Q)

Or access the paper from [arXiv](https://arxiv.org/abs/1904.12360).
