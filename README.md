# ARCS: An order-based BN learning method
We implement Annealing Regularized Cholesky Score (ARCS) algorithm to search over topological sorts for a high-scoring Bayesian network. Our scoring function is derived from regularizing Gaussian DAG likelihood. We combine global simulated annealing over permutations with a fast proximal gradient algorithm, operating on triangular matrices of edge coefficients, to compute our scoring function. 

## Getting Started
Download and install [MATLAB](https://www.mathworks.com/downloads/).

## Running the tests
Clone the package.
```
$ git clone https://github.com/yeqiaoling/arcs_bn.git
```

Run the "test.m" file in the directory "/arcs_bn/test" for an example. 

## Paper
[Optimizing regularized Cholesky score for order-based learning of Bayesian networks. Qiaoling Ye, Arash A. Amini, and Qing Zhou. 2019.](https://arxiv.org/abs/1904.12360)
