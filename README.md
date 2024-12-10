# SparseSpectralClustering.jl

[![Build Status](https://github.com/rwhebell/SparseSpectralClustering.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/rwhebell/SparseSpectralClustering.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Uses kmeans clustering on approximated eigenvectors for the k smallest eigenvalues of the normalized graph laplacian of the similarity matrix.

Usage, where `X` is an m by n matrix of n points in m-d space:
```julia
nbrs = 10 # number of nearest neighbours used to build S
sigma = 0.1 # scale factor for similarity rule, s_{ij} = exp(-distance^2/sigma^2)
S = knnSimilarity(X, nbrs, sigma)
k = 5 # number of clusters
idxs = spectralcluster(S, k)
```
