using SparseSpectralClustering

using LinearAlgebra, NearestNeighbors, StatsBase, 
    StaticArrays

function test(n)
    X = hcat(
        rand(3, n),
        1.1 .+ rand(3, n),
        [3.8, 0, 0] .+ 3 .* rand(3, n), 
        [-3.8, 0, 0] .+ 2 * rand(3, n),
        [0, 0, 2] .+ rand(3, n),
        [0.7, 0.7, 2.7] .+ rand(3, n),
        [1.6, 1.6, 3.6] .+ rand(3, n)
    ) |> eachcol .|> SVector{3}
    tree = KDTree(X)
    nbrs, dists = knn(tree, X, 20)
    nearestNbrDists = [sort!(d)[2:10] for d in dists]
    σ = median.(nearestNbrDists)
    features = zip(X, σ) |> collect
    similarityFunc(a,b) = exp( -norm(a[1]-b[1])^2 / (a[2]*b[2]) )
    stopFunc(mask) = count(mask) < 1.1n
    idxs = iterativeBipartition(features, similarityFunc, stopFunc; 
        neighbourLists=nbrs)
end