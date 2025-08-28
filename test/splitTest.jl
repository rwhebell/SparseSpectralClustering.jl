using Revise
using SparseSpectralClustering

using MKL, LinearAlgebra, NearestNeighbors, StatsBase, 
    StaticArrays, WriteVTK, BenchmarkTools, Arpack, 
    CuthillMcKee

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
    nbrs, dists = knn(tree, X, 5)
    nearestNbrDists = [sort!(d)[2:5] for d in dists]
    σ = median.(nearestNbrDists)
    features = zip(X, σ) |> collect
    similarityFunc(a,b) = exp( -norm(a[1]-b[1])^2 / (a[2]*b[2]) )
    stopFunc(F) = length(F)<(1.1n)
    idxs = iterativeBipartition(features, similarityFunc, stopFunc; neighbourLists=nbrs)
    vtk_grid("foo", X, MeshCell[]) do f
        f["idx"] = idxs
        f["d"] = σ
    end
end

function timeRCM(n)
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
    nbrs, dists = knn(tree, X, 5)
    nearestNbrDists = [sort!(d)[2:5] for d in dists]
    σ = median.(nearestNbrDists)
    features = zip(X, σ) |> collect
    similarityFunc(a,b) = exp( -norm(a[1]-b[1])^2 / (a[2]*b[2]) )
    S = SparseSpectralClustering.makeSimilarityMatrix(features, similarityFunc, nbrs)
    L, _ = SparseSpectralClustering.makeLaplacian(S, :symmetric)
    Lp = rcmpermute(L)
    maxClusters = 10
    eigvalTol = 1e-10
    
    println("eigs(L)")
    display(@benchmark eigs(
        $L;
        nev = $maxClusters, 
        ritzvec = true, 
        which = :SM, 
        maxiter = 10*size($L,1), 
        tol = $eigvalTol
    ))
    
    println("eigs(Symmetric(L))")
    display(@benchmark eigs(
        Symmetric($L);
        nev = $maxClusters, 
        ritzvec = true, 
        which = :SM, 
        maxiter = 10*size($L,1), 
        tol = $eigvalTol
    ))

    println("eigs(Lp)")
    display(@benchmark eigs(
        $Lp;
        nev = $maxClusters, 
        ritzvec = true, 
        which = :SM, 
        maxiter = 10*size($L,1), 
        tol = $eigvalTol
    ))

    println("eigs(Symmetric(Lp))")
    display(@benchmark eigs(
        Symmetric($Lp);
        nev = $maxClusters, 
        ritzvec = true, 
        which = :SM, 
        maxiter = 10*size($L,1), 
        tol = $eigvalTol
    ))

end