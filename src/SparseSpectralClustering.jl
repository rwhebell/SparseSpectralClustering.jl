module SparseSpectralClustering

using LinearAlgebra, Arpack, Clustering, SparseArrays, Distances, NearestNeighbors

export spectralcluster, knnSimilarity

""" 
idxs = spectralcluster(S, k; method=:kmedoids)
- `S` is the n×n similarity matrix (should be sparse, especially if large)
- `k` is the number of clusters desired

kmeans is used on the eigenvectors of the normalized Laplacian

- `idxs` is a vector of `Int`s that give cluster numbers for each point
"""
function spectralcluster(S, k)

    @assert size(S,1) == size(S,2) "S should be square."

    # S = Symmetric(S)

    d = getDegree(S)
    L = makeNormalizedLaplacian(S, d)

    # this is the sparse, approximate eigenpair finder from Arpack
    λ, v, _ = eigs(L; nev=k, ritzvec=true, which=:SM, maxiter=10_000, tol=1e-4)

    v = real.(v)

    # normalise to unit length because we used the symmetric normalized Laplacian
    v ./= norm.(eachrow(v))
    # transpose to pass to clustering with rows as points
    v = transpose(v)

    clustering = kmeans(v, k)

    return assignments(clustering)

end

function getDegree(S)
    n = size(S, 1)
    return S * ones(eltype(S), n)
end

function makeNormalizedLaplacian(S, d)
    # Ng-Jordan-Weiss: symmetric normalized Laplacian
    inv_sqrt_D = sparse(Diagonal(d .^(-1/2)))
    return (I - inv_sqrt_D * S * inv_sqrt_D)
    # equivalent to inv_sqrt_D * (D - S) * inv_sqrt_D
end

function knnSimilarity(X, m, σ)
    n = size(X,2)
    kdtree = KDTree(X)
    Si = Int[]
    Sj = Int[]
    Sv = eltype(X)[]
    for i in 1:n
        J, dists = knn(kdtree, X[:,i], m)
        for (j, d) in zip(J, dists)
            push!(Si, i)
            push!(Sj, j)
            push!(Sv, exp(-d^2/σ^2))
        end
    end
    return SparseArrays.sparse!(Si, Sj, Sv, n, n, max)
end

end