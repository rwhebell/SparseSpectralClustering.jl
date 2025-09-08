"""
idxs = spectralcluster(S, k; method=:kmedoids)
- `S` is the n×n similarity matrix (should be sparse, especially if large)
- `k` is the number of clusters desired

kmeans is used on the eigenvectors of the normalized Laplacian

- `idxs` is a vector of `Int`s that give cluster numbers for each point
"""
function spectralcluster(S::AbstractMatrix{T}, k) where T<:Real

    ϵ = eps(T)

    @assert size(S,1) == size(S,2) "S should be square."

    L = makeLaplacian(S, :symmetric)

    # this is the sparse, approximate eigenpair finder from Arpack
    λ, v, _ = eigs(L; nev=k, ritzvec=true, which=:SM, maxiter=100*size(S,1), tol=ϵ)

    v = real.(v)

    # normalise to unit length because we used the symmetric normalized Laplacian
    v ./= norm.(eachrow(v))
    # transpose to pass to clustering with rows as points
    v = transpose(v)

    clustering = kmeans(v, k)

    return assignments(clustering)

end