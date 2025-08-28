module SparseSpectralClustering

using LinearAlgebra, Arpack, Clustering, SparseArrays, 
    Distances, NearestNeighbors

export spectralcluster, knnSimilarity, iterativeBipartition

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

    # S = Symmetric(S)

    d = getDegree(S)
    L = makeNormalizedLaplacian(S, d)
    L = sparse(L)
    L = fkeep!((i,j,x) -> abs(x)>(ϵ), L)
    L = Symmetric(L)

    # this is the sparse, approximate eigenpair finder from Arpack
    λ, v, _ = eigs(L; nev=k, ritzvec=true, which=:SM, maxiter=100_000, tol=√(ϵ))

    v = real.(v)

    # normalise to unit length because we used the symmetric normalized Laplacian
    v ./= norm.(eachrow(v))
    # transpose to pass to clustering with rows as points
    v = transpose(v)

    clustering = kmeans(v, k, display=:iter)

    return assignments(clustering)

end

function clusterDisconnected(S; maxClusters=10, eigvalTol=1e-8)
    L, _ = makeLaplacian(S, :symmetric)
    @inbounds @fastmath λ, v, _ = eigs(
        L; 
        nev = maxClusters, 
        ritzvec = true, 
        which = :SM, 
        maxiter = 10*size(L,1), 
        tol = eigvalTol
    )
    λ = abs.(λ)
    v = abs.(v)
    numClusters = count(<(eigvalTol), λ)
    if numClusters == maxClusters
        @warn "There may be more highly disconnected clusters (numClusters == maxClusters)"
    end
    v = transpose(v[:, 1:numClusters])
    v ./= norm.(eachcol(v))'
    idxs = assignments(kmeans(v, numClusters, display=:iter))
    return idxs
end

function iterativeBipartition(features::AbstractVector{FEATURE_TYPE}, similarityFunc, stopFunc::Function; 
    neighbourLists=nothing, plotFunc=(idxs,iter)->nothing) where {FEATURE_TYPE}

    # similarityFunc : (FEATURE_TYPE, FEATURE_TYPE) -> Real
    # stopFunc : Vec{Bool} -> Bool

    N = length(features)

    S = makeSimilarityMatrix(features, similarityFunc, neighbourLists)
    @info "Similarity matrix density=$(nnz(S)/length(S))"
    
    @info "Checking for very disconnected clusters..."
    idxs = clusterDisconnected(S .> 0, maxClusters=10)
    queue = unique(idxs)
    lastClusterIdx = maximum(idxs)
    @info "Found $lastClusterIdx very disconnected clusters."

    for i in 1:lastClusterIdx
        if count(==(i), idxs) > 1 && !stopFunc(idxs .== i)
            push!(queue, i)
        end
    end

    mask = BitVector(undef, length(features))

    iter = 0
    plotFunc(idxs, iter)

    while !isempty(queue)

        i = pop!(queue)
        mask .= idxs .== i
        n = count(mask)
        @info "Cluster $i: length=$n."

        if n == 1 || stopFunc(mask)
            @info "Done with cluster $i."
            continue
        end

        if n == 2
            split = [true false]
        else
            split = splitCluster(S, mask)
        end

        if all(split) || !any(split)
            # could not split the cluster based on the Fiedler vector
            # push nothing to the queue
            @info "Split failed."
            continue
        end

        lastClusterIdx += 1
        in_cluster = findall(mask)
        idxs[in_cluster[split]] .= lastClusterIdx
        
        @info "Split complete: lengths $(count(split)), $(count(.!split))."
        
        push!(queue, i)
        push!(queue, lastClusterIdx)

        iter += 1
        plotFunc(idxs, iter)

    end

    return idxs

end

function splitCluster(S::AbstractArray{T}, mask) where T<:Real

    S_i = S[mask, mask]

    normalize = :none
    L, B = makeLaplacian(S_i, normalize)

    v_f = getFiedlerVec(L, B)
    
    return real.(v_f) .≥ 0

end

function getFiedlerVec(Laplacian, B=I)
    @fastmath λ, v, _ = eigs(
        Laplacian, B; 
        nev=3, 
        ritzvec=true, 
        which=:SM, 
        maxiter=10*size(Laplacian,1), 
        tol=1e-10
    )
    λ = real.(λ)
    v = real.(v)
    if λ[2] < 1e-10
        @warn "The Fiedler eigenvalue is very small."
    end
    return v[:,2]
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

function makeLaplacian(S, normalize=:none)
    d = getDegree(S)
    if normalize === :randomwalk
        D = spdiagm(d)
        return D - S, D
    elseif normalize === :symmetric
        inv_sqrt_D = spdiagm(1 ./ sqrt.(d))
        @fastmath return (I - inv_sqrt_D * S * inv_sqrt_D), I
    else
        D = spdiagm(d)
        return D - S, I
    end
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
            push!(Sv, exp(-d^2/σ^2)/2)
            push!(Si, j)
            push!(Sj, i)
            push!(Sv, exp(-d^2/σ^2)/2)
        end
    end
    return SparseArrays.sparse!(Si, Sj, Sv, n, n, +)
end

function makeSimilarityMatrix(features, distFunc, neighbourLists)

    n = length(features)
    Si = Int[]
    Sj = Int[]
    Sv = Float64[]
    for i in 1:n
        for j in neighbourLists[i]
            i == j && continue
            dist = distFunc(features[i], features[j])
            push!(Si, i)
            push!(Sj, j)
            push!(Sv, dist)
            push!(Si, j)
            push!(Sj, i)
            push!(Sv, dist)
        end
    end
    S = SparseArrays.sparse!(Si, Sj, Sv, n, n, max)
    return S

end

function makeSimilarityMatrix(features, distFunc, ::Nothing)

    n = length(features)
    Si = Int[]
    Sj = Int[]
    Sv = Float64[]
    for i in 1:n
        for j in i:n
            i == j && continue
            dist = distFunc(features[i], features[j])
            push!(Si, i)
            push!(Sj, j)
            push!(Sv, dist)
            push!(Si, j)
            push!(Sj, i)
            push!(Sv, dist)
        end
    end
    return SparseArrays.sparse!(Si, Sj, Sv, n, n, max)

end

end