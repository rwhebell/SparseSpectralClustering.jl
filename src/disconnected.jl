function clusterDisconnected(S::AbstractMatrix{T}; minClusters=1, maxClusters=10) where {T}

    tol = √eps(T)
    N = size(S,1)
    I_minus_L = make_I_minus_L(S)

    if N < 4
        λ, v = eigen(Symmetric(Matrix(I_minus_L)))
    else
        λ, v, _ = eigs(
            I_minus_L;
            nev = maxClusters,
            ritzvec = true,
            which = :LR,
            tol,
            check = 1,
            maxiter = N
        )
    end

    λ = 1 .- real.(λ)
    p = sortperm(λ)

    numSmallEigs = count(<(tol), λ)
    numClusters = min( length(λ), max( numSmallEigs, minClusters ) )
    if numClusters == maxClusters
        @warn "There may be more highly disconnected clusters (numClusters == maxClusters)"
    elseif numClusters < 2
        return ones(Int, size(S,1))
    end

    v = transpose(real.(v[:, p[1:numClusters]]))
    v ./= norm.(eachcol(v))'
    idxs = assignments(kmeans(v, numClusters))

    return idxs

end