@xport function iterativeBipartition(features::AbstractVector{FEATURE_TYPE}, similarityFunc::SF, badnessFunc::BF, neighbourLists;
    plotFunc = (idxs,iter)->nothing,
    maxClusters = length(features),
    params = ()
) where {FEATURE_TYPE, SF, BF}

    # similarityFunc : (FEATURE_TYPE, FEATURE_TYPE) -> Real
    # badnessFunc : (Vec{FEATURE_TYPE}, params::Any) -> Real (high badness clusters are split first. A badness of 0 indicates we should stop splitting)

    maxSubClusters = 5 # make this an optional input?

    ## Setup
    N = length(features)
    idxs = ones(Int, N)
    mask = BitVector(undef, N)

    queue = @NamedTuple{badness::Float64, len::Int, idx::Int}[]
    push!(queue,
        (
            badness = badnessFunc(features, params),
            len = N,
            idx = 1
        )
    )

    ## Make similarity (affinity) matrix, S
    S = makeSimilarityMatrix(features, similarityFunc, neighbourLists, params)
    @info "Similarity matrix density = $(nnz(S)/length(S))"

    ## Do an initial split for very small eigenvalues => very disconnected clusters
    # @info "Checking for very disconnected clusters..."
    # idxs = clusterDisconnected(S, maxClusters = min(10,maxClusters))
    # uniqueIdxs = unique(idxs)
    # lastClusterIdx = maximum(idxs)
    # @info "Found $lastClusterIdx very disconnected clusters."

    ## Add the very disconnected clusters to the queue
    # for idx in uniqueIdxs
    #     mask .= idxs .== idx
    #     len = count(mask)
    #     badness = badnessFunc(features, mask, params)
    #     if len > 1 && badness > 0
    #         push!(queue, (; badness, len, idx))
    #     end
    # end
    # sort!(queue) # sorts by badness first and breaks ties with length

    iter = 0
    plotFunc(idxs, iter)

    while !isempty(queue) && maximum(idxs) < maxClusters
        # implicit assumption that numClusters == lastClusterIdx (i.e., there are no empty clusters)

        badness, len, i = pop!(queue)
        mask .= idxs .== i
        @info "Cluster $i: length=$len."

        if len == 1 || badness == 0
            @info "\tCluster $i is final."
            continue
        end

        if len == 2
            split = [1, 2]
            t = 0
        else
            # split = splitCluster(S, mask, maxFiedlerIters)
            split, t, _ = @timed clusterDisconnected(S[mask, mask]; minClusters=2, maxClusters=maxSubClusters)
        end

        if length(unique(split)) == 1
            # could not split the cluster based on the Fiedler vector
            # push nothing to the queue
            @warn "\tSplit failed on cluster $i in $(t)s."
            continue
        else
            subClusterLengths = [count(==(j), split) for j in unique(split)]
            @info "\tSplit successful in $(t)s. \n\t\tlengths = $subClusterLengths"
        end

        # lastClusterIdx += 1
        # in_cluster = findall(mask)
        # idxs[in_cluster[split]] .= lastClusterIdx
        # @info "\tSplit complete: lengths $(count(split)), $(count(.!split))."

        inClusterIdxsView = view(idxs, mask)
        for s in unique(split)

            if s == 1
                thisClusterIdx = i
            else
                thisClusterIdx = maximum(idxs) + 1
            end

            inSplit = split .== s

            inClusterIdxsView[inSplit] .= thisClusterIdx

            @info "\tNew cluster #$thisClusterIdx: length $(count(inSplit))"

            badness = badnessFunc(features[mask][inSplit], params)
            len = count(inSplit)
            newQueueItem = (; badness, len, idx=thisClusterIdx)
            splice!(queue, searchsorted(queue, newQueueItem), [newQueueItem])

        end

        iter += 1
        plotFunc(idxs, iter)

    end

    return idxs

end

# function alternatingBipartition(features::AbstractVector{FEAT}, simFunc) where {FEAT}
# end

function splitCluster(S, mask, maxiter)

    S_i = S[mask, mask]

    I_minus_L = make_I_minus_L(S_i)

    v_f = getFiedlerVecFast(I_minus_L, maxiter)

    return v_f .≥ 0

end

function getFiedlerVecFast(I_minus_L::AbstractMatrix{T}, maxiter) where {T}

    tol = 1e-5

    λ, v, _ = eigs(
        I_minus_L;
        nev = 2,
        ritzvec = true,
        which = :LR,
        maxiter,
        tol,
        check=1
    )

    if length(λ) < 2
        return zeros(T, size(I_minus_L,1))
    end

    λ = 1 .- real.(λ)
    p = sortperm(λ)

    λ_f = λ[p[2]]

    v_f = real.(v[:, p[2]])

    if λ_f < 1e-15
        @warn "small Fiedler eigenvalue = $(λ_f)"
    end

    return v_f

end

function getFiedlerVec(Laplacian::AbstractMatrix{T}, B, maxiter) where {T}

    tol = 1e-8

    if size(Laplacian,1) < 6
        return getFiedlerVecExact(Laplacian, B)
    end

    λ, v, _ = eigs(
        Laplacian, B;
        nev = 2,
        ritzvec=true,
        which=:SR,
        maxiter,
        tol,
        check=1
    )

    if length(λ) < 2
        return zeros(T, size(Laplacian,1))
    end

    λ = real.(λ)
    p = sortperm(λ)

    λ_f = λ[p[2]]

    v_f = real.(v[:, p[2]])

    if λ_f < 1e-15
        @warn "small Fiedler eigenvalue = $(λ_f)"
    end

    return v_f

end

getFiedlerVecExact(L) = getFiedlerVecExact(L, I)

function getFiedlerVecExact(Laplacian::AbstractMatrix{T}, ::UniformScaling) where T

    λ, v = eigen(Symmetric(Matrix(Laplacian)))

    λ = real.(λ)
    p = sortperm(λ)

    λ_f = λ[p[2]]

    v_f = real.(v[:, p[2]])

    if λ_f < 10*eps(T)
        @warn "small Fiedler eigenvalue = $(λ_f)"
    end

    return v_f

end

function getFiedlerVecExact(Laplacian::AbstractMatrix{T}, D::Diagonal) where T

    λ, v = eigen(Symmetric(Matrix(Laplacian)), D)

    λ = real.(λ)
    p = sortperm(λ)

    λ_f = λ[p[2]]

    v_f = real.(v[:, p[2]])

    if λ_f < 10*eps(T)
        @warn "small Fiedler eigenvalue = $(λ_f)"
    end

    return v_f
end