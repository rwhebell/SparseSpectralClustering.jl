
function getDegree(S::AbstractMatrix{T}) where T
    n = size(S, 1)
    return S * ones(T, n)
end

function makeLaplacian(S, normalise=:none)
    d = getDegree(S)
    if normalise === :randomwalk
        D = spdiagm(d)
        return D - S, Diagonal(d)
    elseif normalise === :symmetric
        inv_sqrt_d = 1 ./ sqrt.(d)
        inv_sqrt_d[d .== 0] .= 0
        inv_sqrt_D = spdiagm(inv_sqrt_d)
        return (I - inv_sqrt_D * S * inv_sqrt_D), I
    elseif normalise === :none
        D = spdiagm(d)
        return D - S, I
    else
        @error "Unrecognised normalise option."
    end
end

function make_I_minus_L(S)
    # Returns D^(-1/2) * S * D^(-1/2)
    # This isn't actually a graph laplacian, L.
    # It's (I-L), which has the same eigvecs but eigvals 1 - λ(L).
    # We use this one because it's faster to find
    #   the biggest eigenvalues rather than the smallest.
    d = getDegree(S)
    inv_sqrt_D = (1 ./ sqrt.(d))
    inv_sqrt_D[d .== 0] .= 0
    L = deepcopy(S)
    nzL = nonzeros(L)
    for col in 1:size(L, 2)
        for r in nzrange(L, col)
            row = rowvals(L)[r]
            nzL[r] *= inv_sqrt_D[row] * inv_sqrt_D[col]
        end
    end
    return L
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

function makeSimilarityMatrix(features, similarityFunc, nbrs, params)

    n = length(features)

    S = SparseMatrixCOO{Float64}(n,n)

    for i in 1:n
        nbrs_i = nbrs[i]
        for j in nbrs_i
            j < i && continue
            s_ij = similarityFunc(features[i], features[j], params)
            S[i,j] = s_ij
            S[j,i] = s_ij
        end
    end

    return sparse(S)

end

function makeSimilarityMatrix(features, similarityFunc, params)

    n = length(features)
    Si = Int[]
    Sj = Int[]
    Sv = Float64[]
    for i in 1:n
        for j in i:n
            i == j && continue
            s_ij = similarityFunc(features[i], features[j], params)
            push!(Si, i)
            push!(Sj, j)
            push!(Sv, s_ij)
            push!(Si, j)
            push!(Sj, i)
            push!(Sv, s_ij)
        end
    end
    S = SparseArrays.sparse!(Si, Sj, Sv, n, n, max)
    return S

end