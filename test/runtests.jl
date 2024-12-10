using SparseSpectralClustering
using Test
using LinearAlgebra, SparseArrays

@testset "SparseSpectralClustering.jl small" begin
    k = 3
    σ = 0.1
    X = [ rand(3,300) 1.5.+rand(3,300) 2.8.+3.0.*rand(3,400) ]
    S = [ exp( -norm(X[:,i]-X[:,j])^2 / σ^2 ) for i in 1:1000, j in 1:1000 ]
    S = fkeep!((i,j,sij) -> sij > 1e-12, sparse(S))
    idxs = spectralcluster(S, k)
    @test allequal(idxs[1:300])
    @test allequal(idxs[301:600])
    @test allequal(idxs[601:end])
    @test allunique(idxs[[1,301,601]])
end

@testset "SparseSpectralClustering.jl large" begin
    k = 3
    σ = 0.1
    X = [ rand(3,30000) 1.5.+rand(3,30000) 2.8.+3.0.*rand(3,40000) ]
    S = knnSimilarity(X, 10, σ)
    idxs = spectralcluster(S, k)
    @test allequal(idxs[1:30000])
    @test allequal(idxs[30001:60000])
    @test allequal(idxs[60001:end])
    @test allunique(idxs[[1,30001,60001]])
end