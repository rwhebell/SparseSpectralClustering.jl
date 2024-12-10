using SparseSpectralClustering
using Test
using LinearAlgebra, SparseArrays

@testset "SparseSpectralClustering.jl" begin
    k = 3
    X = [ rand(3,300) 1.1.+rand(3,300) 2.8.+3.0.*rand(3,400) ]
    S = [ exp( -norm(X[:,i]-X[:,j])^2 / 0.2^2 ) for i in 1:1000, j in 1:1000 ]
    S = fkeep!((i,j,sij) -> sij > 1e-8, sparse(S))
    idxs = spectralcluster(S, k)
    @test allequal(idxs[1:300])
    @test allequal(idxs[301:600])
    @test allequal(idxs[601:end])
    @test allunique(idxs[[1,301,601]])

    k = 3
    X = [ rand(3,30000) 1.1.+rand(3,30000) 2.8.+3.0.*rand(3,40000) ]
    S = knnSimilarity(X, 10)
    idxs = spectralcluster(S, k)
    @test allequal(idxs[1:30000])
    @test allequal(idxs[30001:60000])
    @test allequal(idxs[60001:end])
    @test allunique(idxs[[1,30001,60001]])
end