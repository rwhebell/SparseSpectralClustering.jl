using SparseSpectralClustering
using Test
using LinearAlgebra, SparseArrays

@testset "SparseSpectralClustering.jl" begin
    k = 3
    X = [ rand(3,300) 1.1.+rand(3,300) 2.8.+3.0.*rand(3,400) ]
    S = [ exp( -norm(X[:,i]-X[:,j])^2 / 0.2^2 ) for i in 1:1000, j in 1:1000 ]
    S = fkeep!((i,j,sij) -> sij > 1e-8, sparse(S))
    idxs = spectralcluster(S, k)
    @test length(unique(idxs[1:300])) == 1
    @test length(unique(idxs[301:600])) == 1
    @test length(unique(idxs[601:end])) == 1
    @test allunique(idxs[[1,301,601]])
end