using Jchemo
using Test 

@testset "vip dispatch" begin
    X = [1. 2 3 4; 4 1 6 7; 12 5 6 13; 27 18 7 6; 12 11 28 7]
    y = [10., 120, 8, 1, 100]
    model = plskern(; nlv = 3)
    fit!(model, X, y)

    @test length(vip(model.fitm).vi) == size(X, 2)
    @test length(vip(model.fitm, nothing).vi) == size(X, 2)
    @test length(vip(model.fitm, 2).vi) == size(X, 2)
    @test length(vip(model.fitm, y).vi) == size(X, 2)
end
