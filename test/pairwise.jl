function equality_test(dist, d, A, B)
    for p in CartesianIndices(A)
        for q in CartesianIndices(B)
            dist[p, q] != d(A[p], B[q]) && return false
        end
    end
    return true
end

@testset "pairwise" begin
    @testset "NullCache" begin
        A, B = rand(1:5, 6), rand(1:5, 4)
        d = Euclidean()

        dist = PairwiseDistance(d, A, B)
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, d, A, B)

        dist = PairwiseDistance{Float32}(d, A, B)
        @test eltype(dist) == Float32
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, d, A, B)

        # This has significant overhead
        map_op(A_inds, B_inds) = @views A[A_inds...], B[B_inds...]
        dist = PairwiseDistance(map_op, d, CartesianIndices(A), CartesianIndices(B))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, d, A, B)
    end

    @testset "LocalWindowCache" begin
        A, B = rand(1:5, 6), rand(1:5, 4)
        d = Euclidean()

        dist = PairwiseDistance(d, A, B, LocalWindowCache((4, )))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, d, A, B)

        dist = PairwiseDistance{Float32}(d, A, B, LocalWindowCache((4, )))
        @test eltype(dist) == Float32
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, d, A, B)

        # This has significant overhead
        map_op(A_inds, B_inds) = @views A[A_inds...], B[B_inds...]
        dist = PairwiseDistance(map_op, d, CartesianIndices(A), CartesianIndices(B), LocalWindowCache((4, )))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, d, A, B)
    end
end
