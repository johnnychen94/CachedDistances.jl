function equality_test(dist, d, A, B)
    for i in CartesianIndices(A)
        for j in CartesianIndices(B)
            dist[i, j] != d(A[i], B[j]) && return false
        end
    end
    return true
end
function equality_test(dist, index_map, d, A, B)
    for i in CartesianIndices(A)
        for j in CartesianIndices(B)
            p = A[index_map(i)]
            q = B[index_map(j)]
            dist[i, j] != d(p, q) && return false
        end
    end
    return true
end
function equality_test(dist, index_map, getindex_op, d, A, B)
    for i in CartesianIndices(A)
        for j in CartesianIndices(B)
            p = getindex_op(A, index_map(i))
            q = getindex_op(B, index_map(j))
            dist[i, j] != d(p, q) && return false
        end
    end
    return true
end

@testset "pairwise" begin
    @testset "NullCache" begin
        A, B = rand(1:5, 6), rand(1:5, 4)
        d = Euclidean()

        # (_, _, f)
        dist = PairwiseDistance(d, (A, B))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, d, A, B)

        # {T}(_, _, f)
        dist = PairwiseDistance{Float32}(d, (A, B))
        @test eltype(dist) == Float32
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, d, A, B)

        # (index_map, _, f)
        index_map = i->CartesianIndex(1)
        dist = PairwiseDistance(index_map, d, (A, B))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, index_map, d, A, B)

        # (index_map_list, _, f)
        index_map = (
            i->CartesianIndex(1),
            j->CartesianIndex(2)
        )
        dist = PairwiseDistance(index_map, d, (A, B))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test all(isequal(d(A[1], B[2])), dist)

        # {T}(index_map, _, f)
        index_map = i->CartesianIndex(1)
        dist = PairwiseDistance{Float32}(index_map, d, (A, B))
        @test eltype(dist) == Float32
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, index_map, d, A, B)

        # {T}(index_map_list, _, f)
        index_map = (
            i->CartesianIndex(1),
            j->CartesianIndex(2)
        )
        dist = PairwiseDistance{Float32}(index_map, d, (A, B))
        @test eltype(dist) == Float32
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test all(isequal(d(A[1], B[2])), dist)

        # (index_map, getindex_op, f)
        index_map = i->i[1]
        getindex_op = (A, i)->i
        dist = PairwiseDistance(index_map, getindex_op, d, (A, B))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, index_map, getindex_op, d, A, B)

        # (index_map, getindex_op, f)
        index_map = i->i[1]
        getindex_op = (A, i)->i
        dist = PairwiseDistance(index_map, getindex_op, d, (A, B))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, index_map, getindex_op, d, A, B)

        # (index_map_list, getindex_op_list, f)
        index_map = (
            i->i[1],
            j->j[1]
        )
        getindex_op = (
            (A, i)->i,
            (A, j)->j
        )
        dist = PairwiseDistance(index_map, getindex_op, d, (A, B))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, index_map[1], getindex_op[1], d, A, B)

        # {T}(index_map, getindex_op, f)
        index_map = i->i[1]
        getindex_op = (A, i)->i
        dist = PairwiseDistance{Float32}(index_map, getindex_op, d, (A, B))
        @test eltype(dist) == Float32
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, index_map, getindex_op, d, A, B)

        # (index_map_list, getindex_op_list, f)
        index_map = (
            i->i[1],
            j->j[1]
        )
        getindex_op = (
            (A, i)->i,
            (A, j)->j
        )
        dist = PairwiseDistance{Float32}(index_map, getindex_op, d, (A, B))
        @test eltype(dist) == Float32
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, index_map[1], getindex_op[1], d, A, B)

        @testset "Invalid cases" begin
            # all inputs should be tuple

            index_map = [
                i->i[1],
                j->j[1]
            ]
            getindex_op = [
                (A, i)->i,
                (A, j)->j
            ]
            dist = PairwiseDistance{Float32}(index_map, getindex_op, d, (A, B));
            @test_throws MethodError dist[1, 1]

            @test_throws MethodError PairwiseDistance(d, A, B);
            @test_throws MethodError PairwiseDistance{Float32}(d, A, B);
            @test_throws MethodError PairwiseDistance(index_map, getindex_op, d, A, B);
            @test_throws MethodError PairwiseDistance{Float32}(index_map, getindex_op, d, A, B);

            # Currently we only support two arrays
            @test_throws ArgumentError PairwiseDistance{Float32}(d, (A, B, A));
        end
    end

    @testset "LocalWindowCache" begin
        A, B = rand(1:5, 6), rand(1:5, 4)
        d = Euclidean()

        # (_, _, f)
        dist = PairwiseDistance(d, (A, B), LocalWindowCache((4, )))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, d, A, B)

        # {T}(_, _, f)
        dist = PairwiseDistance{Float32}(d, (A, B), LocalWindowCache((4, )))
        @test eltype(dist) == Float32
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, d, A, B)

        # (index_map, _, f)
        index_map = i->CartesianIndex(1)
        dist = PairwiseDistance(index_map, d, (A, B), LocalWindowCache((4, )))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, index_map, d, A, B)

        # (index_map_list, _, f)
        index_map = (
            i->CartesianIndex(1),
            j->CartesianIndex(2)
        )
        dist = PairwiseDistance(index_map, d, (A, B), LocalWindowCache((4, )))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test all(isequal(d(A[1], B[2])), dist)

        # {T}(index_map, _, f)
        index_map = i->CartesianIndex(1)
        dist = PairwiseDistance{Float32}(index_map, d, (A, B), LocalWindowCache((4, )))
        @test eltype(dist) == Float32
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, index_map, d, A, B)

        # {T}(index_map_list, _, f)
        index_map = (
            i->CartesianIndex(1),
            j->CartesianIndex(2)
        )
        dist = PairwiseDistance{Float32}(index_map, d, (A, B), LocalWindowCache((4, )))
        @test eltype(dist) == Float32
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test all(isequal(d(A[1], B[2])), dist)

        # (index_map, getindex_op, f)
        index_map = i->i[1]
        getindex_op = (A, i)->i
        dist = PairwiseDistance(index_map, getindex_op, d, (A, B), LocalWindowCache((4, )))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, index_map, getindex_op, d, A, B)

        # (index_map, getindex_op, f)
        index_map = i->i[1]
        getindex_op = (A, i)->i
        dist = PairwiseDistance(index_map, getindex_op, d, (A, B), LocalWindowCache((4, )))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, index_map, getindex_op, d, A, B)

        # (index_map_list, getindex_op_list, f)
        index_map = (
            i->i[1],
            j->j[1]
        )
        getindex_op = (
            (A, i)->i,
            (A, j)->j
        )
        dist = PairwiseDistance(index_map, getindex_op, d, (A, B), LocalWindowCache((4, )))
        @test eltype(dist) == result_type(d, A, B)
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, index_map[1], getindex_op[1], d, A, B)

        # {T}(index_map, getindex_op, f)
        index_map = i->i[1]
        getindex_op = (A, i)->i
        dist = PairwiseDistance{Float32}(index_map, getindex_op, d, (A, B), LocalWindowCache((4, )))
        @test eltype(dist) == Float32
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, index_map, getindex_op, d, A, B)

        # (index_map_list, getindex_op_list, f)
        index_map = (
            i->i[1],
            j->j[1]
        )
        getindex_op = (
            (A, i)->i,
            (A, j)->j
        )
        dist = PairwiseDistance{Float32}(index_map, getindex_op, d, (A, B), LocalWindowCache((4, )))
        @test eltype(dist) == Float32
        @test axes(dist) == (axes(A, 1), axes(B, 1))
        @test equality_test(dist, index_map[1], getindex_op[1], d, A, B)

        @testset "Invalid cases" begin
            # all inputs should be tuple

            index_map = [
                i->i[1],
                j->j[1]
            ]
            getindex_op = [
                (A, i)->i,
                (A, j)->j
            ]
            dist = PairwiseDistance{Float32}(index_map, getindex_op, d, (A, B), LocalWindowCache((4, )));
            @test_throws MethodError dist[1, 1]

            @test_throws MethodError PairwiseDistance(d, A, B, LocalWindowCache((4, )));
            @test_throws MethodError PairwiseDistance{Float32}(d, A, B, LocalWindowCache((4, )));
            @test_throws MethodError PairwiseDistance(index_map, getindex_op, d, A, B, LocalWindowCache((4, )));
            @test_throws MethodError PairwiseDistance{Float32}(index_map, getindex_op, d, A, B, LocalWindowCache((4, )));

            # Currently we only support two arrays
            @test_throws ArgumentError PairwiseDistance{Float32}(d, (A, B, A), LocalWindowCache((4, )));
        end
    end
end
