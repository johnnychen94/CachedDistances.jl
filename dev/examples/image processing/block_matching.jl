using Images
using TestImages
using LazyDistances

img = imresize(float.(testimage("cameraman")), (64, 64))

p = CartesianIndex(20, 30)
rₚ = CartesianIndex(3, 3) # patch size (7, 7)
patch_p = img[p-rₚ:p+rₚ]

function block_matching(f, img, p, rₚ; num_patches=10)
    R = CartesianIndices(img)
    # to simplify the implementation and not consider border case
    candidates = first(R)+rₚ:last(R)-rₚ

    # Measure the similarity of two patches with `f`. Typically, `f` is `sqeuclidean`
    patch_p = @view img[p-rₚ:p+rₚ]
    dist = map(candidates) do q
        patch_q = @view img[q-rₚ:q+rₚ]
        f(patch_p, patch_q)
    end |> vec

    # sort from small(the most similar) to large(the least similar)
    # TODO: use partialsortperm! to reduce allocation
    order = partialsortperm(dist, 1:num_patches) # 34.394 μs (4 allocations: 26.44 KiB)
    qs = @view candidates[order]
end

matched_patches = block_matching(SqEuclidean(), img, p, rₚ) # 212.266 μs (7 allocations: 39.83 KiB)
mosaicview(map(q->img[q-rₚ:q+rₚ], matched_patches); npad=2, nrow=2)

function patched_mean(img, rₚ; num_patches=10)
    out = fill(zero(eltype(img)), axes(img))

    R = CartesianIndices(img)
    for p in first(R)+rₚ:last(R)-rₚ
        matched_patches = block_matching(SqEuclidean(), img, p, rₚ; num_patches=num_patches)
        out[p] = mapreduce(q->img[q], +, matched_patches)/length(matched_patches)
    end
    out
end

patched_mean(img, rₚ) # 794.575 ms (23549 allocations: 130.86 MiB)

eval_op(x, y) = abs2(x - y)
pointwise_dist = PairwiseDistance(eval_op, img, img); # 1.982 ns (0 allocations: 0 bytes)

# pointwise_dist[I, J] is defined as f(img[I], img[J])
pq1 = pointwise_dist[CartesianIndex(1, 1), CartesianIndex(2, 2)]
pq2 = eval_op(img[CartesianIndex(1, 1)], img[CartesianIndex(2, 2)])
pq1 == pq2

R = CartesianIndices(img)
valid_R = first(R)+rₚ:last(R)-rₚ

# For simplicity, we didn't deal with boundary condition here, so it will error
# when we index with `patchwise_dist[1, 1, 1, 1]`.
patchwise_dist = let rₚ = rₚ, img = img
        PairwiseDistance(SqEuclidean(), img, img) do p, q
        # here we specify how patches are generated from given pixel p and q
        p, q = CartesianIndex(p), CartesianIndex(q)
        @views img[p-rₚ:p+rₚ], img[q-rₚ:q+rₚ]
    end
end; # 4.903 ns (0 allocations: 0 bytes)

p = CartesianIndex(4, 4)
q = CartesianIndex(5, 5)
# @btime getindex($patchwise_dist, $p, $q) # 54.374 ns (0 allocations: 0 bytes)
patchwise_dist[p, q] == sqeuclidean(img[p-rₚ:p+rₚ], img[q-rₚ:q+rₚ])

function patched_mean_fast(img, rₚ; num_patches=10)
    out = fill(zero(eltype(img)), axes(img))

    patchwise_dist = PairwiseDistance(SqEuclidean(), img, img) do p, q
        p, q = CartesianIndex(p), CartesianIndex(q)
        @views img[p-rₚ:p+rₚ], img[q-rₚ:q+rₚ]
    end

    R = CartesianIndices(img)
    R0 = first(R)+rₚ:last(R)-rₚ
    for p in R0
        dist = vec(patchwise_dist[p, R0])
        matched_patches = R0[partialsortperm(dist, 1:num_patches)]
        out[p] = mapreduce(q->img[q], +, matched_patches)/length(matched_patches)
    end
    out
end

# @btime patched_mean_fast($img, rₚ);
#  657.075 ms (26913 allocations: 131.63 MiB)
# @btime patched_mean($img, rₚ);
#  797.023 ms (23549 allocations: 130.86 MiB)
patched_mean_fast(img, rₚ) == patched_mean(img, rₚ)

eval_op(x, y) = abs2(x - y)
pointwise_dist = PairwiseDistance(eval_op, img, img, LocalWindowCache((7, 7))) # 32.575 μs (3 allocations: 980.12 KiB)

pq1 = pointwise_dist[CartesianIndex(1, 1), CartesianIndex(2, 2)]
pq2 = eval_op(img[CartesianIndex(1, 1)], img[CartesianIndex(2, 2)])
pq1 == pq2

patchwise_dist = let img=img, rₚ=rₚ
        eval_op(x, y) = abs2(x - y)
        pointwise_dist = PairwiseDistance(eval_op, img, img, LocalWindowCache((7, 7)))

        patch_eval_op(patch_p, patch_q) = mapreduce((p, q)->pointwise_dist[p, q], +, patch_p, patch_q)
        PairwiseDistance{Float32}(patch_eval_op, img, img) do p, q
        p, q = CartesianIndex(p), CartesianIndex(q)
        p-rₚ:p+rₚ, q-rₚ:q+rₚ
    end
end;

# TODO: unfortunately, this is currently slower than the non-cache version: 10x slower :cry:
# @btime getindex($patchwise_dist, $p, $q) # 526.684 ns (3 allocations: 448 bytes)
patchwise_dist[p, q] == sqeuclidean(img[p-rₚ:p+rₚ], img[q-rₚ:q+rₚ])

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

