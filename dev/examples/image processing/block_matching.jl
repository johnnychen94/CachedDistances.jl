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

# @btime patched_mean($img, $rₚ) # 799.883 ms (23549 allocations: 130.86 MiB)
patched_mean(img, rₚ)

eval_op(x, y) = abs2(x - y)
pointwise_dist = PairwiseDistance(eval_op, (img, img)); # 1.982 ns (0 allocations: 0 bytes)

# pointwise_dist[I, J] is defined as f(img[I], img[J])
pq1 = pointwise_dist[CartesianIndex(1, 1), CartesianIndex(2, 2)]
pq2 = eval_op(img[CartesianIndex(1, 1)], img[CartesianIndex(2, 2)])
pq1 == pq2

R = CartesianIndices(img)
valid_R = first(R)+rₚ:last(R)-rₚ

# For simplicity, we didn't deal with boundary condition here, so it will error
# when we index with `patchwise_dist[1, 1, 1, 1]`.
patchwise_dist = let rₚ = rₚ, img = img
    PairwiseDistance(SqEuclidean(), (img, img)) do i
        i-rₚ:i+rₚ
    end
end; # 4.903 ns (0 allocations: 0 bytes)

p = CartesianIndex(4, 4)
q = CartesianIndex(5, 5)
# @btime getindex($patchwise_dist, $p, $q) # 53.874 ns (0 allocations: 0 bytes)
patchwise_dist[p, q] == sqeuclidean(img[p-rₚ:p+rₚ], img[q-rₚ:q+rₚ])

function patched_mean_lazy(img, rₚ; num_patches=10)
    out = fill(zero(eltype(img)), axes(img))

    patchwise_dist = PairwiseDistance(SqEuclidean(), (img, img)) do i
        i-rₚ:i+rₚ
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

# @btime patched_mean_lazy($img, $rₚ);
#  643.413 ms (37015 allocations: 131.99 MiB)
# @btime patched_mean($img, $rₚ);
#  824.653 ms (23549 allocations: 130.86 MiB)
patched_mean_lazy(img, rₚ) == patched_mean(img, rₚ)

eval_op(x, y) = abs2(x - y)
pointwise_dist = PairwiseDistance(eval_op, (img, img), LocalWindowCache((7, 7))); # 32.575 μs (3 allocations: 980.12 KiB)

pq1 = pointwise_dist[CartesianIndex(1, 1), CartesianIndex(2, 2)]
pq2 = eval_op(img[CartesianIndex(1, 1)], img[CartesianIndex(2, 2)])
pq1 == pq2

patchwise_dist = let img=img, rₚ=rₚ
    PairwiseDistance(SqEuclidean(), (img, img), LocalWindowCache(size(img))) do i
        i-rₚ:i+rₚ
    end;
end;

p = CartesianIndex(4, 4)
q = CartesianIndex(5, 5)
# @btime getindex($patchwise_dist, $p, $q) # 7.946 ns (0 allocations: 0 bytes)
# @btime sqeuclidean($(img[p-rₚ:p+rₚ]), $(img[q-rₚ:q+rₚ])) # 19.858 ns (0 allocations: 0 bytes)
patchwise_dist[p, q] == sqeuclidean(img[p-rₚ:p+rₚ], img[q-rₚ:q+rₚ])

function patched_mean_cache(img, rₚ; num_patches=10)
    out = fill(zero(eltype(img)), axes(img))

    patchwise_dist = let rₚ=rₚ
        PairwiseDistance(SqEuclidean(), (img, img), LocalWindowCache(size(img))) do i
            i-rₚ:i+rₚ
        end;
    end;

    R = CartesianIndices(img)
    R0 = first(R)+rₚ:last(R)-rₚ
    for p in R0
        dist = vec(patchwise_dist[p, R0])
        matched_patches = R0[partialsortperm(dist, 1:num_patches)]
        out[p] = mapreduce(q->img[q], +, matched_patches)/length(matched_patches)
    end
    out
end

# @btime patched_mean_cache($img, $rₚ);
#  752.450 ms (26920 allocations: 214.15 MiB)
# @btime patched_mean_lazy($img, $rₚ);
#  532.812 ms (26916 allocations: 131.63 MiB)
# @btime patched_mean($img, $rₚ);
#  824.653 ms (23549 allocations: 130.86 MiB)
patched_mean_cache(img, rₚ) == patched_mean(img, rₚ)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

