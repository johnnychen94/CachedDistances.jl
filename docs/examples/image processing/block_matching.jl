# ---
# title: Block matching
# author: Johnny Chen
# date: 2020-10-15
# ---

# Block matching algorithm is one key stage in patch-based image processing algorithm. This example
# shows how block matching algorithm can be implemented efficiently and easily with the help of
# LazyDistances.jl.

using Images
using TestImages
using LazyDistances

img = imresize(float.(testimage("cameraman")), (64, 64))

# For given patch radius `rₚ`, we can get a patch at each pixel `p` in the image.

p = CartesianIndex(20, 30)
rₚ = CartesianIndex(3, 3) # patch size (7, 7)
patch_p = img[p-rₚ:p+rₚ]

# For each pixel `p`, what block matching does is to find a group of patches that are similar to
# `patch_p`.

function block_matching(f, img, p, rₚ; num_patches=10)
    R = CartesianIndices(img)
    ## to simplify the implementation and not consider border case
    candidates = first(R)+rₚ:last(R)-rₚ

    ## Measure the similarity of two patches with `f`. Typically, `f` is `sqeuclidean`
    patch_p = @view img[p-rₚ:p+rₚ]
    dist = map(candidates) do q
        patch_q = @view img[q-rₚ:q+rₚ]
        f(patch_p, patch_q)
    end |> vec

    ## sort from small(the most similar) to large(the least similar)
    ## TODO: use partialsortperm! to reduce allocation
    order = partialsortperm(dist, 1:num_patches) # 34.394 μs (4 allocations: 26.44 KiB)
    qs = @view candidates[order]
end

# Here we get a group of patches that are similar to `patch_p` in the sense of F-norm.

matched_patches = block_matching(SqEuclidean(), img, p, rₚ) # 212.266 μs (7 allocations: 39.83 KiB)
mosaicview(map(q->img[q-rₚ:q+rₚ], matched_patches); npad=2, nrow=2)

# This implementation looks good, but there's one key challenge here, that we need to loop over the
# whole image and do block matching for each pixel. This means we are doing a massive computation
# and there are a lot of unnecessary redundant calculation if we are doing a simple loop.
#
# For example, this is what a naive implementation of patch-based algorithm might looks like

function patched_mean(img, rₚ; num_patches=10)
    out = fill(zero(eltype(img)), axes(img))

    R = CartesianIndices(img)
    for p in first(R)+rₚ:last(R)-rₚ
        matched_patches = block_matching(SqEuclidean(), img, p, rₚ; num_patches=num_patches)
        out[p] = mapreduce(q->img[q], +, matched_patches)/length(matched_patches)
    end
    out
end

## @btime patched_mean($img, $rₚ) # 799.883 ms (23549 allocations: 130.86 MiB)
patched_mean(img, rₚ)

# What's wrong here? There are a lot of repeated calculation in `f(patch_q, patch_q)`. For example,
# when `f = SqEuclidean()`
#
# ```math
# d = \sum_{p, q} (I[p] - I[q])^2
# ```
#
# Please do note that `(I[p] -  I[q])^2` is repeatedly calculated in the whole iteration. One single
# calculation of such is not expensive, but it is a siginficant amount of work when it's in a nested
# for-loops.
#
# Among all the existing MATLAB and Python implementations, there are two things done to work around
# this key challenge. The first workaround is to pre-calculate the pixelwise distances and then
# reuse it in the inner loop. It works quite well but it also introduces another key challenge, that
# the memeory will complain for large image size; to store the result for image with size `(256,
# 256)` we need an array of size `(256, 256, 256, 256)`, which is 32 GB memory and
# unaccptable. The second workaround, is then to specify a search window, that to only search
# similar patches in a larger neighborhood of pixel `p`. For search window size `(17, 17)`, the
# memeory needed to store the result is 0.14 GB, which is more acceptable.
#
# We are not going to explore how search window and pre-calculation are implemented because this is
# quite a dirty work; you almost have to write the whole implementation in a long long for-loop.
# Instead, we are going to see how this can be efficiently and easily implemented with
# LazyDistances.jl.

# A [`PairwiseDistance`](@ref) is a lazy array that mimics the output of pairwise distance.

eval_op(x, y) = abs2(x - y)
pointwise_dist = PairwiseDistance(eval_op, (img, img)); # 1.982 ns (0 allocations: 0 bytes)

## pointwise_dist[I, J] is defined as f(img[I], img[J])
pq1 = pointwise_dist[CartesianIndex(1, 1), CartesianIndex(2, 2)]
pq2 = eval_op(img[CartesianIndex(1, 1)], img[CartesianIndex(2, 2)])
pq1 == pq2

# Generating this array does not doing any actual computation; the computation does not happen until
# you ask for the data. We can also build a patchwise distance with this.

R = CartesianIndices(img)
valid_R = first(R)+rₚ:last(R)-rₚ

## For simplicity, we didn't deal with boundary condition here, so it will error
## when we index with `patchwise_dist[1, 1, 1, 1]`.
patchwise_dist = let rₚ = rₚ, img = img
    PairwiseDistance(SqEuclidean(), (img, img)) do i
        i-rₚ:i+rₚ
    end
end; # 4.903 ns (0 allocations: 0 bytes)

p = CartesianIndex(4, 4)
q = CartesianIndex(5, 5)
## @btime getindex($patchwise_dist, $p, $q) # 53.874 ns (0 allocations: 0 bytes)
patchwise_dist[p, q] == sqeuclidean(img[p-rₚ:p+rₚ], img[q-rₚ:q+rₚ])

# This way we have generated the patchwise distances, although the actual computation doesn't happen
# until we need it.

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

## @btime patched_mean_lazy($img, $rₚ);
##  643.413 ms (37015 allocations: 131.99 MiB)
## @btime patched_mean($img, $rₚ);
##  824.653 ms (23549 allocations: 130.86 MiB)
patched_mean_lazy(img, rₚ) == patched_mean(img, rₚ)

# Great! We haven't incoporate the pre-calculation trick yet, we still get a bit faster by making
# things lazy.

# More generally speaking, pre-calculation is just a cache. `PairwiseDistance` supports a so-called
# [`LocalWindowCache`](@ref) cache which stores the results in neighborhood `(p, q-r:q+r)` for each pixel
# pair `(p, q)`.

eval_op(x, y) = abs2(x - y)
pointwise_dist = PairwiseDistance(eval_op, (img, img), LocalWindowCache((7, 7))); # 32.575 μs (3 allocations: 980.12 KiB)

pq1 = pointwise_dist[CartesianIndex(1, 1), CartesianIndex(2, 2)]
pq2 = eval_op(img[CartesianIndex(1, 1)], img[CartesianIndex(2, 2)])
pq1 == pq2

# Instead of caching the result of pixel distances, we choose to cache the result of patch distances:

patchwise_dist = let img=img, rₚ=rₚ
    PairwiseDistance(SqEuclidean(), (img, img), LocalWindowCache(size(img))) do i
        i-rₚ:i+rₚ
    end;
end;

p = CartesianIndex(4, 4)
q = CartesianIndex(5, 5)
## @btime getindex($patchwise_dist, $p, $q) # 7.946 ns (0 allocations: 0 bytes)
## @btime sqeuclidean($(img[p-rₚ:p+rₚ]), $(img[q-rₚ:q+rₚ])) # 19.858 ns (0 allocations: 0 bytes)
patchwise_dist[p, q] == sqeuclidean(img[p-rₚ:p+rₚ], img[q-rₚ:q+rₚ])

# The time 7.946 ns is the cache read overhead. This says as long as our calculate takes more than
# this amount of time, it worths caching the results. The patch distance takes about 20 ns, so caching
# it is expected to make a performance improvment.

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

## @btime patched_mean_cache($img, $rₚ);
##  752.450 ms (26920 allocations: 214.15 MiB)
## @btime patched_mean_lazy($img, $rₚ);
##  532.812 ms (26916 allocations: 131.63 MiB)
## @btime patched_mean($img, $rₚ);
##  824.653 ms (23549 allocations: 130.86 MiB)
patched_mean_cache(img, rₚ) == patched_mean(img, rₚ)

# Oops! It's worse than our non-cache version. TODO: "fix" it
