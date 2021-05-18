var documenterSearchIndex = {"docs":
[{"location":"references/","page":"Function References","title":"Function References","text":"Modules = [LazyDistances, LazyDistances.CachedViews]","category":"page"},{"location":"references/#LazyDistances.PairwiseDistance","page":"Function References","title":"LazyDistances.PairwiseDistance","text":"PairwiseDistance([index_map], [getindex_op], f, As, [cache_strategy])\n\nLazily calculate the pairwise result of array-like objects As with function f.\n\nThe output dist is a read-only array with dist[i, j, ...] defined as f(p, q, ...), where ĩ = index_map(i), and p = getindex_op(As[1], ĩ). More formally speaking,\n\ndist[inds...] = f(map(getindex_op, index_map, inds, As) do proj, σ, i, A\n    proj(A, σ(i))\nend)\n\nRequired Arguments\n\nf: A function or function-like object.\nAs: A tuple of arrays.\n\nOptional Arguments\n\nindex_map: index_map is a function or function-like object that maps given index into a new index or indices. By default it is identity.\ngetindex_op: getindex_op is a function or funtion-like object that defines how p and q are get. By default it is a viewed version of getindex (without data copy).  If getindex_op is provided, then index_map should be provided, too.\ncache_strategy=NullCache(): Specify the cache strategy.\n\ntips: Tips\nindex_map and getindex_op can optionally be a tuple of length length(As). In this case, index_map[i] and getindex_op[i] will applied to As[i].\n\nCache Strategies\n\nFor non-trivial f, it could be expensive if we calculate it everytime when dist[i, j] is required. cache_strategy provides a way to automatically store some of the result in cache so that f(A[i], B[j]) only gets called once in specific cases.\n\nCurrent available cache strategies are:\n\nNullCache(): no cache behavior. Every time when you indexing dist[i, j] you're calculating f(A[i], B[j]).\nLocalWindowCache(window_size): cache a local window of i so that f(A[i], B[j]) only gets called once if j ∈ i-r:i+r, and gets called multiple times otherwise.\n\nwarning: Warning\nHow data is cached is internal implementation details. Generally, you should not directly talk to cache.\n\nExamples\n\njulia> using LazyDistances, Distances\n\njulia> A, B = 1:6, 1:4\n\njulia> dist = PairwiseDistance(Euclidean(), A, B)\n6×4 PairwiseDistance{Float64, 2}:\n 0.0  1.0  2.0  3.0\n 1.0  0.0  1.0  2.0\n 2.0  1.0  0.0  1.0\n 3.0  2.0  1.0  0.0\n 4.0  3.0  2.0  1.0\n 5.0  4.0  3.0  2.0\n\n# For each `i`, `dist[i, i-2:i+2]` will be cached, which means `f(A[i], B[j])` for `j ∈ i-2:i+2`\n# will only be calculated once.\njulia> dist = PairwiseDistance(Euclidean(), A, B, LocalWindowCache((5, )))\n6×4 PairwiseDistance{Float64, 2}:\n 0.0  1.0  2.0  3.0\n 1.0  0.0  1.0  2.0\n 2.0  1.0  0.0  1.0\n 3.0  2.0  1.0  0.0\n 4.0  3.0  2.0  1.0\n 5.0  4.0  3.0  2.0\n\n\n\n\n\n","category":"type"},{"location":"examples/image processing/block_matching/#Block-matching","page":"Block matching","title":"Block matching","text":"","category":"section"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"(Image: Source code) (Image: notebook) (Image: Author) (Image: Update time)","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"Block matching algorithm is one key stage in patch-based image processing algorithm. This example shows how block matching algorithm can be implemented efficiently and easily with the help of LazyDistances.jl.","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"using Images\nusing TestImages\nusing LazyDistances\n\nimg = imresize(float.(testimage(\"cameraman\")), (64, 64))","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"For given patch radius rₚ, we can get a patch at each pixel p in the image.","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"p = CartesianIndex(20, 30)\nrₚ = CartesianIndex(3, 3) # patch size (7, 7)\npatch_p = img[p-rₚ:p+rₚ]","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"For each pixel p, what block matching does is to find a group of patches that are similar to patch_p.","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"function block_matching(f, img, p, rₚ; num_patches=10)\n    R = CartesianIndices(img)\n    # to simplify the implementation and not consider border case\n    candidates = first(R)+rₚ:last(R)-rₚ\n\n    # Measure the similarity of two patches with `f`. Typically, `f` is `sqeuclidean`\n    patch_p = @view img[p-rₚ:p+rₚ]\n    dist = map(candidates) do q\n        patch_q = @view img[q-rₚ:q+rₚ]\n        f(patch_p, patch_q)\n    end |> vec\n\n    # sort from small(the most similar) to large(the least similar)\n    # TODO: use partialsortperm! to reduce allocation\n    order = partialsortperm(dist, 1:num_patches) # 34.394 μs (4 allocations: 26.44 KiB)\n    qs = @view candidates[order]\nend","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"Here we get a group of patches that are similar to patch_p in the sense of F-norm.","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"matched_patches = block_matching(SqEuclidean(), img, p, rₚ) # 212.266 μs (7 allocations: 39.83 KiB)\nmosaicview(map(q->img[q-rₚ:q+rₚ], matched_patches); npad=2, nrow=2)","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"This implementation looks good, but there's one key challenge here, that we need to loop over the whole image and do block matching for each pixel. This means we are doing a massive computation and there are a lot of unnecessary redundant calculation if we are doing a simple loop.","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"For example, this is what a naive implementation of patch-based algorithm might looks like","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"function patched_mean(img, rₚ; num_patches=10)\n    out = fill(zero(eltype(img)), axes(img))\n\n    R = CartesianIndices(img)\n    for p in first(R)+rₚ:last(R)-rₚ\n        matched_patches = block_matching(SqEuclidean(), img, p, rₚ; num_patches=num_patches)\n        out[p] = mapreduce(q->img[q], +, matched_patches)/length(matched_patches)\n    end\n    out\nend\n\n# @btime patched_mean($img, $rₚ) # 799.883 ms (23549 allocations: 130.86 MiB)\npatched_mean(img, rₚ)","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"What's wrong here? There are a lot of repeated calculation in f(patch_q, patch_q). For example, when f = SqEuclidean()","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"d = sum_p q (Ip - Iq)^2","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"Please do note that (I[p] -  I[q])^2 is repeatedly calculated in the whole iteration. One single calculation of such is not expensive, but it is a siginficant amount of work when it's in a nested for-loops.","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"Among all the existing MATLAB and Python implementations, there are two things done to work around this key challenge. The first workaround is to pre-calculate the pixelwise distances and then reuse it in the inner loop. It works quite well but it also introduces another key challenge, that the memeory will complain for large image size; to store the result for image with size (256, 256) we need an array of size (256, 256, 256, 256), which is 32 GB memory and unaccptable. The second workaround, is then to specify a search window, that to only search similar patches in a larger neighborhood of pixel p. For search window size (17, 17), the memeory needed to store the result is 0.14 GB, which is more acceptable.","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"We are not going to explore how search window and pre-calculation are implemented because this is quite a dirty work; you almost have to write the whole implementation in a long long for-loop. Instead, we are going to see how this can be efficiently and easily implemented with LazyDistances.jl.","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"A PairwiseDistance is a lazy array that mimics the output of pairwise distance.","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"eval_op(x, y) = abs2(x - y)\npointwise_dist = PairwiseDistance(eval_op, (img, img)); # 1.982 ns (0 allocations: 0 bytes)\n\n# pointwise_dist[I, J] is defined as f(img[I], img[J])\npq1 = pointwise_dist[CartesianIndex(1, 1), CartesianIndex(2, 2)]\npq2 = eval_op(img[CartesianIndex(1, 1)], img[CartesianIndex(2, 2)])\npq1 == pq2","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"Generating this array does not doing any actual computation; the computation does not happen until you ask for the data. We can also build a patchwise distance with this.","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"R = CartesianIndices(img)\nvalid_R = first(R)+rₚ:last(R)-rₚ\n\n# For simplicity, we didn't deal with boundary condition here, so it will error\n# when we index with `patchwise_dist[1, 1, 1, 1]`.\npatchwise_dist = let rₚ = rₚ, img = img\n    PairwiseDistance(SqEuclidean(), (img, img)) do i\n        i-rₚ:i+rₚ\n    end\nend; # 4.903 ns (0 allocations: 0 bytes)\n\np = CartesianIndex(4, 4)\nq = CartesianIndex(5, 5)\n# @btime getindex($patchwise_dist, $p, $q) # 53.874 ns (0 allocations: 0 bytes)\npatchwise_dist[p, q] == sqeuclidean(img[p-rₚ:p+rₚ], img[q-rₚ:q+rₚ])","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"This way we have generated the patchwise distances, although the actual computation doesn't happen until we need it.","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"function patched_mean_lazy(img, rₚ; num_patches=10)\n    out = fill(zero(eltype(img)), axes(img))\n\n    patchwise_dist = PairwiseDistance(SqEuclidean(), (img, img)) do i\n        i-rₚ:i+rₚ\n    end\n\n    R = CartesianIndices(img)\n    R0 = first(R)+rₚ:last(R)-rₚ\n    for p in R0\n        dist = vec(patchwise_dist[p, R0])\n        matched_patches = R0[partialsortperm(dist, 1:num_patches)]\n        out[p] = mapreduce(q->img[q], +, matched_patches)/length(matched_patches)\n    end\n    out\nend\n\n# @btime patched_mean_lazy($img, $rₚ);\n#  643.413 ms (37015 allocations: 131.99 MiB)\n# @btime patched_mean($img, $rₚ);\n#  824.653 ms (23549 allocations: 130.86 MiB)\npatched_mean_lazy(img, rₚ) == patched_mean(img, rₚ)","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"Great! We haven't incoporate the pre-calculation trick yet, we still get a bit faster by making things lazy.","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"More generally speaking, pre-calculation is just a cache. PairwiseDistance supports a so-called LocalWindowCache cache which stores the results in neighborhood (p, q-r:q+r) for each pixel pair (p, q).","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"eval_op(x, y) = abs2(x - y)\npointwise_dist = PairwiseDistance(eval_op, (img, img), LocalWindowCache((7, 7))); # 32.575 μs (3 allocations: 980.12 KiB)\n\npq1 = pointwise_dist[CartesianIndex(1, 1), CartesianIndex(2, 2)]\npq2 = eval_op(img[CartesianIndex(1, 1)], img[CartesianIndex(2, 2)])\npq1 == pq2","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"Instead of caching the result of pixel distances, we choose to cache the result of patch distances:","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"patchwise_dist = let img=img, rₚ=rₚ\n    PairwiseDistance(SqEuclidean(), (img, img), LocalWindowCache(size(img))) do i\n        i-rₚ:i+rₚ\n    end;\nend;\n\np = CartesianIndex(4, 4)\nq = CartesianIndex(5, 5)\n# @btime getindex($patchwise_dist, $p, $q) # 7.946 ns (0 allocations: 0 bytes)\n# @btime sqeuclidean($(img[p-rₚ:p+rₚ]), $(img[q-rₚ:q+rₚ])) # 19.858 ns (0 allocations: 0 bytes)\npatchwise_dist[p, q] == sqeuclidean(img[p-rₚ:p+rₚ], img[q-rₚ:q+rₚ])","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"The time 7.946 ns is the cache read overhead. This says as long as our calculate takes more than this amount of time, it worths caching the results. The patch distance takes about 20 ns, so caching it is expected to make a performance improvment.","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"function patched_mean_cache(img, rₚ; num_patches=10)\n    out = fill(zero(eltype(img)), axes(img))\n\n    patchwise_dist = let rₚ=rₚ\n        PairwiseDistance(SqEuclidean(), (img, img), LocalWindowCache(size(img))) do i\n            i-rₚ:i+rₚ\n        end;\n    end;\n\n    R = CartesianIndices(img)\n    R0 = first(R)+rₚ:last(R)-rₚ\n    for p in R0\n        dist = vec(patchwise_dist[p, R0])\n        matched_patches = R0[partialsortperm(dist, 1:num_patches)]\n        out[p] = mapreduce(q->img[q], +, matched_patches)/length(matched_patches)\n    end\n    out\nend\n\n# @btime patched_mean_cache($img, $rₚ);\n#  752.450 ms (26920 allocations: 214.15 MiB)\n# @btime patched_mean_lazy($img, $rₚ);\n#  532.812 ms (26916 allocations: 131.63 MiB)\n# @btime patched_mean($img, $rₚ);\n#  824.653 ms (23549 allocations: 130.86 MiB)\npatched_mean_cache(img, rₚ) == patched_mean(img, rₚ)","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"Oops! It's worse than our non-cache version. TODO: \"fix\" it","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"","category":"page"},{"location":"examples/image processing/block_matching/","page":"Block matching","title":"Block matching","text":"This page was generated using DemoCards.jl and Literate.jl.","category":"page"},{"location":"examples/statistics/distances/#Row/Column-wise-Distances","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"","category":"section"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"(Image: Source code) (Image: notebook) (Image: Author) (Image: Update time)","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"This example shows how LazyDistances.PairwiseDistance can be used to get a better Distances.pairwise performance.","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"warning: Warning\nThe benchmark result @btime is not generated automatically so it may be outdated. If so please file an issue/PR to update it.","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"using Distances, LazyDistances\nusing LinearAlgebra\nusing Random\nRandom.seed!(0)","category":"page"},{"location":"examples/statistics/distances/#pairwise-distance","page":"Row/Column-wise Distances","title":"pairwise distance","text":"","category":"section"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"X = rand(1:10, 100, 2);\nY = rand(1:10, 100, 3);\n\ndist = pairwise(Euclidean(), X, Y)","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"Distances.pairwise explicitly generates a large array, which does not perform well for massive scale data. PairwiseDistance mimics the pairwise operation in a lazy and optionally cached manner, so that you don't need to worry about memory allocation and the overhead due to GC. Unlike pairwise, PairwiseDistance does not assume the data to be row or column shaped; instead you need to manually specify how data pair is constructed:","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"lazy_dist = let X=X, Y=Y\n    # defines:\n    #   ĩ := σ(i)\n    #   j̃ := σ(j)\n    # where `i` and `j` are `CartesianIndex`.\n    # We don't need to operate on index `i` and `j` here, so use `identity`.\n    index_map = (\n        identity,\n        identity\n    )\n    # defines:\n    #   p = A[ĩ] := @view X[:, ĩ]\n    #   q = B[j̃] := @view Y[:, j̃]\n    # Here the input A is axes(X, 2) and is not used.\n    getindex_op = (\n        (ax, i) -> view(X, :, i),\n        (ax, j) -> view(Y, :, j)\n    )\n    # defines:\n    #   out[i, j] := d(p, q)\n    d = Euclidean()\n    PairwiseDistance(index_map, getindex_op, d, (axes(X, 2), axes(Y, 2)))\nend","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"dist == lazy_dist","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"The actual computation is not done until we retrieve the result:","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"julia> @btime getindex($dist, 2, 3);\n 1.706 ns (0 allocations: 0 bytes)\n\njulia> @btime getindex($lazy_dist, 2, 3)\n 27.086 ns (0 allocations: 0 bytes)","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"To properly benchmark the performance, let's do some simple statistics, e.g, sum, over the result.","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"X = rand(1:10, 100, 200);\nY = rand(1:10, 100, 300);\n\nfunction naive_pairwise_sum(d, X, Y)\n    return sum(pairwise(d, X, Y))\nend\n\nfunction lazy_pairwise_sum(d, X, Y)\n    getindex_op = (\n        (ax, i) -> view(X, :, i),\n        (ax, j) -> view(Y, :, j)\n    )\n    dist = PairwiseDistance(identity, getindex_op, d, map(A->axes(A, 2), (X, Y)))\n    sum(dist)\nend\n\nnaive_pairwise_sum(Euclidean(), X, Y) ≈ lazy_pairwise_sum(Euclidean(), X, Y) # true\n\n@btime naive_pairwise_sum(Euclidean(), $X, $Y); # 4.513 ms (5 allocations: 473.23 KiB)\n@btime lazy_pairwise_sum(Euclidean(), $X, $Y); # 2.265 ms (13 allocations: 352 bytes)","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"Great that we have successfully removed the memory allocation overhead and it does improve the performance. Of course, if you need to reuse the data repeatedly, you still need to collect it so that you don't suffer from the repeat calculation.","category":"page"},{"location":"examples/statistics/distances/#columnwise-distance","page":"Row/Column-wise Distances","title":"columnwise distance","text":"","category":"section"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"Strictly speaking, Distances.columnwise distance is just a simple map function. Again, in contrast, PairwiseDistance does not assume the data layout, and you have to specify it","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"X = rand(1:10, 100, 2);\nY = rand(1:10, 100, 2);\ndist = colwise(Euclidean(), X, Y)\n\nlazy_dist = let X=X, Y=Y\n    getindex_op = (\n        (ax, i) -> view(X, :, i),\n        (ax, j) -> view(Y, :, j)\n    )\n    PairwiseDistance(identity, getindex_op, Euclidean(), (axes(X, 2), axes(Y, 2)))\nend\n\ndiag(lazy_dist) == dist","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"Let's also benchmark the results:","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"X = rand(1:10, 100, 200);\nY = rand(1:10, 100, 200);\n\nfunction naive_colwise_sum(d, X, Y)\n    return sum(colwise(d, X, Y))\nend\n\nfunction lazy_colwise_sum(d, X, Y)\n    getindex_op = (\n        (ax, i) -> view(X, :, i),\n        (ax, j) -> view(Y, :, j)\n    )\n    dist = PairwiseDistance(identity, getindex_op, d, map(A->axes(A, 2), (X, Y)))\n    sum(diag(dist))\nend\n\nnaive_colwise_sum(Euclidean(), X, Y) ≈ lazy_colwise_sum(Euclidean(), X, Y) # true\n\n@btime naive_colwise_sum(Euclidean(), $X, $Y); # 6.002 μs (1 allocation: 1.77 KiB)\n@btime lazy_colwise_sum(Euclidean(), $X, $Y); # 8.166 μs (14 allocations: 2.11 KiB)","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"As you can see, this time our lazy version becomes slower than the colwise version. This is because the diag operation itself gives some overhead. PairwiseDistance itself also contributes some of the overhead.","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"","category":"page"},{"location":"examples/statistics/distances/","page":"Row/Column-wise Distances","title":"Row/Column-wise Distances","text":"This page was generated using DemoCards.jl and Literate.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = LazyDistances","category":"page"},{"location":"#LazyDistances","page":"Home","title":"LazyDistances","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"LazyDistances is provides an array abstraction on lazily operates on the data with given distances.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}