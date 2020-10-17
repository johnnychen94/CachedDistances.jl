struct LocalWindowCache{N} <: AbstractCacheStrategy
    window_size::NTuple{N, Int}
end
LocalWindowCache(window_size::Tuple) = LocalWindowCache{length(window_size)}(map(i->convert(Int, i), window_size))
make_cache(T, S::LocalWindowCache{N}, axesA::NTuple{N}, axesB::NTuple{N}, args...) where N = LocalWindowCacheArray{T}(axesA, axesB, S.window_size)

struct LocalWindowCacheArray{T, N, NA, AT<:AbstractArray{T, N}, IA, IB, IC} <: CachedView{T, N}
    cache::AT
    axesA::IA
    axesB::IB
    R::IC
end

function LocalWindowCacheArray{T}(axesA::NTuple{NA}, axesB::NTuple{NA}, window_size::NTuple{NA, Integer}) where {T, NA}
    window_ax = map(window_size) do w
        r = w ÷ 2
        -r:r
    end
    cache_ax = (axesA..., window_ax...)
    N = length(axesA)+length(axesB)

    FT = Union{Missing, T}
    cache = OffsetArray{FT, N}(missing, cache_ax)

    # https://github.com/JuliaArrays/OffsetArrays.jl/issues/166
    # For OffsetArray, `axes(cache)` can be expensive to be done in each `getindex` call.
    # Thus we cache the result and use it to accelerate checkbounds.
    R = CartesianIndices(cache)
    LocalWindowCacheArray{FT, N, length(axesA), typeof(cache), typeof(axesA), typeof(axesB), typeof(R)}(cache, axesA, axesB, R)
end

Base.axes(A::LocalWindowCacheArray) = (A.axesA..., A.axesB...)
Base.size(A::LocalWindowCacheArray) = map(length, axes(A))
