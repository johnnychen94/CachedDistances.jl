# cache strategy -- the interface
struct LocalWindowCache{N} <: AbstractCacheStrategy
    window_size::NTuple{N, Int}
end
LocalWindowCache(window_size::Tuple) = LocalWindowCache{length(window_size)}(map(i->convert(Int, i), window_size))
make_cache(T, S::LocalWindowCache{N}, axesA::NTuple{N}, axesB::NTuple{N}, args...) where N = LocalWindowCacheArray{T}(axesA, axesB, S.window_size)

# cache implementation
struct LocalWindowCacheArray{T, N, NA, AT<:AbstractArray{T, N}, IA, IB} <: CachedView{T, N}
    cache::AT
    axesA::IA
    axesB::IB
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
    LocalWindowCacheArray{FT, N, length(axesA), typeof(cache), typeof(axesA), typeof(axesB)}(cache, axesA, axesB)
end

Base.axes(A::LocalWindowCacheArray) = (A.axesA..., A.axesB...)
Base.size(A::LocalWindowCacheArray) = map(length, axes(A))
