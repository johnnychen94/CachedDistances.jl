# cache strategy -- the interface
struct LocalWindowCache{N} <: AbstractCacheStrategy
    window_size::NTuple{N, Int}
end
LocalWindowCache(window_size::Tuple) = LocalWindowCache{length(window_size)}(map(i->convert(Int, i), window_size))
make_cache(T, S::LocalWindowCache{N}, axesA::NTuple{N}, axesB::NTuple{N}, args...) where N = LocalWindow{T}(axesA, axesB, S.window_size)

# cache implementation
struct LocalWindow{T, N, NA, AT<:AbstractArray{T, N}, IA, IB} <: CachedView{T, N}
    cache::AT
    axesA::IA
    axesB::IB
end

function LocalWindow{T}(axesA::NTuple{NA}, axesB::NTuple{NA}, window_size::NTuple{NA, Integer}) where {T, NA}
    window_ax = map(window_size) do w
        r = w ÷ 2
        -r:r
    end
    cache_ax = (axesA..., window_ax...)
    N = length(axesA)+length(axesB)

    FT = Union{Missing, T}
    cache = OffsetArray{FT, N}(missing, cache_ax)
    LocalWindow{FT, N, length(axesA), typeof(cache), typeof(axesA), typeof(axesB)}(cache, axesA, axesB)
end

Base.axes(A::LocalWindow) = (A.axesA..., A.axesB...)
Base.size(A::LocalWindow) = map(length, axes(A))

@inline function cache_index(::LocalWindow{T, N, NA}, I::Vararg{Int, N}) where {T, N, NA}
    i, j = I[1:NA], I[NA+1:end]
    return i, j .- i
end

@inline function is_cached(A::LocalWindow{T, N}, I::Vararg{Int, N}) where {T, N}
    # if I is outside of axes(A), it's clearly that not in cache
    i, o = cache_index(A, I...)
    return checkbounds(Bool, A.cache, i..., o...)
end

Base.@propagate_inbounds function Base.getindex(A::LocalWindow{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)

    i, o = cache_index(A, I...)
    if checkbounds(Bool, A.cache, i..., o...)
        return A.cache[i..., o...]
    else
        return missing
    end
end

Base.@propagate_inbounds function Base.setindex!(A::LocalWindow{T, N}, v, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkcache(A, I...)

    i, o = cache_index(A, I...)
    A.cache[i..., o...] = v
    return A
end

Base.@propagate_inbounds function Base.setindex!(A::LocalWindow, v, i::Int)
    # delay the boundscheck in the IndexCartesian version and not set @inbounds meta here
    setindex!(A, v, Base._to_subscript_indices(A, i)...)
end

@inline function checkcache(A::LocalWindow{T, N}, I::Vararg{Int, N}) where {T, N}
    # This gives some performance boost https://github.com/JuliaLang/julia/issues/33273
    _throw_argument_error() = throw(ArgumentError("LocalWindow do not support (re)setting the padding value. Consider making a copy of the array first."))
    _throw_bounds_error(A, i) = throw(BoundsError(A, i))

    if checkbounds(Bool, A, I...)
        i, o = cache_index(A, I...)
        checkbounds(Bool, A.cache, i..., o...) || _throw_argument_error()
    else
        _throw_bounds_error(A, I)
    end
end
