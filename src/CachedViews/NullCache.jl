struct NullCache <:AbstractCacheStrategy end

struct NullCacheArray{T, N, AT} <: CachedView{T, N}
    axes::AT
end
function make_cache(T, ::NullCache, axesA, axesB, args...)
    axes = (axesA..., axesB...)
    NullCacheArray{T, length(axes), typeof(axes)}(axes)
end
is_cached(::NullCacheArray, ::Int...) = false
getindex(::NullCacheArray{T, N}, ::Vararg{Int, N}) where {T, N} = missing
setindex!(::NullCacheArray{T, N}, v, ::Vararg{Int, N}) where {T, N} = v
Base.axes(C::NullCacheArray) = C.axes
Base.size(C::NullCacheArray) = map(length, axes(C))
