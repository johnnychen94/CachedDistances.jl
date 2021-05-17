struct NullCache <:AbstractCacheStrategy end
function make_cache(T, ::NullCache, ax_list)
    axes = reduce(ax_list; init=()) do x, y
        (x..., y...)
    end
    NullCacheArray{T, length(axes), typeof(axes)}(axes)
end

# cache implementation details
struct NullCacheArray{T, N, AT} <: CachedView{T, N}
    axes::AT
end
Base.axes(C::NullCacheArray) = C.axes
Base.size(C::NullCacheArray) = map(length, axes(C))
