# a more efficient checkbounds implementation to minimize the overhead
_checkbounds(::Type{Bool}, A::AbstractArray, I::CartesianIndex) = _in(I, CartesianIndices(A))
function _checkbounds(::Type{Bool}, A::AbstractArray, p::CartesianIndex, q::CartesianIndex)
    _checkbounds(Bool, A, CartesianIndex(p.I..., q.I...))
end

# # short-circuiting gives better performance
@generated function _in(i::CartesianIndex{N}, r::CartesianIndices{N}) where N
    ex = :()
    for k = 1:N
        ex = :($ex; i.I[$k] in r.indices[$k] || return false)
    end
    ex = :($ex; return true)
end
# _in(i::CartesianIndex, r::CartesianIndices) = _alltsc(map(in, i.I, r.indices))
# @inline function _alltsc(t::Tuple{Bool,Vararg{Bool}})
#     first(t) || return false
#     return _alltsc(Base.tail(t))
# end
# _alltsc(::Tuple{}) = true
