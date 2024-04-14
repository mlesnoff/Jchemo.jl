Base.@kwdef mutable struct Model{T <: Function, K <: Base.Pairs}
    fun::T   
    fm
    kwargs::K
end

function model(fun::Function; kwargs...)
    Model(fun, nothing, kwargs)
end
