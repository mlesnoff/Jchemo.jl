Base.@kwdef mutable struct Model
    fun::Function   
    fm
    kwargs    
end

model = function(fun::Function; kwargs...)
    Model(fun, nothing, kwargs)
end



