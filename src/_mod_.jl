Base.@kwdef mutable struct JchemoModel{T <: Function, K <: Base.Pairs}
    algo::T   
    fitm
    kwargs::K
end

"""
    mod_(algo::Function; kwargs...)
Build a model.
* `algo` : The algorthm (function) defining the model.
* `kwargs...`: Keyword arguments of `algo`.

## Examples
```julia
X = rand(5, 10)
y = rand(5)

model = mod_(detrend_pol)  # use the default arguments of 'detrend_pol'
#model = detrend_pol(X; degree = 2)
pnames(model)
fit!(model, X)
Xp = transf(model, X)

model = mod_(plskern; nlv = 3) 
fit!(model, X, y)
pred = predict(model, X).pred
```
"""
function mod_(algo::Function; kwargs...)
    JchemoModel(algo, nothing, kwargs)
end
