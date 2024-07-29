###### Fit

function fit!(mod::Jchemo.Model, X)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X; kwargs...)
    return
end  
function fit!(mod::Jchemo.Model, X, weights::Weight)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, weights; kwargs...)
    return
end

function fit!(mod::Jchemo.Model, X, Y)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, Y; kwargs...)
    return
end  
function fit!(mod::Jchemo.Model, X, Y, weights::Weight)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, Y, weights; kwargs...)
    return
end  

###### Transf

function transf(mod::Jchemo.Model, X; nlv = nothing)
    isnothing(nlv) ? transf(mod.fm, X) : transf(mod.fm, X; nlv)
end
function transf(mod::Jchemo.Model, X, Y; nlv = nothing)
    isnothing(nlv) ? transf(mod.fm, X, Y) : transf(mod.fm, X, Y; nlv)
end

function transfbl(mod::Jchemo.Model, X; nlv = nothing)
    isnothing(nlv) ? transfbl(mod.fm, X) : transfbl(mod.fm, X; nlv)
end
function transfbl(mod::Jchemo.Model, X, Y; nlv = nothing)
    isnothing(nlv) ? transfbl(mod.fm, X, Y) : transfbl(mod.fm, X, Y; nlv)
end

###### Predict 

function predict(mod::Jchemo.Model, X; nlv = nothing, lb = nothing)
    if isnothing(nlv) && isnothing(lb)
        predict(mod.fm, X)
    elseif !isnothing(nlv) 
        predict(mod.fm, X; nlv)
    elseif !isnothing(lb) 
        predict(mod.fm, X; lb)
    end
end  

###### Coef 

function coef(mod::Jchemo.Model; nlv = nothing, lb = nothing)
    if isnothing(nlv) && isnothing(lb)
        coef(mod.fm)
    elseif !isnothing(nlv) 
        coef(mod.fm; nlv)
    elseif !isnothing(lb) 
        coef(mod.fm; lb)
    end
end

###### Summary 

function Base.summary(mod::Jchemo.Model)
    Base.summary(mod.fm)
end
function Base.summary(mod::Jchemo.Model, X)
    Base.summary(mod.fm, X)
end
function Base.summary(mod::Jchemo.Model, X, Y)
    Base.summary(mod.fm, X, Y)
end

