struct Baggr
    fm
    s_obs
    s_var
end

""" 
    baggr(X, Y, weights = nothing ; fun, B, mtry = size(X, 2), kwargs...)
Bagging for regression models.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `fun` : Name (string) of the function computing the model to bagg.
* `B` : Nb. of bootstrap repetitions.
* `mtry` : Nb. variables (columns) sub-sampled in `X`.
* `kwargs` : Named arguments to pass in 'fun`.
""" 
function baggr(X, Y, weights = nothing ; fun, B, mtry = size(X, 2), kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)
    learn = eval(Meta.parse(fun))    
    fm = list(B)
    mtry = min(mtry, p)
    zX = similar(X, n, mtry)
    zY = similar(Y, n, q)
    s_obs = sample(1:n, (n, B); replace = true)
    s_var = similar(s_obs, mtry, B)
    @inbounds for i = 1:B
        sobs = vcol(s_obs, i)
        svar = sample(1:p, mtry; replace = false)
        zX .= X[sobs, svar]
        zY .= Y[sobs, :]
        if(isnothing(weights))
            fm[i] = learn(zX, zY; kwargs...)
        else
            w = weights[sobs]
            zweights = w / sum(w)
            fm[i] = learn(zX, zY, zweights; kwargs...)
        end
        s_var[:, i] .= svar
    end
    Baggr(fm, s_obs, s_var)
end

function predict(object::Baggr, X)
    B = length(object.fm)
    svar = vcol(object.s_var, 1)
    acc = predict(object.fm[1], @view(X[:, svar])).pred
    @inbounds for i = 2:B
        svar = vcol(object.s_var, i)
        acc .+= predict(object.fm[i], @view(X[:, svar])).pred
    end
    pred = acc ./ B
    (pred = pred,)
end




