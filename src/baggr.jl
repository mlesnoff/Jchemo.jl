struct Baggr
    fm
    s_obs
    s_var
end

""" 
    baggr(X, Y, weights = ones(size(X, 1)) ; fun, B, mtry = size(X, 2), kwargs...)
Bagging regression models.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `fun` : Name (string) of the function computing the model to bagg.
* `B` : Nb. of bootstrap repetitions.
* `mtry` : Nb. variables (columns) to sample in `X`.
* `kwargs` : Named arguments to pass in 'fun`.
""" 
function baggr(X, Y, weights = ones(size(X, 1)) ; fun, B, mtry = size(X, 2), kwargs...)
    flearn = eval(Meta.parse(fun))    
    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)
    listfm = list(B)
    zX = similar(X, n, mtry)
    zY = similar(Y, n, q)
    s_obs = sample(1:n, (n, B); replace = true)
    s_var = similar(s_obs, mtry, B)
    @inbounds for j = 1:B
        sobs = vcol(s_obs, j)
        svar = sample(1:p, mtry; replace = false)
        zX .= X[sobs, svar]
        zY .= Y[sobs, :]
        listfm[j] = flearn(zX, zY, weights; kwargs...)
        s_var[:, j] .= svar
    end
    Baggr(listfm, s_obs, s_var)
end

function predict(object::Baggr, X)
    B = length(object.fm)
    svar = vcol(object.s_var, 1)
    acc = predict(object.fm[1], @view(X[:, svar])).pred
    @inbounds for j = 2:B
        svar = vcol(object.s_var, j)
        acc .+= predict(object.fm[j], @view(X[:, svar])).pred
    end
    pred = acc ./ B
    (pred = pred,)
end




