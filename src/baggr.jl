struct Baggr
    fm
    s_row  # in-bag
    s_oob
    s_col
end

""" 
    baggr(X, Y, weights = nothing ; fun, B, 
        samp_row = size(X, 1), withr = false, samp_col = size(X, 2), kwargs...)
Bagging of regression models.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `B` : Nb. of bagging repetitions.
* `fun` : Name (string) of the function computing the model to bagg.
* `samp_row` : Proportion of rows (observations) sampled in `X` at each repetition.
* `withr`: Boolean defining the type of sampling of the observations when `samp_row` < 1 
    (`withr = false` => sampling without replacement).
* `samp_col` : Proportion of columns (variables) sampled in `X` at each repetition.
* `kwargs` : Optional named arguments to pass in 'fun`.

Assume that `X` is (n, p).

If `samp_row` = 1, each repetition consists in a sampling with replacement over 
the n observations.

If `samp_row` < 1, each repetition consist in a sampling of `samp_row` * n observations.
The sampling can be without (default) or with replacement, depending on 
argument `withr`.

If `samp_col` < 1 , a proportion of `samp_col` * p variables are sampled without replacement 
at each repetition, and taken as predictors for the given repetition.

## References

Breiman, L., 1996. Bagging predictors. Mach Learn 24, 123–140. 
https://doi.org/10.1007/BF00058655

Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. 
https://doi.org/10.1023/A:1010933404324

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, 
sélection de variables et applications. PhD Thesis. Université Paris Sud - Paris XI.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245
""" 
function baggr(X, Y, weights = nothing; B, fun, 
    samp_row = 1, withr = false, samp_col = 1, kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    q = size(Y, 2)
    flearn = eval(Meta.parse(fun))    
    fm = list(B)
    n_row = Int64(round(samp_row * n))
    n_col = max(Int64(round(samp_col * p)), 1)
    s_row = fill(1, (n_row, B)) 
    srow = similar(s_row, n_row)    
    s_oob = list(B)
    s_col = similar(s_row, n_col, B) 
    scol = similar(s_row, n_col)
    w = similar(X, n_row)
    zncol = collect(1:n_col) 
    zX = similar(X, n_row, n_col)
    zY = similar(Y, n_row, q)
    @inbounds for i = 1:B
        samp_row == 1 ? withr = true : nothing
        srow .= sample(1:n, n_row; replace = withr)
        s_oob[i] = findall(in(srow).(1:n) .== 0)
        samp_col == 1 ? scol .= zncol : scol .= sample(1:p, n_col; replace = false)
        zX .= X[srow, scol]
        zY .= Y[srow, :]
        if(isnothing(weights))
            fm[i] = flearn(zX, zY; kwargs...)
        else
            w .= weights[srow]
            w .= w / sum(w)
            fm[i] = flearn(zX, zY, w; kwargs...)
        end
        s_row[:, i] .= srow    
        s_col[:, i] .= scol
    end
    Baggr(fm, s_row, s_oob, s_col)
end

function predict(object::Baggr, X)
    B = length(object.fm)
    scol = vcol(object.s_col, 1)
    # @view is not accepted by XGBoost.predict
    # @view(X[:, scol])
    acc = predict(object.fm[1], X[:, scol]).pred
    @inbounds for i = 2:B
        scol = vcol(object.s_col, i)
        acc .+= predict(object.fm[i], X[:, scol]).pred
    end
    pred = acc ./ B
    (pred = pred,)
end




