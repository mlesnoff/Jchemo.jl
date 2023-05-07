function plskernda(X, y, weights = ones(nro(X)); nlv, 
        prior = "unif", h = nothing, a = 1, scal = false)
    res = dummy(y)
    ni = tab(y).vals
    fm_pls = plskern(X, res.Y, weights; nlv = nlv, scal = scal)
    fm_da = list(nlv)
    @inbounds for i = 1:nlv
        fm_da[i] = kernda(vcol(fm_pls.T, 1:i), y; prior = prior,
            h = h, a = a)
    end
    fm = (fm_pls = fm_pls, fm_da = fm_da)
    Plslda(fm, res.lev, ni)
end


