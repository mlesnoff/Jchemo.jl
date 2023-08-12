function splsqda(X, y, weights = ones(nro(X)); nlv, 
        meth = "soft", delta = 0, nvar = nco(X),
        alpha = 0, prior = "unif", scal::Bool = false)
    res = dummy(y)
    ni = tab(y).vals
    fm_pls = splskern(X, res.Y, weights; nlv = nlv, 
        meth = meth, delta = delta, nvar = nvar,
        scal = scal)
    fm_da = list(nlv)
    @inbounds for i = 1:nlv
        fm_da[i] = qda(vcol(fm_pls.T, 1:i), y, weights; 
            alpha = alpha, prior = prior)
    end
    fm = (fm_pls = fm_pls, fm_da = fm_da)
    Plslda(fm, res.lev, ni)
end


