function plsr_stack(X, y, weights = ones(size(X, 1)); nlv,
        K = 5, rep = 10, scal = false)
    plsr_stack!(copy(ensure_mat(X)), copy(ensure_mat(y)), weights; nlv = nlv, 
        K = K, rep = rep, scal = scal)
end

function plsr_stack!(X::Matrix, y::Matrix, weights = ones(size(X, 1)); nlv, 
        K = 5, rep = 10, scal = false)
    n, p = size(X)
    weights = mweight(weights)
    nlv = eval(Meta.parse(nlv))
    nlv = (min(minimum(nlv), n, p):min(maximum(nlv), n, p))
    nlvmax = maximum(nlv)    
    segm = segmkf(n, K; rep = rep)
    ystack = []
    Xstack = []
    ## Replications and segments are concatenated vertically
    ## ==> Xstack (y predictions), ystack (y observed values)
    ## Corresponding observation weights also ==> weights_stack 
    k = 1
    for i = 1:nro(segm)
        for j = 1:nro(segm[1])
            s = segm[i][j]
            Xcal = rmrow(X, s)
            ycal = rmrow(y, s)
            wcal = rmrow(weights, s)
            Xval = vrow(X, s)
            yval = vrow(y, s)
            wval = vrow(weights, s)
            zfm = plskern(Xcal, ycal, wcal; nlv =  nlvmax, scal = scal) ;
            pred = Jchemo.predict(zfm, Xval; nlv = 0:nlvmax).pred
            pred = reduce(hcat, pred)
            if k == 1
                Xstack = copy(pred) 
                ystack = copy(yval)
                weights_stack = copy(wval)
            else
                Xstack = vcat(Xstack, pred) 
                ystack = vcat(ystack, yval)
                weights_stack = vcat(weights_stack, wval)
            end
            k = k + 1
        end
    end
    ## Computation of the stacking weights (w)
    ## Linear model without intercept
    weights_stack = mweight(weights_stack)
    Xstack = vcol(Xstack, nlv .+ 1)
    XtD = Xstack' * Diagonal(weights_stack)
    w = vec(cholesky!(Hermitian(XtD * Xstack)) \ (XtD * ystack))
    ## Model that will be "averaged" (w)
    fm = plskern!(X, y, weights; nlv = nlvmax, scal = scal)
    PlsrStack(fm, nlv, w, Xstack, ystack, weights_stack)
end

