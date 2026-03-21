using Distributions

function waldtest(b, L, varb, h0 = nothing; dfdenom = nothing)
    dfnum = nro(L)
    isnothing(h0) ? h = L * b : h = L * b - h0 
    varLb = L * varb * L' 
    z =  h' * inv(varLb) * h
    if isnothing(dfdenom)
        d = Distributions.Chisq(dfnum)
    else
        d = Distributions.FDist(dfnum, dfdenom)
        z = z / dfnum
    end 
    pval = Distributions.ccdf(d, z)
    (z = z, pval, dfnum, dfdenom)
end 
