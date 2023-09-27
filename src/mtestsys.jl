function mtestsys(Y::DataFrame, id = 1:nro(Y); ntest)
    nam = names(Y)
    p = length(nam)
    idtrain = list(p, Vector)  
    idtest = list(p, Vector)
    length(ntest) == 1 ? ntest = repeat([ntest], p) : nothing
    for i = 1:p
        y = Y[:, nam[i]]
        s_all = findall(ismissing.(y) .== 0)      
        ntot = length(s_all)
        ntrain = ntot - ntest[i]
        res = sampsys(y[s_all]; k = ntrain)
        idtrain[i] = sort(id[s_all[res.train]])
        idtest[i] = sort(id[s_all[res.test]])      
    end
    (train = idtrain, test = idtest, nam)
end

