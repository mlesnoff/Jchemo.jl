function mtestsys(Y::DataFrame, id = 1:nro(Y); ntest)
    nam = names(Y)
    p = length(nam)
    idtrain = list(p, Vector)  
    idtest = list(p, Vector)
    length(ntest) == 1 ? ntest = repeat([ntest], p) : nothing
    for i = 1:p
        y = Y[:, nam[i]]
        s_all = findall(ismissing.(y) .== 0)      
        res = sampsys(y[s_all]; k = ntest[i])
        idtrain[i] = sort(id[s_all[res.test]])
        idtest[i] = sort(id[s_all[res.train]])      
    end
    (train = idtrain, test = idtest, nam)
end

