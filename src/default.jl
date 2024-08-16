"""
    default(fun::Function)
Display the keyword arguments (with their default values) of a function
* `fun` : The name of the functions.

## Examples
```julia
using Jchemo

default(svmr)
```
"""
function default(fun::Function)

    in((detrend,)).(fun) ? dump(Jchemo.ParDetrend()) : nothing
    in((asls,)).(fun) ? dump(Jchemo.ParAsls()) : nothing
    in((airpls,)).(fun) ? dump(Jchemo.ParAirpls()) : nothing
    in((arpls,)).(fun) ? dump(Jchemo.ParAirpls()) : nothing
    in((fdif,)).(fun) ? dump(Jchemo.ParFdif()) : nothing
    in((interpl,)).(fun) ? dump(Jchemo.ParInterpl()) : nothing
    in((mavg,)).(fun) ? dump(Jchemo.ParMavg()) : nothing
    in((savgol,)).(fun) ? dump(Jchemo.ParSavgol()) : nothing
    in((snv,)).(fun) ? dump(Jchemo.ParSnv()) : nothing
    in((rmgap,)).(fun) ? dump(Jchemo.ParRmgap()) : nothing

    ##

    in((nipals, nipalsmiss)).(fun) ? dump(Jchemo.ParNipals()) : nothing
    in((snipals, snipalsh, snipalsmix)).(fun) ? dump(Jchemo.ParSnipals()) : nothing

    in((pcasvd, pcaeigen, pcaeigenk, pcasph)).(fun) ? dump(Jchemo.ParPca()) : nothing
    in((pcanipals, pcanipalsmiss)).(fun) ? dump(Jchemo.ParPcanipals()) : nothing
    in((pcapp,)).(fun) ? dump(Jchemo.ParPcapp()) : nothing
    in((pcaout,)).(fun) ? dump(Jchemo.ParOut()) : nothing

    in((rp,)).(fun) ? dump(Jchemo.ParRp()) : nothing

    in((spca,)).(fun) ? dump(Jchemo.ParSpca()) : nothing

    in((kpca,)).(fun) ? dump(Jchemo.ParKpca()) : nothing
    in((umap,)).(fun) ? dump(Jchemo.ParUmap()) : nothing

    in((fda, fdasvd)).(fun) ? dump(Jchemo.ParFda()) : nothing

    in((blockscal,)).(fun) ? dump(Jchemo.ParBlock()) : nothing
    in((plscan, plstuck)).(fun) ? dump(Jchemo.ParPls2bl()) : nothing
    in((rasvd,)).(fun) ? dump(Jchemo.ParRasvd()) : nothing
    in((rrr,)).(fun) ? dump(Jchemo.ParRrr()) : nothing

    in((mbpca, comdim, )).(fun) ? dump(Jchemo.ParMbpca()) : nothing
    in((cca,)).(fun) ? dump(Jchemo.ParCca()) : nothing
    in((ccawold,)).(fun) ? dump(Jchemo.ParCcawold()) : nothing

    ##

    in((mlr, mlrpinv, mlrvec)).(fun) ? dump(Jchemo.ParMlr()) : nothing

    in((rr, rrchol)).(fun) ? dump(Jchemo.ParRr()) : nothing

    in((pcr,)).(fun) ? dump(Jchemo.ParPcr()) : nothing

    in((plskern, plsnipals, plswold, plsrosa, plssimp, plsravg, plsravg_unif)).(fun) ? dump(Jchemo.ParPlsr()) : nothing
    in((cglsr, aicplsr)).(fun) ? dump(Jchemo.ParCglsr()) : nothing
    in((plsrout,)).(fun) ? dump(Jchemo.ParPlsrout()) : nothing

    in((splskern,)).(fun) ? dump(Jchemo.ParSplsr()) : nothing

    in((krr,)).(fun) ? dump(Jchemo.ParKrr()) : nothing
    in((kplsr, dkplsr)).(fun) ? dump(Jchemo.ParKplsr()) : nothing

    in((svmr, svmda)).(fun) ? dump(Jchemo.ParSvm()) : nothing

    in((treer, treeda)).(fun) ? dump(Jchemo.ParTree()) : nothing
    in((rfr, rfda)).(fun) ? dump(Jchemo.ParRf()) : nothing

    in((knnr, lwmlr)).(fun) ? dump(Jchemo.ParKnn()) : nothing
    in((lwplsr, lwplsravg)).(fun) ? dump(Jchemo.ParLwplsr()) : nothing

    in((mbplsr, mbplswest)).(fun) ? dump(Jchemo.ParMbplsr()) : nothing
    in((rosaplsr, soplsr)).(fun) ? dump(Jchemo.ParSoplsr()) : nothing

    ##

    in((dmnorm, dmnormlog)).(fun) ? dump(Jchemo.ParDmnorm()) : nothing
    in((dmkern,)).(fun) ? dump(Jchemo.ParDmkern()) : nothing

    in((lda,)).(fun) ? dump(Jchemo.ParLda()) : nothing
    in((qda,)).(fun) ? dump(Jchemo.ParQda()) : nothing
    in((rda,)).(fun) ? dump(Jchemo.ParRda()) : nothing
    in((kdeda,)).(fun) ? dump(Jchemo.ParKdeda()) : nothing
    
    in((mlrda,)).(fun) ? dump(Jchemo.ParMlrda()) : nothing
    in((rrda,)).(fun) ? dump(Jchemo.ParRrda()) : nothing
    in((plsrda, plslda)).(fun) ? dump(Jchemo.ParPlsda()) : nothing
    in((plsqda,)).(fun) ? dump(Jchemo.ParPlsqda()) : nothing
    in((plskdeda,)).(fun) ? dump(Jchemo.ParPlskdeda()) : nothing

    in((splsrda, splslda)).(fun) ? dump(Jchemo.ParSplsda()) : nothing
    in((splsqda,)).(fun) ? dump(Jchemo.ParSplsqda()) : nothing
    in((splskdeda,)).(fun) ? dump(Jchemo.ParSplskdeda()) : nothing

    in((krrda,)).(fun) ? dump(Jchemo.ParKrrda()) : nothing
    in((kplsrda, kplslda, dkplsrda, dkplslda)).(fun) ? dump(Jchemo.ParKplsda()) : nothing
    in((kplsqda, dkplsqda)).(fun) ? dump(Jchemo.ParKplsqda()) : nothing
    in((kplskdeda, dkplskdeda)).(fun) ? dump(Jchemo.ParKplskdeda()) : nothing

    in((knnda, lwmlrda)).(fun) ? dump(Jchemo.ParKnn()) : nothing
    in((lwplsrda, lwplslda)).(fun) ? dump(Jchemo.ParLwplsda()) : nothing
    in((lwplsqda,)).(fun) ? dump(Jchemo.ParLwplsqda()) : nothing

    in((mbplsrda, mbplslda)).(fun) ? dump(Jchemo.ParMbplsda()) : nothing
    in((mbplsqda,)).(fun) ? dump(Jchemo.ParMbplsqda()) : nothing
    in((mbplskdeda,)).(fun) ? dump(Jchemo.ParMbplskdeda()) : nothing

    ##

    in((outeucl, outstah,)).(fun) ? dump(Jchemo.ParOut()) : nothing
    in((occsd, occod, occsdod)).(fun) ? dump(Jchemo.ParOcc()) : nothing
    in((occstah,)).(fun) ? dump(Jchemo.ParOut()) : nothing

end

