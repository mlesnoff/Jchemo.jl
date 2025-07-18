module Jchemo  # Start-Module

using DataInterpolations  # 1D interpolations (interpl) 
using DecisionTree
using Distributions
using DataFrames
using Distances
using ImageFiltering      # convolutions in preprocessing (mavg, savgol), alternative = DSP.jl
using LIBSVM 
using LinearAlgebra
using Loess
using Makie
using NearestNeighbors
using Random
using SparseArrays 
using Statistics
using StatsBase           # countmap, ecdf, sample etc.
using UMAP

## The order below is required
include("_struct_param.jl")
include("_struct_fun.jl")      
include("_model_work.jl")
include("_pip.jl")
## End
include("defaults.jl")

######---- Misc

include("_util.jl")
include("_util_stat.jl")
include("_util_colwise.jl")
include("_util_rowwise.jl")
include("_util_center_scale.jl")
include("fweight.jl")
include("angles.jl")
include("colmedspa.jl")
include("ellipse.jl")
include("matW.jl")
include("nipals.jl")
include("nipalsmiss.jl")
include("simpp.jl")
include("snipals_shen.jl")
include("snipals_mix.jl")
include("snipals_post.jl")

######---- Preprocessing

include("preprocessing.jl") 
include("detrend_asls.jl") 
include("detrend_airpls.jl") 
include("detrend_arpls.jl") 
include("center_scale.jl") 
include("rmgap.jl")

######---- Graphics

include("plotsp.jl")
include("plotxy.jl")
include("plotxyz.jl")
include("plotlv.jl")
include("plotgrid.jl")
include("plotconf.jl")

######---- Distributions

include("dmnorm.jl")
include("dmnormlog.jl")
include("dmkern.jl")

######---- Exploratory

include("fda.jl")  # Here since ::Fda called in pcasvd
include("fdasvd.jl")     
include("pcasvd.jl")
include("pcaeigen.jl")
include("pcaeigenk.jl")
include("pcanipals.jl")
include("pcanipalsmiss.jl")
include("pcasph.jl") 
include("pcapp.jl") 
include("pcaout.jl") 
include("kpca.jl")
include("covsel.jl")
include("rpmat.jl")
include("rp.jl")
include("umap.jl")

## Sparse
include("spca.jl")

## Multiblock 
include("_util_mb.jl")
include("cca.jl")
include("ccawold.jl")
include("plscan.jl")
include("plstuck.jl")
include("rasvd.jl")
include("mbpca.jl")
include("comdim.jl")

######---- Regression 

include("aov1.jl")
include("mlr.jl")
include("rr.jl")
include("rrchol.jl")
include("pcr.jl")
include("rrr.jl") 
include("plskern.jl")
include("plsnipals.jl")
include("plswold.jl") 
include("plsrosa.jl")
include("plssimp.jl")
include("cglsr.jl")
include("plsrout.jl")
include("plsravg.jl")
include("plsravg_unif.jl")
include("krr.jl")
include("kplsr.jl")
include("dkplsr.jl")

include("dfplsr_cg.jl")
include("aicplsr.jl")
include("vip.jl") 

include("xfit.jl")
include("xresid.jl")

## Sparse
include("splsr.jl")
include("spcr.jl")

## Multiblock
include("mbplsr.jl") 
include("mbplswest.jl")
include("rosaplsr.jl") 
include("soplsr.jl") 

## Local
include("locw.jl")
include("locwlv.jl")
include("knnr.jl")
include("lwmlr.jl")
include("lwplsr.jl")
include("lwplsravg.jl")
include("loessr.jl")

## Validation
include("mpar.jl")
include("scores.jl")
include("conf.jl")
include("segmkf.jl")
include("segmts.jl")
include("gridscore.jl")
include("gridscore_pip.jl")
include("gridscore_br.jl")
include("gridscore_lv.jl")
include("gridscore_lb.jl")
include("gridcv.jl")
include("gridcv_br.jl")
include("gridcv_lv.jl")
include("gridcv_lb.jl")
include("selwold.jl")

## Variable importance (direct methods) 
include("isel.jl")
include("viperm.jl")

## Svm, Trees
include("svmr.jl")
include("treer.jl")
include("rfr.jl")

######---- Discrimination 

include("lda.jl")
include("qda.jl")
include("rda.jl")
include("kdeda.jl")
include("mlrda.jl")
include("rrda.jl")
include("plsrda.jl") 
include("plslda.jl")
include("plsqda.jl")
include("plskdeda.jl")
include("krrda.jl")
include("kplsrda.jl")
include("kplslda.jl")
include("kplsqda.jl")
include("kplskdeda.jl")
include("dkplsrda.jl")
include("dkplslda.jl")
include("dkplsqda.jl")
include("dkplskdeda.jl")

## Sparse
include("splsrda.jl")
include("splslda.jl")
include("splsqda.jl")
include("splskdeda.jl")

## Multiblock
include("mbplsrda.jl") 
include("mbplslda.jl") 
include("mbplsqda.jl") 
include("mbplskdeda.jl") 

## One-class
include("outstah.jl")
include("outeucl.jl")
include("outsd.jl")
include("outod.jl")
include("outsdod.jl")
include("outknn.jl")
include("outlknn.jl")
include("occstah.jl")
include("occsd.jl")
include("occod.jl") 
include("occsdod.jl")
include("occknn.jl")
include("occlknn.jl")

## Local
include("lwmlrda.jl")
include("lwplsrda.jl")
include("lwplslda.jl")
include("lwplsqda.jl")
include("knnda.jl")

## Svm, Trees
include("svmda.jl")
include("treeda.jl")
include("rfda.jl")

######---- Calibration transfer

include("calds.jl")
include("calpds.jl")
include("difmean.jl")
include("eposvd.jl")

######---- Sampling

include("sampks.jl")
include("sampdp.jl")
include("sampwsp.jl")
include("samprand.jl")
include("sampsys.jl")
include("sampcla.jl")
include("sampdf.jl")

include("distances.jl")
include("getknn.jl")
include("wdis.jl") 
include("wtal.jl") 
include("winvs.jl")
include("kernels.jl")

export 
    model,
    modelx, modelxy, 
    fit!,
    transf!,
    pip,
    ######---- Utilities
    @head, @pmod, @names, @pars, @plist, @type,
    ##
    aggmean, aggstat, 
    aggsumv,  
    sumv, meanv, stdv, varv, madv, iqrv, normv, 
    colsum, colmean, colnorm, colstd, colvar, colmed, colmad, 
    colsumskip, colmeanskip, colstdskip, colvarskip,
    convertdf,
    covv, covm, 
    corv, corm,
    cosv, cosm, 
    dummy,
    dupl, findmiss,
    ensure_df, ensure_mat,
    fblockscal, fblockscal!,
    fcenter, fcenter!, 
    fcscale, fcscale!, 
    fweight, fweight!,
    recod_catbyind,
    findmax_cla, 
    frob, frob2, 
    fscale, fscale!,
    wdis, wtal, 
    list, 
    matB, matW, 
    mblock,
    mlev,
    mweight, mweightcla,
    nipals,
    nipalsmiss,
    nro, nco, 
    out,
    parsemiss,
    pval,
    recod_catbyint, recod_numbyint, 
    recovkw,
    recod_catbylev, recod_indbylev, 
    recod_catbydict, 
    recod_miss, 
    rmcol, rmrow, 
    finduniq,
    rowmean, rownorm, rowstd, rowsum, rowvar,
    rowmeanskip, rowstdskip, rowsumskip, rowvarskip,
    simpphub, simppsph,   
    snipals_shen,
    thresh_soft, thresh_hard, 
    softmax,
    sourcedir,
    summ,
    tab, tab, tabdupl,
    vcatdf,
    vcol, vrow,
    ######---- Distributions
    dmnorm, dmnorm!,
    dmnormlog, dmnormlog!,
    dmkern,
    ## Pre-processing
    detrend_pol, detrend_lo,  
    detrend_asls, detrend_airpls, detrend_arpls,
    fdif,
    interpl, 
    center, scale, cscale,
    blockscal,
    #cubic_spline,
    mavg, 
    rmgap,
    savgk, savgol,
    snorm,
    snv, 
    ######---- Calibration ransfer
    calds, calpds,
    difmean,
    eposvd,
    ######---- Exploratory
    pcasvd, pcasvd!, 
    pcaeigen, pcaeigen!, 
    pcaeigenk, pcaeigenk!,
    pcanipals, pcanipals!,
    pcanipalsmiss, pcanipalsmiss!,
    pcasph, pcasph!,
    pcapp, pcapp!,
    pcaout, pcaout!,
    spca, spca!,
    kpca,
    covsel,
    rpmatgauss, rpmatli, rp, rp!,
    umap,
    ## Multiblock
    rd, rv, 
    mbconcat, fconcat,
    cca, cca!,
    ccawold, ccawold!,
    plscan, plscan!,
    plstuck, plstuck!,
    rasvd, rasvd!,
    mbpca, mbpca!,
    comdim, comdim!, 
    ######---- Regression
    aov1,
    mlr, mlr!, mlrchol, mlrchol!, 
    mlrpinv, mlrpinv!, mlrpinvn, mlrpinvn!,
    mlrvec, mlrvec!,
    plskern, plskern!, 
    plsnipals, plsnipals!, 
    plsrosa, plsrosa!, 
    plssimp, plssimp!,
    plswold, plswold!,
    cglsr, cglsr!,
    plsrout, plsrout!,
    pcr,
    rr, rr!, rrchol, rrchol!,
    rrr, rrr!,   
    krr, krr!, kplsr, kplsr!, 
    dkplsr, dkplsr!,
    plsravg, plsravg!,
    dfplsr_cg, aicplsr,
    svmr,
    treer, rfr, 
    ## Sparse 
    splsr, splsr!, 
    spcr, spcr!,
    ## Multi-block
    mbplsr, mbplsr!,
    mbplswest, mbplswest!,
    rosaplsr, rosaplsr!,
    soplsr,
    ## Variable selection/importance (direct methods) 
    isel!,
    vip, 
    viperm,
    ## Utils
    xfit, xfit!, xresid, xresid!,
    ## Local
    locw, locwlv,
    knnr,
    lwmlr,
    lwplsr, lwplsravg,
    loessr,
    ######---- Discrimination
    fda, fda!, fdasvd, fdasvd!,
    mlrda,
    rrda, krrda,
    lda, qda, kdeda,
    rda,
    plsrda,
    plslda, plsqda, plskdeda,
    kplsrda, 
    kplslda, kplsqda, kplskdeda, 
    dkplsrda,
    dkplslda, dkplsqda, dkplskdeda, 
    svmda, 
    treeda, rfda,
    ## One-class
    outstah, outeucl,
    outsd, outod, outsdod,
    outknn, outlknn,
    occstah,
    occsd, occod, occsdod,
    occknn, occlknn,
    ## Sparse 
    splsrda,
    splslda, splsqda, splskdeda,
    ## Local 
    lwmlrda,
    lwplsrda, 
    lwplslda, lwplsqda,
    knnda,
    ## Multiblock
    mbplsrda, 
    mbplslda, mbplsqda, mbplskdeda,
    ## Auxiliary
    transf, coef, predict,
    transfbl, 
    ## Validation
    residreg, residcla, 
    ssr, msep, rmsep, rmsepstand, rrmsep, mae,
    bias, sep, cor2, r2, rpd, rpdr, mse, 
    errp, merrp,
    mpar,
    gridscore, 
    gridscore_br, gridscore_lv, gridscore_lb,
    segmts, segmkf,
    gridcv, 
    gridcv_br, gridcv_lv, gridcv_lb, 
    selwold,
    conf, 
    ######---- Sampling
    sampks, sampdp, sampwsp, samprand, sampsys, sampcla, 
    sampdf,
    ######---- Distances
    getknn, wdis, wtal, winvs, winvs!,
    euclsq, mahsq, mahsqchol,
    krbf, kpol,
    ######---- Graphics
    plotsp,
    plotxy, plotxyz,
    plotlv,
    plotgrid, 
    plotconf
    ## Not exported since surcharge:
    ## - summary => Base.summary

end # End-Module




