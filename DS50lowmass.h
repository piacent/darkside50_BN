// ***************************************************************
// This file was created using the bat-project script.
// bat-project is part of Bayesian Analysis Toolkit (BAT).
// BAT can be downloaded from http://mpp.mpg.de/bat
//
// Author: Stefano Piacentini
// Date: 03/02/2023
// ***************************************************************

#ifndef __BAT__DS50LOWMASS__H
#define __BAT__DS50LOWMASS__H

#include <BAT/BCModel.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <fstream>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_linalg.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "cnpy.h"
#include "Math/ProbFunc.h"

// This is a DS50lowmass header file.
// Model source code is located in file ./DS50lowmass.cxx

// ---------------------------------------------------------
class  DS50lowmass : public BCModel
{

public:

    // Constructor
     DS50lowmass(const std::string& name, int Nmin, int Nmax, int nth, std::string mass, bool idebug,
                 bool statonly, bool rebin, bool iexpected);

    // Destructor
    ~ DS50lowmass();

    // LogLikelihood
    double LogLikelihood(const std::vector<double>& pars);

    // LogAprioriProbability
    double LogAPrioriProbability(const std::vector<double> & pars);
    
    // Physics
    double QyER(double E, double g2, double Cbox, double rho, double p0, double p1);
    double QyNR(double E, double Cbox, double fB, double g2);


    // Functions for matrices manipulation in LogPrior
    void print_mat_contents(gsl_matrix *matrix);
    gsl_matrix *getFromDoubleArray(const double original_matrix[5][5]);
    gsl_matrix *getFromDoubleArray(const double original_matrix[2][2]);
    gsl_matrix *invert_a_matrix(gsl_matrix *matrix);
    gsl_matrix *invert_a_matrix2(gsl_matrix *matrix);
    double matrix_det(gsl_matrix *matrix);
    double matrix_det2(gsl_matrix *matrix);
    void getMatrixFromGsl(gsl_matrix *matrix, double mat[5][5]);
    void getMatrixFromGsl(gsl_matrix *matrix, double mat[2][2]);
    
    // Function for initializing and loading the fit inputs
    std::vector<std::string> getValueFromScreening(const std::string, const std::string);
    void initScreening();
    void getar39_theo();
    void getkr85_theo();
    void getar39_Q();
    void getkr85_Q();
    void getpmt_theo();
    void getcryo_theo();
    void getMigdal(std::string mass);
    void getMigdal_SM();  
    void getWIMP_theo(std::string mass);
    void getWIMP_SM();
    void getx_theo();
    void getdata();
    void getar39_SM();
    void getkr85_SM();
    void getpmt_SM();
    void getcryo_SM();
    void init_cuda();
    
    void getPeakEff();
    void getNormNR();
    
private:
    
    int Nmin, Nmax;
    double FANO = 0.11;

    // Name of the single bkg components
    std::string norms_names[20] ={"ar39n", "kr85n",
                                "cryostats_co60",
                                "cryostats_k40",
                                "cryostats_th232",
                                "cryostats_u238lo",
                                "cryostats_u238up",
                                "cryostats_u235",
                                "pmt_body_co60",
                                "pmt_stem_k40",
                                "pmt_stem_mn54",
                                "pmt_stem_th232",
                                "pmt_stem_u235",
                                "pmt_stem_u238lo",
                                "pmt_stem_u238up",
                                "pmt_ceramic_u235",
                                "pmt_ceramic_u238lo",
                                "pmt_ceramic_u238up",
                                "pmt_ceramic_th232",
                                "pmt_ceramic_k40"};
                                
    // MC normalizzation for the rescaling of the various bkg sources' normalizations
    std::string inputScreening_filename = "./datafiles/input_screening.dat";

    double eff      = 0.99*0.9975*0.97;
    double day      = 3600*24;
    double livetime = 653.1*day*eff;

    std::vector<std::string> pmt_names;
    int Npmt;
    std::vector<double> pmt_scale;
    std::vector<std::string> cryo_names;
    int Ncryo;
    std::vector<double> cryo_scale;
    double ar39_scale;
    double kr85_scale;

    bool irebin = true;
    
    bool expected = false;
    
    bool ScreeningInit = false;
    bool y_ar39Init = false;
    bool y_ar39Init_SM = false;
    bool y_migInit  = false;
    bool y_migInit_SM = false;
    bool y_kr85Init = false;
    bool y_kr85Init_SM = false;
    bool y_pmtInit = false;
    bool y_pmtInit_SM = false;
    bool y_cryoInit = false;
    bool y_cryoInit_SM = false;

    bool WIMP_init = false;
    bool WIMP_SM_init = false;

    bool x_theoInit = false;

    bool dataInit = false;

    bool debug = false;
    bool stat_only = false;

    std::vector<std::string> active_ch = {"24", "25", "26", "29", "30", "31", "35"};
    int Nch = 7;

    std::vector<double> x_theo;

    std::vector<double> y_data;

    int debug_count;
    
    std::string MW;

    std::vector<std::vector<double>> y_WIMP;
    std::vector<double>              x_WIMP;

    std::vector<std::vector<double>> y_ar39;
    std::vector<double> y_ar39_Q;
    std::vector<std::vector<double>> y_kr85;
    std::vector<double> y_kr85_Q;
    std::vector<std::vector<std::vector<double>>> y_pmt;
    std::vector<std::vector<std::vector<double>>> y_cryo;

    
    std::vector<std::vector<std::vector<double>>> SM_WIMP;
    std::vector<std::vector<std::vector<double>>> SM_ar39;
    std::vector<std::vector<std::vector<double>>> SM_migdal;
    std::vector<std::vector<std::vector<double>>> y_migdal;
    std::vector<std::vector<std::vector<double>>> SM_kr85;
    std::vector<std::vector<std::vector<std::vector<double>>>> SM_pmt;
    std::vector<std::vector<std::vector<std::vector<double>>>> SM_cryo;

    std::vector<double> peak_eff_central;
    std::vector<double> peak_eff_ring;
    
    std::vector<double> norm_nr;



    //////////// cuda - GPU VARIABLES //////////////
    // They must be threadprivate to avoid issues
    // when running parallel chains

    static cudaError_t cudaStat;
    static cublasStatus_t stat;
    static cublasHandle_t handle;
    #pragma omp threadprivate(cudaStat, stat, handle)
    
    
    static float** yar39ch;
    static float** car39ch;
    static float** ykr85ch;
    static float** ckr85ch;
    #pragma omp threadprivate(yar39ch, car39ch, ykr85ch, ckr85ch)
    
    
    static float** ymigch;
    static float** cmigch;
    #pragma omp threadprivate(ymigch, cmigch)
    
    static float** ywimpch;
    static float** cwimpch;
    #pragma omp threadprivate(ywimpch, cwimpch)
    
    static float** ypmtch;
    static float** cpmtch;
    static float** ycryoch;
    static float** ccryoch;
    #pragma omp threadprivate(ypmtch, cpmtch, ycryoch, ccryoch)
    
    
    static std::vector<double> cache_pars;
    #pragma omp threadprivate(cache_pars)
    
    static float* cM1;
    static float* dM1;
    static float* cM1mig;
    static float* dM1mig;
    static std::vector<std::vector<double>> cache_M1;
    static std::vector<std::vector<double>> cache_M1mig;
    static float** cSMar39;
    static float** dSMar39;
    static float** cSMkr85;
    static float** dSMkr85;
    #pragma omp threadprivate(cM1, dM1, cM1mig, dM1mig, cache_M1, cache_M1mig, cSMar39, dSMar39, cSMkr85, dSMkr85)
    
    static float* PeakEffCent_c;
    static float* PeakEffCent_d;
    static float* PeakEffRing_c;
    static float* PeakEffRing_d;
    #pragma omp threadprivate(PeakEffCent_c, PeakEffCent_d, PeakEffRing_c, PeakEffRing_d)

    
    static float** cSMmig;
    static float** dSMmig;
    #pragma omp threadprivate(cSMmig, dSMmig)
    
    static float* cM1NR;
    static float* dM1NR;
    static std::vector<std::vector<double>> cache_M1NR;
    #pragma omp threadprivate(cM1NR, dM1NR, cache_M1NR)
    
    static float** cSMWIMP;
    static float** dSMWIMP;
    static float** tmp5ch;
    static float* totwimp;
    #pragma omp threadprivate(cSMWIMP, dSMWIMP, tmp5ch, totwimp)
    
    static float** cSMpmt;
    static float** dSMpmt;
    static float** cSMcryo;
    static float** dSMcryo;
    #pragma omp threadprivate(cSMpmt, dSMpmt, cSMcryo, dSMcryo)
    
    static float** tmp1ch;
    static float** tmp2ch;
    static float* totar;
    static float* totkr;
    #pragma omp threadprivate(tmp1ch, tmp2ch, totar, totkr)
    
    static float** tmp3ch;
    static float** tmp4ch;
    static float* totpmt;
    static float* totcryo;
    #pragma omp threadprivate(tmp3ch, tmp4ch, totpmt, totcryo)
    
    static float* M1_tr;
    static float* M1mig_tr;
    #pragma omp threadprivate(M1mig_tr)
    
    
    static float** tmp6ch;
    static float** tmp7ch;
    static float** tmp8ch;
    static float* totmig;
    #pragma omp threadprivate(tmp6ch, tmp7ch, tmp8ch, totmig)
    
    static float** yAr_00;
    static float** yAr_01;
    static float** yKr_00;
    static float** yKr_01;
    #pragma omp threadprivate(yAr_00, yAr_01, yKr_00, yKr_01)
    
    static float** cAr_00;
    static float** cAr_01;
    static float** cKr_00;
    static float** cKr_01;
    #pragma omp threadprivate(cAr_00, cAr_01, cKr_00, cKr_01)
    
    
    static float** yAr_02;
    static float** yKr_02;
    static float** cAr_02;
    static float** cKr_02;
    #pragma omp threadprivate(yAr_02, yKr_02, cAr_02, cKr_02)
    
    static float* zerov;
    static float* czerov;
    #pragma omp threadprivate(zerov, czerov)
    
    static int ncalls;
    #pragma omp threadprivate(ncalls)
    
    static float* yBKG;
    static float* ySIG;
    #pragma omp threadprivate(yBKG, ySIG)
    
    int calls =0;
    int nth =0;

};
// ---------------------------------------------------------

#endif
