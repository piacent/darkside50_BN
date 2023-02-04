// ***************************************************************
// This file was created using the bat-project script.
// bat-project is part of Bayesian Analysis Toolkit (BAT).
// BAT can be downloaded from http://mpp.mpg.de/bat
//
// Author: Stefano Piacentini
// Date: 03/02/2023
// ***************************************************************

#include "DS50lowmass.h"
#include "cnpy.h"
#include <unistd.h>
#include <BAT/BCMath.h>
#include "Math/ProbFunc.h"
#include "Math/PdfFuncMathCore.h"
#include <cmath>
#include <sstream>
#include <omp.h>
#include <ctime>
#include <chrono>
#define IDX2C(i,j,ld) (((j)*(ld))+( i ))

///////// ThreadPrivate Variables ////////////
int  DS50lowmass::ncalls;
float*  DS50lowmass::yBKG;
float*  DS50lowmass::ySIG;

//////////// cuda - GPU VARIABLES //////////////
cudaError_t  DS50lowmass::cudaStat;
cublasStatus_t  DS50lowmass::stat;
cublasHandle_t  DS50lowmass::handle;


float**  DS50lowmass::ymigch;
float**  DS50lowmass::cmigch;
float**  DS50lowmass::cSMmig;
float**  DS50lowmass::dSMmig;

float**  DS50lowmass::yar39ch;
float**  DS50lowmass::car39ch;
float**  DS50lowmass::ykr85ch;
float**  DS50lowmass::ckr85ch;
float**  DS50lowmass::ywimpch;
float**  DS50lowmass::cwimpch;
float**  DS50lowmass::ypmtch;
float**  DS50lowmass::cpmtch;
float**  DS50lowmass::ycryoch;
float**  DS50lowmass::ccryoch;

float*  DS50lowmass::cM1;
float*  DS50lowmass::dM1;
float*  DS50lowmass::cM1mig;
float*  DS50lowmass::dM1mig;
std::vector<std::vector<double>>  DS50lowmass::cache_M1;
std::vector<std::vector<double>>  DS50lowmass::cache_M1mig;

float*  DS50lowmass::M1_tr;
float*  DS50lowmass::M1mig_tr;
float*  DS50lowmass::cM1NR;
float*  DS50lowmass::dM1NR;
std::vector<std::vector<double>>  DS50lowmass::cache_M1NR;

float*  DS50lowmass::PeakEffCent_c;
float*  DS50lowmass::PeakEffCent_d;
float*  DS50lowmass::PeakEffRing_c;
float*  DS50lowmass::PeakEffRing_d;

std::vector<double>  DS50lowmass::cache_pars;

float**  DS50lowmass::cSMar39;
float**  DS50lowmass::dSMar39;
float**  DS50lowmass::cSMkr85;
float**  DS50lowmass::dSMkr85;
float**  DS50lowmass::cSMpmt;
float**  DS50lowmass::dSMpmt;
float**  DS50lowmass::cSMcryo;
float**  DS50lowmass::dSMcryo;
float**  DS50lowmass::cSMWIMP;
float**  DS50lowmass::dSMWIMP;

float**  DS50lowmass::tmp1ch;
float**  DS50lowmass::tmp2ch;
float**  DS50lowmass::tmp3ch;
float**  DS50lowmass::tmp4ch;
float**  DS50lowmass::tmp5ch;
float**  DS50lowmass::tmp6ch;
float**  DS50lowmass::tmp7ch;
float**  DS50lowmass::tmp8ch;
float*  DS50lowmass::totar;
float*  DS50lowmass::totkr;
float*  DS50lowmass::totpmt;
float*  DS50lowmass::totcryo;
float*  DS50lowmass::totwimp;
float*  DS50lowmass::totmig;

float**  DS50lowmass::yAr_00;
float**  DS50lowmass::yAr_01;
float**  DS50lowmass::yAr_02;
float**  DS50lowmass::yKr_00;
float**  DS50lowmass::yKr_01;
float**  DS50lowmass::yKr_02;
float**  DS50lowmass::cAr_00;
float**  DS50lowmass::cAr_01;
float**  DS50lowmass::cAr_02;
float**  DS50lowmass::cKr_00;
float**  DS50lowmass::cKr_01;
float**  DS50lowmass::cKr_02;
float*  DS50lowmass::zerov;
float*  DS50lowmass::czerov;


// ---------------------------------------------------------
 DS50lowmass:: DS50lowmass(const std::string& name, int Nmin, int Nmax, int nth, std::string mass, bool idebug = true,
                           bool statonly = false, bool rebin = true, bool iexpected = false)
    : BCModel(name)
{
    
    std::cout<<"STARTING FIT FOR MASS = "<<mass<<" GEV"<<std::endl;
     DS50lowmass::debug = idebug; // debug flag
     DS50lowmass::irebin = rebin;

     DS50lowmass::Nmin = Nmin;
     DS50lowmass::Nmax = Nmax;
        
     DS50lowmass::expected = iexpected;

        
    std::string allowed_m[35] = {"0.030",
                                 "0.063", "0.070", "0.079", "0.089", "0.100",
                                 "0.117", "0.137", "0.161",
                                 "0.189", "0.221", "0.259", "0.304", "0.356",
                                 "0.418", "0.489", "0.574", "0.672", "0.788",
                                 "0.924", "1.08", "1.27", "1.49", "1.74",
                                 "2.04", "2.40", "2.81", "3.29", "3.86",
                                 "4.52", "5.30", "6.21", "7.28", "8.53", "10.0"};
        
    // Upper integration bound for r_S
    double rs_max_v[35] = {1.0e4,
                           1.0e4,   1.0e4,  1.0e4, 1.0e4,   1.0e4,
                           1.0e4,   1.0e4,  1.0e4,
                           1.0e3,   1.0e3,  1.0e3, 1.0e3,   1.0e3,
                           1.0e2,   1.0e2,  1.0e2, 1.0e1,   1.0e1,
                           1.0e1,   1.0e1,  1.0e1, 1.0e-1,   1.0e-1,
                           1.0e-2,   1.0e-2,  1.0e-2, 1.0e-2, 1.0e-2,
                           1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2
                          };
        
    // Check if the mass is ok
    int i, i_rs;
    i_rs = -1;
    for(i=0;i<35;i++) {
        if(mass.compare(allowed_m[i]) == 0) {
            i_rs = i;
        }
    }
    if(i_rs <0) {
        throw std::runtime_error("Wrong mass input");
    }
    
     DS50lowmass::stat_only = statonly;
    
        
     DS50lowmass::initScreening();
    
     DS50lowmass::getPeakEff();
     DS50lowmass::getNormNR();
         
     DS50lowmass::getar39_theo();
     DS50lowmass::getkr85_theo();
     DS50lowmass::getar39_Q();
     DS50lowmass::getkr85_Q();
     DS50lowmass::getpmt_theo();
     DS50lowmass::getcryo_theo();
     DS50lowmass::getx_theo();
        
     DS50lowmass::getdata();
    
    
     DS50lowmass::getWIMP_theo(mass);
     DS50lowmass::getWIMP_SM();

     DS50lowmass::getar39_SM();
     DS50lowmass::getkr85_SM();
     DS50lowmass::getpmt_SM();
     DS50lowmass::getcryo_SM();
        
    
     DS50lowmass::getMigdal(mass);
        
     DS50lowmass::getMigdal_SM();
    
    omp_set_dynamic(0);
    omp_set_num_threads(nth);
        
        
    #pragma omp parallel
    {
        if( DS50lowmass::stat_only) {
            int dimA = int( DS50lowmass::SM_ar39[0].size());
             DS50lowmass::yBKG = (float *)malloc (dimA*sizeof(float));
             DS50lowmass::ySIG = (float *)malloc (dimA*sizeof(float));
        }
         DS50lowmass::init_cuda();
        
    }
     
    // DEFINING PARAMETERS
    
    double ekr = 0.047;
    AddParameter("rB_IntKr", 1.0-5.0*ekr, 1.0+5.0*ekr, "r_{B, IntKr}", "[events]");
    if(statonly) GetParameter("rB_IntKr").Fix(1.0);

    double eg2 = 1.0;
    AddParameter("g2", 23.0-5.0*eg2, 23.0+eg2, "g_{2}","");
    GetParameter("g2").Fix(23.0);
    if(statonly) GetParameter("g2").Fix(23.0);
        
    double ep0 = sqrt(0.339);
    double mp0 = 2.49;
    AddParameter("p0", mp0-5.0*ep0, mp0+5.0*ep0, "p_{0}", "");
    if(statonly) GetParameter("p0").Fix(2.49);
        
    double ep1 = sqrt(4.85);
    double mp1 = 21.84;
    AddParameter("p1", mp1-5.0*ep1, mp1+5.0*ep1, "p_{1}", "");
    if(statonly) GetParameter("p1").Fix(21.84);
        
    double ep2 = sqrt(0.000737);
    double mp2 = 0.1141;
    AddParameter("p2", mp2-5.0*ep2, mp2+5.0*ep2, "p_{2}", "");
    if(statonly) GetParameter("p2").Fix(0.1141);
        
    double ep3 = sqrt(0.00591);
    double mp3 = 1.7089;
    AddParameter("p3", mp3-5.0*ep3, mp3+5.0*ep3, "p_{3}", "");
    if(statonly) GetParameter("p3").Fix(1.7089);
    
    double ecboxnr = 0.15;
    double mcboxnr = 8.05;
    AddParameter("CboxNR", mcboxnr-5.0*ecboxnr, mcboxnr+5.0*ecboxnr, "C_{box,NR}", "");
    if(statonly) GetParameter("CboxNR").Fix(mcboxnr);
    
    double efb = 0.02;
    double mfb = 0.67;
    AddParameter("fB", mfb-5.0*efb, mfb+5.0*efb, "f_{B}", "");
    if(statonly) GetParameter("fB").Fix(mfb);
        
    AddParameter("rS", 0.0 , rs_max_v[i_rs], "r_{S}", "");
        
        
    AddParameter("scr", -5.0, 5.0, "sigma_{scr}", "");
    if(statonly) GetParameter("scr").Fix(0.0);
    
    double epmt = 0.126;
    AddParameter("rB_ExtPMT", 1.0-5.0*epmt, 1.0+5.0*epmt, "r_{B, ExtPMT}", "[events]");
    if(statonly) GetParameter("rB_ExtPMT").Fix(1.0);

    double eexp = 0.015;
    AddParameter("r_exp", 1.0-5.0*eexp, 1.0+5.0*eexp, "r_{Exp}", "[events]");
    if(statonly) GetParameter("r_exp").Fix(1.0);
        
    AddParameter("Q_ar", -5.0, 5.0, "Q_{Ar}", "");
    if(statonly) GetParameter("Q_ar").Fix(0.0);
        
    AddParameter("Q_kr", -5.0, 5.0, "Q_{Kr}", "");
    if(statonly) GetParameter("Q_kr").Fix(0.0);
        
    double ear = 0.14;
    AddParameter("rB_IntAr", 1.0-5.0*ear, 1.0+5.0*ear, "r_{B, IntAr}", "[events]");
    if(statonly) GetParameter("rB_IntAr").Fix(1.0);
    
    double ecryo = 0.066;
    AddParameter("rB_Extcryo", 1.0-5.0*ecryo, 1.0+5.0*ecryo, "r_{B, Extcryo}", "[events]");
    if(statonly) GetParameter("rB_Extcryo").Fix(1.0);
    
    AddParameter("scrKr", -5.0, 5.0, "sigma_{scrKr}", "");
    if(statonly) GetParameter("scrKr").Fix(0.0);
        
     DS50lowmass::debug_count = 0;
        
}

// ---------------------------------------------------------
 DS50lowmass::~ DS50lowmass()
{
    // destructor
}


// ---------------------------------------------------------
double  DS50lowmass::LogLikelihood(const std::vector<double>& pars)
{
    int i, j, k, l;
    double tmp1, tmp2;
    double yield_tmp;
    
    int nm, Nm;
    
    int dimA, dimB, dimC;
    // prior sampling
    bool priorsampling = false; 
    if(priorsampling) return 1.0; //constant likelihood ---> prior == posterior

        
    double LL = 0.0;
    
    dimA = int( DS50lowmass::SM_ar39[0].size());
    dimB = int( DS50lowmass::y_ar39[0].size());
    dimC = int( DS50lowmass::x_WIMP.size());
    
    std::string allowed_m[35] = { "0.030",
                        "0.063", "0.070", "0.079", "0.089", "0.100",
                        "0.117", "0.137", "0.161",
                        "0.189", "0.221", "0.259", "0.304", "0.356",
                        "0.418", "0.489", "0.574", "0.672", "0.788",
                        "0.924", "1.08", "1.27", "1.49", "1.74",
                        "2.04", "2.40", "2.81", "3.29", "3.86",
                        "4.52", "5.30", "6.21", "7.28", "8.53", "10.0"};
    
    int i_rs;
    i_rs = -1;
    for(i=0;i<35;i++) {
        if( DS50lowmass::MW.compare(allowed_m[i]) == 0) {
            i_rs = i;
        }
    }
    
    // CACHING on calib. parameters
    bool cache_check = true;
    for(i=0;i<7;i++) {
        if(cache_pars[i] != pars[i+1]) {
            cache_check = false;
        }
    }
    
    if(cache_check == false) {
        for(i=0;i<7;i++) {
            cache_pars[i] = pars[i+1];
        }
    }
    
    // If not stat_only or if stat_only but it's the first call to the
    // Likelihood we have to compute the spectra
    if((! DS50lowmass::stat_only) || ( DS50lowmass::stat_only && ncalls == 0)) {
    
        // M1 ER Matrix
        
        if(cache_check == false) {
            for(j=0;j<dimB;j++){
                yield_tmp = 1.0 +  DS50lowmass::QyER( DS50lowmass::x_theo[j],
                                          pars[1],
                                          pars[2],
                                          pars[3],
                                          pars[4],
                                          pars[5]
                                         )*( DS50lowmass::x_theo[j]);
                for(i=0;i<dimA;i++){
                    tmp1 = ROOT::Math::normal_cdf(double(i+1.0), // x
                                                  sqrt( DS50lowmass::FANO * yield_tmp), // sigma
                                                  yield_tmp // mu
                                                  );
                    tmp2 = ROOT::Math::normal_cdf(double(i), // x
                                                  sqrt( DS50lowmass::FANO * yield_tmp), // sigma
                                                  yield_tmp // mu
                                                  );
                    cM1[IDX2C(i, j, dimA)] = tmp1 - tmp2;
                    cache_M1[i][j] = cM1[IDX2C(i, j, dimA)];
                }
            }
     
            // M1 NR matrix
            double W = 19.5*1.0e-3;
            double L = 1.0;
            for(j=0; j<dimC; j++){
                yield_tmp =  DS50lowmass::QyNR( DS50lowmass::x_WIMP[j],
                                            pars[6],
                                            pars[7],
                                            pars[1]
                                           );
                
                for(i=0; i<dimA;i++) {
                    double p_bin = std::min(1.0, W * yield_tmp / L);
                    double n_bin = floor(x_WIMP[j] * L / W);

                    cM1NR[IDX2C(i, j, dimA)] =ROOT::Math::binomial_pdf(i,     // k successes
                                                                   p_bin,     // p probability of success
                                                                   n_bin     // n trials
                                                                   );
                    cache_M1NR[i][j] = cM1NR[IDX2C(i, j, dimA)];
                }
            }
            
        } else {
            for(i=0;i<dimA;i++){
                for(j=0;j<dimB;j++){
                    cM1[IDX2C(i, j, dimA)] = cache_M1[i][j];
                }
            
                for(j=0;j<dimC;j++){
                    cM1NR[IDX2C(i, j, dimA)] = cache_M1NR[i][j];
                }
            }
            
        }
        stat =cublasSetMatrix(dimA,dimB,sizeof(float),cM1,dimA,dM1,dimA); // load on GPU
        stat =cublasSetMatrix(dimA,dimC,sizeof(float),cM1NR,dimA,dM1NR,dimA); // load on GPU


        /////////////////////// GPU COMPUTATIONS //////////////////////////
        float al = 1.0f;
        float bet = 0.0f;
        float ars = (float)ar39_scale;
        float krs = (float)kr85_scale;
        ///////
        float sar = (float)pars[9];
        float skr = (float)pars[16];
        
        float Qar = (float)pars[12];
        float Qkr = (float)pars[13];
        ///////

        // Fiducial volume fraction in each channel
        // numerically (computed with the lowmass code)
        float fiducial[7] = {0.1427*7.0, 0.1427*7.0, 0.1422*7.0, 0.1432*7.0, 0.1430*7.0, 0.1426*7.0, 0.1435*7.0};

        float* mig;
        float* pmat;
        
        pmat   = (float *)malloc(dimA*dimA*sizeof(float));
        mig = (float *)malloc(dimA*sizeof(float));


        for(l=0;l<Nch;l++) {
            /////// Apply sigma_scr to the Ar and Kr theretical spectra
            stat=cublasScopy(handle, dimB, czerov, 1, cAr_01[l], 1);
            stat=cublasScopy(handle, dimB, czerov, 1, cKr_01[l], 1);

            stat=cublasSaxpy(handle,dimB,&al,car39ch[l],1,cAr_01[l],1);
            stat=cublasSaxpy(handle,dimB,&sar,cAr_00[l],1,cAr_01[l],1);
            stat=cublasSaxpy(handle,dimB,&Qar,cAr_02[l],1,cAr_01[l],1);
            
            stat=cublasSaxpy(handle,dimB,&al,ckr85ch[l],1,cKr_01[l],1);
            stat=cublasSaxpy(handle,dimB,&skr,cKr_00[l],1,cKr_01[l],1);
            stat=cublasSaxpy(handle,dimB,&Qkr,cKr_02[l],1,cKr_01[l],1);
            ///////

            // Compute M2 * M1 * y_Ar
            stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimB, &al, dM1, dimA, cAr_01[l] , 1, &bet, tmp1ch[l], 1);
            stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, dSMar39[l], dimA, tmp1ch[l] , 1, &bet, tmp1ch[l], 1);
            
            
            // Compute M2 * M1 * y_Kr
            stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimB, &al, dM1, dimA, cKr_01[l] , 1, &bet, tmp2ch[l], 1);
            stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, dSMkr85[l], dimA, tmp2ch[l] , 1, &bet, tmp2ch[l], 1);

            // Compute M2 * M1 * y_sig_NR
            stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimC, &al, dM1NR, dimA, cwimpch[l] , 1, &bet, tmp5ch[l], 1);
            stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, dSMWIMP[l], dimA, tmp5ch[l] , 1, &bet, tmp5ch[l], 1);

            
            
            // Compute M1_ER^T
            stat=cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, dimB, dimA,
                             &al, dM1, dimA, &bet, M1_tr, dimB, M1_tr, dimB);

            // Compute dR * M1_ER^T
            stat=cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimC, dimA, dimB,
                             &al, cmigch[l], dimC, M1_tr, dimB,   
                             &bet, tmp6ch[l], dimC);

            // Compute pmat = (M1_NR * dR * M1_ER^T)
            stat=cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimA, dimA, dimC,
                             &al, dM1NR, dimA, tmp6ch[l], dimC, // M1NR * migdal * M1^T
                             &bet, tmp7ch[l], dimA);

            // Read pmat from GPU for Migdal computation
            stat=cublasGetMatrix(dimA,dimA, sizeof(float),tmp7ch[l],dimA,pmat,dimA);

            // Migdal Ne spectrum computation
            double tmpsum;
            for(i=0;i<dimA;i++) {
                tmpsum   = 0.0;
                for(j=0;j<=i;j++) {
                      tmpsum   +=   pmat[(i-j)  + j * dimA];
                }
                mig[i] = tmpsum;
            }
            
            // Load Migdal on GPU
            stat=cublasSetVector(dimA, sizeof(float), mig, 1, tmp8ch[l], 1);
            
            // M2 * y_Migdal_Ne
            stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, dSMmig[l], dimA, tmp8ch[l] , 1, &bet, tmp8ch[l], 1);

            // Sum on all the Nch channels and multiply by the activities, efficiencies, etc
            stat=cublasSscal(handle,dimA,&ars,tmp1ch[l],1);
            stat=cublasSscal(handle,dimA,&krs,tmp2ch[l],1);
            stat=cublasSscal(handle,dimA,&(fiducial[l]),tmp5ch[l],1);
            stat=cublasSscal(handle,dimA,&(fiducial[l]),tmp8ch[l],1);
            
            //----------------------------
            // SIG (NR + MIG), AR and KR
            //----------------------------
            
            // Apply Peak Time Efficiency
            if(l==4) { // central PMT
                stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, PeakEffCent_d, dimA,
                                 tmp1ch[l],1, &bet, tmp1ch[l], 1);
                stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, PeakEffCent_d, dimA,
                                 tmp2ch[l],1, &bet, tmp2ch[l], 1);
                stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, PeakEffCent_d, dimA,
                                 tmp5ch[l],1, &bet, tmp5ch[l], 1);
                stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, PeakEffCent_d, dimA,
                                 tmp8ch[l],1, &bet, tmp8ch[l], 1);
            } else {
                stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, PeakEffRing_d, dimA,
                                 tmp1ch[l],1, &bet, tmp1ch[l], 1);
                stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, PeakEffRing_d, dimA,
                                 tmp2ch[l],1, &bet, tmp2ch[l], 1);
                stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, PeakEffRing_d, dimA,
                                 tmp5ch[l],1, &bet, tmp5ch[l], 1);
                stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, PeakEffRing_d, dimA,
                                 tmp8ch[l],1, &bet, tmp8ch[l], 1);
            }
            
            if(l==0){
                stat=cublasScopy(handle,dimA,tmp1ch[l],1,totar,1);
                stat=cublasScopy(handle,dimA,tmp2ch[l],1,totkr,1);
                stat=cublasScopy(handle,dimA,tmp5ch[l],1,totwimp,1);
                stat=cublasScopy(handle,dimA,tmp8ch[l],1,totmig,1);
            } else {
                stat=cublasSaxpy(handle,dimA,&al,tmp1ch[l],1,totar,1);
                stat=cublasSaxpy(handle,dimA,&al,tmp2ch[l],1,totkr,1);
                stat=cublasSaxpy(handle,dimA,&al,tmp5ch[l],1,totwimp,1);
                stat=cublasSaxpy(handle,dimA,&al,tmp8ch[l],1,totmig,1);
            }
            
            //----------------------------
            // PMT and CRYO
            //----------------------------
            
            // Compute M2 * M1 * y_PMT and sum on all the Nch channels
            // and multiply by the activities, efficiencies, etc 
            for(k=0;k<Npmt;k++) {
                stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimB, &al, dM1, dimA, cpmtch[k*Nch+l],
                                      1, &bet, tmp3ch[k*Nch+l], 1);
                stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, dSMpmt[k*Nch+l], dimA, tmp3ch[k*Nch+l] ,
                                      1, &bet, tmp3ch[k*Nch+l], 1);

                float pmts = (float) DS50lowmass::pmt_scale[k];
                stat=cublasSscal(handle, dimA, &pmts, tmp3ch[k*Nch+l], 1);
                
                // Apply Peak Time Efficiency
                if(l==4) { // central PMT
                    stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, PeakEffCent_d, dimA,
                                     tmp3ch[k*Nch+l],1, &bet, tmp3ch[k*Nch+l], 1);
                } else {
                    stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, PeakEffRing_d, dimA,
                                     tmp3ch[k*Nch+l],1, &bet, tmp3ch[k*Nch+l], 1);
                }
                
                if(l==0 && k==0) {
                    stat=cublasScopy(handle,dimA,tmp3ch[k*Nch+l],1,totpmt,1);
                } else {
                    stat=cublasSaxpy(handle,dimA,&al,tmp3ch[k*Nch+l],1,totpmt,1);
                }

            }
            
            // Compute M2 * M1 * y_cryo and sum on all the Nch channels
            // and multiply by the activities, efficiencies, etc 
            for(k=0;k<Ncryo;k++) {
                stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimB, &al, dM1, dimA, ccryoch[k*Nch+l],
                                      1, &bet, tmp4ch[k*Nch+l], 1);
                stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, dSMcryo[k*Nch+l], dimA, tmp4ch[k*Nch+l] ,
                                      1, &bet, tmp4ch[k*Nch+l], 1);

                float cryos = (float) DS50lowmass::cryo_scale[k];
                stat=cublasSscal(handle, dimA, &cryos, tmp4ch[k*Nch+l], 1);
                
                // Apply Peak Time Efficiency
                if(l==4) { // central PMT
                    stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, PeakEffCent_d, dimA,
                                     tmp4ch[k*Nch+l],1, &bet, tmp4ch[k*Nch+l], 1);
                } else {
                    stat=cublasSgemv(handle, CUBLAS_OP_N, dimA, dimA, &al, PeakEffRing_d, dimA,
                                     tmp4ch[k*Nch+l],1, &bet, tmp4ch[k*Nch+l], 1);
                }
            
                
                if(l==0 && k==0) {
                    stat=cublasScopy(handle,dimA,tmp4ch[k*Nch+l],1,totcryo,1);
                } else {  
                    stat=cublasSaxpy(handle,dimA,&al,tmp4ch[k*Nch+l],1,totcryo,1);
                }
            }
            
            

        }

        // Free manually allocated memory to avoid memory leaks
        free(pmat);
        free(mig);



        /////////////////////// FROM GPU TO CPU ////////////////////////
        float* yar;
        float* ykr;
        float* ypmt;
        float* ycryo;
        float* ywimp;
        float* ymig;
        yar   = (float *)malloc(dimA*sizeof(float));
        ykr   = (float *)malloc(dimA*sizeof(float));
        ypmt  = (float *)malloc(dimA*sizeof(float));
        ycryo = (float *)malloc(dimA*sizeof(float));
        ywimp = (float *)malloc(dimA*sizeof(float));
        ymig  = (float *)malloc(dimA*sizeof(float));
        stat=cublasGetVector(dimA, sizeof(float),totar  ,1,yar  ,1);
        stat=cublasGetVector(dimA, sizeof(float),totkr  ,1,ykr  ,1);
        stat=cublasGetVector(dimA, sizeof(float),totpmt ,1,ypmt ,1);
        stat=cublasGetVector(dimA, sizeof(float),totcryo,1,ycryo,1);
        stat=cublasGetVector(dimA, sizeof(float),totwimp,1,ywimp,1);
        stat=cublasGetVector(dimA, sizeof(float),totmig ,1,ymig ,1);

        float* ysig;
        ysig  = (float *)malloc(dimA*sizeof(float));

        double lambda_i;

        double exp_factor = 0.414; // Fiducialization factor
        
        double normNR = 0.0;
        normNR = norm_nr[i_rs];
        
        for(i=0;i<dimA;i++) {
                ysig[i] = exp_factor * (ywimp[i] + (ymig[i]) * normNR);
        }
        
        // Rebinning indices
        if(irebin) {
            nm = int(Nmin * 4);
            Nm = int(Nmax - 20 + 20*4);
        } else {
            nm = int(Nmin);
            Nm = int(Nmax);
        }
        
        
        // Likelihood calculation
        
        for(i=nm; i<Nm;i++){
            lambda_i = pars[11] * (pars[14] * yar[i] + pars[0] * ykr[i] +  pars[10] * ypmt[i] + pars[15] * ycryo[i]) + 
                       (pars[11] * pars[8] * ysig[i]);
            
            LL += BCMath::LogPoisson(int( DS50lowmass::y_data[i]), lambda_i);

        }
        
        // Save the background and signal spectra for the statonly case 
        if( DS50lowmass::stat_only && ncalls == 0) {
            for(i=0; i<dimA;i++) {
                 DS50lowmass::yBKG[i] = pars[11] * (pars[14] * yar[i] + pars[0] * ykr[i] +  pars[10] * ypmt[i] + pars[15] * ycryo[i]);
                 DS50lowmass::ySIG[i] = pars[11] * ysig[i];
            }
        }
        
        // Free manually allocated memory to avoid memory leaks
        free(yar);
        free(ykr);
        free(ypmt);
        free(ycryo);
        free(ywimp);
        free(ymig);
        
    } else {
        // if we are in the statonly case then the background and signal spectra
        // has been already computed
        double lambda_i;
        if(irebin) {
            nm = int(Nmin * 4);
            Nm = int(Nmax - 20 + 20*4);
        } else {
            nm = int(Nmin);
            Nm = int(Nmax);
        }
        
        for(i=nm; i<Nm;i++){
            lambda_i =  DS50lowmass::yBKG[i] + pars[8] *  DS50lowmass::ySIG[i];
            LL += BCMath::LogPoisson(int( DS50lowmass::y_data[i]), lambda_i);
        }
    }
    
    if( DS50lowmass::stat_only) ncalls ++;
    
    return LL;
}

// ---------------------------------------------------------
 double  DS50lowmass::LogAPrioriProbability(const std::vector<double>& pars)
{
     // return the log of the prior probability p(pars)
     
    double rB, rBdet;
    double cal_pars[5];
    double rB_mean, rBdet_mean, g2_mean;
    double rB_std, rBdet_std, g2_std;
    
    double rS;
    double cal_parsNR[2];
    double fB_mean, CboxNR_mean;
    double rS_std, fB_std, CboxNR_std;
    double cc_CboxNR_fB;

    double sar_mean;
    double sar_std;
    double sar;
     
    double skr_mean;
    double skr_std;
    double skr;
     
    double Qar_mean;
    double Qar_std;
    double Qar;
    double Qkr_mean;
    double Qkr_std;
    double Qkr;
    
    double rexp_mean;
    double rexp_std;
    double rexp;
     
    
    double rBar_mean;
    double rBar_std;
    double rBar;
    double rBcryo_mean;
    double rBcryo_std;
    double rBcryo;
     
    sar = pars[9];
    sar_mean = 0.0;
    sar_std = 1.0;
     
    skr = pars[16];
    skr_mean = 0.0;
    skr_std = 1.0;
     
     
    Qar = pars[12];
    Qar_mean = 0.0;
    Qar_std = 1.0;
     
    Qkr = pars[13];
    Qkr_mean = 0.0;
    Qkr_std = 1.0;
     
    rBar = pars[14];
    rBar_mean = 1.0;
    rBar_std = 0.140;
     
    rBcryo = pars[15];
    rBcryo_mean = 1.0;
    rBcryo_std = 0.066;

    rB          = pars[0];
    cal_pars[0] = pars[1];
    cal_pars[1] = pars[2];
    cal_pars[2] = pars[3];
    cal_pars[3] = pars[4];
    cal_pars[4] = pars[5];
    rBdet       = pars[10];

    rS            = pars[8];
    cal_parsNR[0] = pars[6];
    cal_parsNR[1] = pars[7];

    
    rS_std  = 10000.0;

    CboxNR_mean  = 8.05;
    CboxNR_std   = 0.15;
    fB_mean      = 0.67;
    fB_std       = 0.02;
    cc_CboxNR_fB = 0.72485; 
    
    rB_mean = 1.0;
    rB_std  = 0.047;
    rBdet_mean = 1.0;
    rBdet_std  = 0.126;

    g2_mean = 23.0;
    g2_std  =  1.0;

    rexp      = pars[11]; 
    rexp_mean = 1.0;
    rexp_std  = 0.015;

    double mean[5] = {g2_mean, 2.49, 21.84, 0.1141, 1.7089};
    double cov[5][5]= {{g2_std*g2_std, 0., 0., 0., 0.},
                       {0.,  0.339,   -1.27,   0.0116,   -0.035},
                       {0.,  -1.27,    4.85,  -0.0478,    0.142},
                       {0., 0.0116, -0.0478, 0.000737, -0.00208},
                       {0., -0.035,   0.142, -0.00208,  0.00591}
                      }; 
     
     
    double meanNR[2] = {CboxNR_mean, fB_mean};
    double covNR[2][2]=  {{CboxNR_std*CboxNR_std             , cc_CboxNR_fB * CboxNR_std * fB_std},
                          {cc_CboxNR_fB * CboxNR_std * fB_std, fB_std*fB_std                     }
                         };
     
    // Maths needed to compute the LogProbability for the multivariate normal
    double cov_inv[5][5];
     DS50lowmass::getMatrixFromGsl( DS50lowmass::invert_a_matrix( DS50lowmass::getFromDoubleArray(cov)), cov_inv);
    double det_sigma;
    det_sigma =  DS50lowmass::matrix_det( DS50lowmass::getFromDoubleArray(cov));

    double cov_invNR[2][2];
     DS50lowmass::getMatrixFromGsl( DS50lowmass::invert_a_matrix2( DS50lowmass::getFromDoubleArray(covNR)), cov_invNR);
    double det_sigmaNR;
    det_sigmaNR =  DS50lowmass::matrix_det2( DS50lowmass::getFromDoubleArray(covNR));
    
    double dim = 5;
    double dimNR = 2;
    double exponent = 0.0;
    double exponentNR = 0.0;
    int i,j;

    for(i=0;i<dim;i++) {
        for(j=0;j<dim;j++) {
            exponent += (cal_pars[i] - mean[i]) * (cal_pars[j] - mean[j]) * cov_inv[i][j];
        }
    }
    for(i=0;i<dimNR;i++) {
        for(j=0;j<dimNR;j++) {
            exponentNR += (cal_parsNR[i] - meanNR[i]) * (cal_parsNR[j] - meanNR[j]) * cov_invNR[i][j];
        }
    }
    
    // LogPrior computation
    double Lprior_cal, Lprior_calNR;
    Lprior_cal   = -0.5*dim   *log(2 * M_PI) - 0.5 * log(det_sigma)   - 0.5 * exponent;
    Lprior_calNR = -0.5*dimNR *log(2 * M_PI) - 0.5 * log(det_sigmaNR) - 0.5 * exponentNR;

    double Lprior_rB, Lprior_rS;
    Lprior_rB = - 0.5 * log(2 * M_PI) - 0.5 * log(rB_std*rB_std)
                - 0.5 * (rB - rB_mean)*(rB-rB_mean) / (rB_std*rB_std);
     
    if(rS<0 || rS>rS_std) {
        Lprior_rS = log(0.0);
    } else {
        Lprior_rS = log(1.0/rS_std);
    }
    
    double Lprior_rBdet;
    Lprior_rBdet = - 0.5 * log(2 * M_PI) - 0.5 * log(rBdet_std*rBdet_std)
                   - 0.5 * (rBdet - rBdet_mean)*(rBdet-rBdet_mean) / (rBdet_std*rBdet_std);
     
    double Lprior_rBother;
    Lprior_rBother = - 0.5 * log(2 * M_PI) - 0.5 * log(rBar_std*rBar_std)
                     - 0.5 * (rBar - rBar_mean)*(rBar-rBar_mean) / (rBar_std*rBar_std)
                     - 0.5 * log(2 * M_PI) - 0.5 * log(rBcryo_std*rBcryo_std)
                     - 0.5 * (rBcryo - rBcryo_mean)*(rBcryo-rBcryo_mean) / (rBcryo_std*rBcryo_std);
     
    double Lprior_rexp;
    Lprior_rexp = - 0.5 * log(2 * M_PI) - 0.5 * log(rexp_std*rexp_std)
                  - 0.5 * (rexp - rexp_mean)*(rexp-rexp_mean) / (rexp_std*rexp_std);

    double Lscreen, LscreenKr;
    Lscreen = - 0.5 * log(2 * M_PI) - 0.5 * log(sar_std*sar_std)
              - 0.5 * (sar - sar_mean)*(sar-sar_mean) / (sar_std*sar_std);
    LscreenKr = - 0.5 * log(2 * M_PI) - 0.5 * log(skr_std*skr_std)
                - 0.5 * (skr - skr_mean)*(skr-skr_mean) / (skr_std*skr_std);
     
    
    double LQ;
    LQ      = - 0.5 * log(2 * M_PI) - 0.5 * log(Qar_std*Qar_std)
              - 0.5 * (Qar - Qar_mean)*(Qar-Qar_mean) / (Qar_std*Qar_std)
              - 0.5 * log(2 * M_PI) - 0.5 * log(Qkr_std*Qkr_std)
              - 0.5 * (Qkr - Qkr_mean)*(Qkr-Qkr_mean) / (Qkr_std*Qkr_std);

    return (Lprior_rB + Lprior_rBdet + Lprior_rS + Lprior_rexp + Lprior_cal + Lprior_calNR + Lscreen +
            LQ + Lprior_rBother + LscreenKr);

}


// ---------------------------------------------------------
double  DS50lowmass::QyER(double E, double g2, double p0, double p1, double p2, double p3){
    double g2_0 = 23.0;

    return (g2/g2_0) * (p1 + p2 * pow(E, p3)) * log(1 + p0 * E) / E;
}

// ---------------------------------------------------------
double  DS50lowmass::QyNR(double E, double Cbox, double fB, double g2) {
    double Z = 18;
    double g2_0 = 23.0;
    double fGamma = 4.0 * Cbox /200.0;
    double fZ = 0.9532;
    double eps = 11.501 * E * pow(Z, -7.0/3.0);
    double eps_z = eps * fZ;
    double Sn = log(1 + 1.1383 * eps_z) * 0.5 /
                (eps_z +
                 0.01321 * pow(eps_z, 0.21226) +
                 0.19593 * pow(eps_z, 0.5)
                );
    double Se = 0.0944 * pow(Z, 1.0/6.0) * pow(eps, 0.5);

    
    double kappa = eps * Se / (Sn + Se);
    double Ni    = kappa * fB * 1.0e+4;
    return (g2/g2_0)*(4.0 * log(1.0 + Ni * fGamma * 0.25) / fGamma) / E;
}


// ---------------------------------------------------------
// Read a ./datafiles/input_screening.dat file column
// ---------------------------------------------------------
std::vector<std::string>  DS50lowmass::getValueFromScreening(const std::string comp_name, const std::string value_name){
    
    std::string name;
    std::string active;
    std::string activity, elements, mass, efff;
    std::ifstream inFile;
    inFile.open( DS50lowmass::inputScreening_filename);

    std::vector<std::string> ret;
    
    while ( !inFile.eof () ) {    
        inFile >> name;
        inFile >> active;
        inFile >> activity;
        inFile >> elements;
        inFile >> mass;
        inFile >> efff;

        if(name.find(comp_name) != std::string::npos) {
            if(value_name == "name") ret.push_back(name);
            if(value_name == "active") ret.push_back(active);
            if(value_name == "activity") ret.push_back(activity);
            if(value_name == "elements") ret.push_back(elements);
            if(value_name == "mass") ret.push_back(mass);
            if(value_name == "efff") ret.push_back(efff);
        }
     
    }
    
    return ret;
}


// ---------------------------------------------------------
// Read the ./datafiles/input_screening.dat file
// ---------------------------------------------------------
void  DS50lowmass::initScreening() {

    int i, j, ntot, index;
    double scale_tmp;

    if(! DS50lowmass::ScreeningInit) {
    
        if( DS50lowmass::debug) std::cout<<"DEBUG: "<< DS50lowmass::livetime<<std::endl;
        std::vector<std::string> pmtname     =  DS50lowmass::getValueFromScreening("pmt", "name");
        std::vector<std::string> pmtac       =  DS50lowmass::getValueFromScreening("pmt", "active");
        std::vector<std::string> pmtactivity =  DS50lowmass::getValueFromScreening("pmt", "activity");
        std::vector<std::string> pmtelements =  DS50lowmass::getValueFromScreening("pmt", "elements");
        std::vector<std::string> pmtmass     =  DS50lowmass::getValueFromScreening("pmt", "mass");
        std::vector<std::string> pmtefff     =  DS50lowmass::getValueFromScreening("pmt", "efff");

        ntot = int(pmtname.size());
        for(i=0;i<ntot;i++) {
            if(std::stoi(pmtac[i])==1) {
                 DS50lowmass::pmt_names.push_back(pmtname[i]);

                index = -1;
                for(j=0; j<20; j++) {
                    if(pmtname[i] == norms_names[j]) {
                        index = j;
                    }
                }

                scale_tmp =  DS50lowmass::livetime *
                            std::stod(pmtactivity[i]) *
                            std::stod(pmtelements[i]) *
                            std::stod(pmtmass[i]) *
                            std::stod(pmtefff[i]) * 1.0e-3;
                
                 DS50lowmass::pmt_scale.push_back(scale_tmp);
                
                if( DS50lowmass::debug) std::cout << "DEBUG: "<<pmtname[i]<<"   "<<scale_tmp<<std::endl;
                
            }
        }
         DS50lowmass::Npmt =  DS50lowmass::pmt_names.size();

         
    
        std::vector<std::string> cryoname     =  DS50lowmass::getValueFromScreening("cryo", "name");
        std::vector<std::string> cryoac       =  DS50lowmass::getValueFromScreening("cryo", "active");
        std::vector<std::string> cryoactivity =  DS50lowmass::getValueFromScreening("cryo", "activity");
        std::vector<std::string> cryoelements =  DS50lowmass::getValueFromScreening("cryo", "elements");
        std::vector<std::string> cryomass     =  DS50lowmass::getValueFromScreening("cryo", "mass");
        std::vector<std::string> cryoefff     =  DS50lowmass::getValueFromScreening("cryo", "efff");

        ntot = int(cryoname.size());
        for(i=0;i<ntot;i++) {
            if(std::stoi(cryoac[i])==1) {
                 DS50lowmass::cryo_names.push_back(cryoname[i]);

                index = -1;
                for(j=0; j<20; j++) {
                    if(cryoname[i] == norms_names[j]) {
                        index = j;
                    }
                }

                scale_tmp =  DS50lowmass::livetime *
                            std::stod(cryoactivity[i]) *
                            std::stod(cryoelements[i]) *
                            std::stod(cryomass[i]) *
                            std::stod(cryoefff[i]) * 1.0e-3;
                
                 DS50lowmass::cryo_scale.push_back(scale_tmp);
                
                
                if( DS50lowmass::debug) std::cout << "DEBUG: "<<cryoname[i]<<"   "<<scale_tmp<<std::endl;
            }
        }
         DS50lowmass::Ncryo =  DS50lowmass::cryo_names.size();

        std::vector<std::string> ar39name     =  DS50lowmass::getValueFromScreening("ar39n", "name");
        std::vector<std::string> ar39ac       =  DS50lowmass::getValueFromScreening("ar39n", "active");
        std::vector<std::string> ar39activity =  DS50lowmass::getValueFromScreening("ar39n", "activity");
        std::vector<std::string> ar39elements =  DS50lowmass::getValueFromScreening("ar39n", "elements");
        std::vector<std::string> ar39mass     =  DS50lowmass::getValueFromScreening("ar39n", "mass");
        std::vector<std::string> ar39efff     =  DS50lowmass::getValueFromScreening("ar39n", "efff");

         DS50lowmass::ar39_scale =  DS50lowmass::livetime *
                              std::stod(ar39activity[0]) *
                              std::stod(ar39elements[0]) *
                              std::stod(ar39mass[0]) *
                              std::stod(ar39efff[0]) * 1.0e-3;

        
        if( DS50lowmass::debug) std::cout << "DEBUG: ar39n   "<< DS50lowmass::ar39_scale<<std::endl;

        std::vector<std::string> kr85name     =  DS50lowmass::getValueFromScreening("kr85n", "name");
        std::vector<std::string> kr85ac       =  DS50lowmass::getValueFromScreening("kr85n", "active");
        std::vector<std::string> kr85activity =  DS50lowmass::getValueFromScreening("kr85n", "activity");
        std::vector<std::string> kr85elements =  DS50lowmass::getValueFromScreening("kr85n", "elements");
        std::vector<std::string> kr85mass     =  DS50lowmass::getValueFromScreening("kr85n", "mass");
        std::vector<std::string> kr85efff     =  DS50lowmass::getValueFromScreening("kr85n", "efff");

         DS50lowmass::kr85_scale =  DS50lowmass::livetime *
                              std::stod(kr85activity[0]) *
                              std::stod(kr85elements[0]) *
                              std::stod(kr85mass[0]) *
                              std::stod(kr85efff[0]) * 1.0e-3;
                              
        if( DS50lowmass::debug) std::cout << "DEBUG: kr85n   "<< DS50lowmass::kr85_scale<<std::endl;



         DS50lowmass::ScreeningInit = true;
    }

}

// ---------------------------------------------------------
// Get WIMP NR signal theoretical spectrum (y and x values)
// ---------------------------------------------------------
void  DS50lowmass::getWIMP_theo(std::string mass) {

    int i, j;

     DS50lowmass::MW = mass;
    
    
    if(! DS50lowmass::WIMP_init) {
        for(i=0; i< DS50lowmass::Nch; i++) {
            std::stringstream filename;
            filename << "./datafiles/th_spectra/wimp_"<<mass<<"_spectrum_theo_"<< DS50lowmass::active_ch[i]<<"_new.npy";
            cnpy::NpyArray arr = cnpy::npy_load(filename.str());

            double length = arr.shape[0];
            double *loaded_data = arr.data<double>();

            std::vector<double> y_ch_i;

            for(j=0;j<length;j++) {
                y_ch_i.push_back(loaded_data[j]);
            }

             DS50lowmass::y_WIMP.push_back(y_ch_i);
        }

        std::stringstream filename;
        filename << "./datafiles/th_spectra/wimp_"<<mass<<"_spectrum_theo_x_new.npy";
        cnpy::NpyArray arr = cnpy::npy_load(filename.str());
        double length = arr.shape[0];
        double *loaded_data = arr.data<double>();
        std::vector<double> y_ch_i;
        for(j=0;j<length;j++) {
            y_ch_i.push_back(loaded_data[j]);
        }
         DS50lowmass::x_WIMP = y_ch_i;
        

        if( DS50lowmass::debug) {
            std::cout<<"DEBUG "<< DS50lowmass::y_WIMP.size()<<"  "<< DS50lowmass::y_WIMP[0].size()<<std::endl;
        }

         DS50lowmass::WIMP_init = true;

    }
}

// ---------------------------------------------------------
// Get Migdal signal theoretical spectrum
// ---------------------------------------------------------
void  DS50lowmass::getMigdal(std::string mass) {

    int i, j, k;
    
    int lengthA, lengthB;
    
    if (! DS50lowmass::y_migInit) {
        for(i=0; i< DS50lowmass::Nch;i++) {
            std::stringstream filename;
            filename << "./datafiles/migdal_all/migdal_"<<mass<<".npy";
            cnpy::NpyArray arr = cnpy::npy_load(filename.str());

            lengthA = arr.shape[0];
            lengthB = arr.shape[1];
            
            double *loaded_data = arr.data<double>();

            std::vector<std::vector<double>> mig_ch_i;
            
            for(j=0;j<lengthA;j++) {
                std::vector<double> line;
                for(k=0;k<lengthB;k++) {
                    line.push_back(loaded_data[j+k*lengthA]/7.0);
                }
                mig_ch_i.push_back(line);
            }
            DS50lowmass::y_migdal.push_back(mig_ch_i);
            
        }
        
         DS50lowmass::y_migInit = true;
        
    }
}


// ---------------------------------------------------------
// Get Ar39 theoretical spectrum
// ---------------------------------------------------------
void  DS50lowmass::getar39_theo() {

    int i, j;
    
    if (! DS50lowmass::y_ar39Init) {
        for(i=0; i< DS50lowmass::Nch;i++) {
            std::stringstream filename;
            filename << "./datafiles/th_spectra/ar39n_spectrum_theo_"<< DS50lowmass::active_ch[i]<<".npy";
            cnpy::NpyArray arr = cnpy::npy_load(filename.str());

            double length = arr.shape[0];
            double *loaded_data = arr.data<double>();

            std::vector<double> y_ch_i;

            for(j=0;j<length;j++) {
                y_ch_i.push_back(loaded_data[j]);
            }

             DS50lowmass::y_ar39.push_back(y_ch_i);
        }

        if( DS50lowmass::debug) {
            std::cout<<"DEBUG "<< DS50lowmass::y_ar39.size()<<"  "<< DS50lowmass::y_ar39[0].size()<<std::endl;
        }
        
         DS50lowmass::y_ar39Init = true;
    }
}


// ---------------------------------------------------------
// Get Kr85 theoretical spectrum
// ---------------------------------------------------------
void  DS50lowmass::getkr85_theo() {

    int i, j;
    
    if (! DS50lowmass::y_kr85Init) {
        for(i=0; i< DS50lowmass::Nch;i++) {
            std::stringstream filename;
            filename << "./datafiles/th_spectra/kr85n_spectrum_theo_"<< DS50lowmass::active_ch[i]<<".npy";
            cnpy::NpyArray arr = cnpy::npy_load(filename.str());

            double length = arr.shape[0];
            double *loaded_data = arr.data<double>();

            std::vector<double> y_ch_i;

            for(j=0;j<length;j++) {
                y_ch_i.push_back(loaded_data[j]);
            }

             DS50lowmass::y_kr85.push_back(y_ch_i);
            
        }

        if( DS50lowmass::debug) {
            std::cout<<"DEBUG "<< DS50lowmass::y_kr85.size()<<"  "<< DS50lowmass::y_kr85[1].size()<<std::endl;
        }

         DS50lowmass::y_kr85Init = true;
    }

}


// ---------------------------------------------------------
// Get Ar39 Q-value uncertainty
// ---------------------------------------------------------
void  DS50lowmass::getar39_Q() {

    int j;
    
    std::stringstream filename;
    filename << "./datafiles/th_spectra/ar39n_Qrel.npy";
    cnpy::NpyArray arr = cnpy::npy_load(filename.str());

    double length = arr.shape[0];
    double *loaded_data = arr.data<double>();


    for(j=0;j<length;j++) {
         DS50lowmass::y_ar39_Q.push_back(loaded_data[j]);
    }

       
}

// ---------------------------------------------------------
// Get kr85 Q-value uncertainty
// ---------------------------------------------------------
void  DS50lowmass::getkr85_Q() {

    int j;
    
    std::stringstream filename;
    filename << "./datafiles/th_spectra/kr85n_Qrel.npy";
    cnpy::NpyArray arr = cnpy::npy_load(filename.str());

    double length = arr.shape[0];
    double *loaded_data = arr.data<double>();

    for(j=0;j<length;j++) {
         DS50lowmass::y_kr85_Q.push_back(loaded_data[j]);
    }
       
}


// ---------------------------------------------------------
// Get Migdal M2 Matrix
// ---------------------------------------------------------
void  DS50lowmass::getMigdal_SM() {

    int i,j,k;
    int lengthA, lengthB;

    if (! DS50lowmass::y_migInit_SM) {
        for(i=0; i< DS50lowmass::Nch;i++) {
            std::stringstream filename;
            //  AR39 and Migdal uniformly distributed then M2_Migdal = M2_Ar39
            if(irebin) {
                filename << "./datafiles/smearing_matrices/ar39n_400_"<< DS50lowmass::active_ch[i]<<"_rebin.npy";
            } else {
                throw std::runtime_error("Unavailable updated M2 matrices for not-rebinned configuration.");
            }
            
            cnpy::NpyArray arr = cnpy::npy_load(filename.str());

            lengthA = arr.shape[0];
            lengthB = arr.shape[1];
            
            double *loaded_data = arr.data<double>();

            std::vector<std::vector<double>> SM_ch_i;

            for(j=0;j<lengthA;j++) {
                std::vector<double> line;
                for(k=0;k<lengthB;k++) {
                    line.push_back(loaded_data[k+j*lengthB]);
                }
                SM_ch_i.push_back(line);
            }
             DS50lowmass::SM_migdal.push_back(SM_ch_i);
        }

         DS50lowmass::y_migInit_SM = true;
    }

}


// ---------------------------------------------------------
// Get WIMP NR M2 Matrix
// ---------------------------------------------------------
void  DS50lowmass::getWIMP_SM() {

    int i,j,k;
    int lengthA, lengthB;

    if (! DS50lowmass::WIMP_SM_init) {
        for(i=0; i< DS50lowmass::Nch;i++) {
            std::stringstream filename;
            if(irebin) {
                filename << "./datafiles/smearing_matrices/NR_400_"<< DS50lowmass::active_ch[i]<<"_rebin.npy";
            } else {
                throw std::runtime_error("Unavailable updated M2 matrices for not-rebinned configuration.");
            }
           
            cnpy::NpyArray arr = cnpy::npy_load(filename.str());

            lengthA = arr.shape[0];
            lengthB = arr.shape[1];
            
            double *loaded_data = arr.data<double>();

            std::vector<std::vector<double>> SM_ch_i;

            for(j=0;j<lengthA;j++) {
                std::vector<double> line;
                for(k=0;k<lengthB;k++) {
                    line.push_back(loaded_data[k+j*lengthB]);
                }
                SM_ch_i.push_back(line);
            }
             DS50lowmass::SM_WIMP.push_back(SM_ch_i);
        }

         DS50lowmass::WIMP_SM_init = true;
    }

}


// ---------------------------------------------------------
// Get Ar39 M2 Matrix
// ---------------------------------------------------------
void  DS50lowmass::getar39_SM() {

    int i,j,k;
    int lengthA, lengthB;

    if (! DS50lowmass::y_ar39Init_SM) {
        for(i=0; i< DS50lowmass::Nch;i++) {
            std::stringstream filename;
            if(irebin) {
                filename << "./datafiles/smearing_matrices/ar39n_400_"<< DS50lowmass::active_ch[i]<<"_rebin.npy";
            } else {
                throw std::runtime_error("Unavailable updated M2 matrices for not-rebinned configuration.");
            }
            cnpy::NpyArray arr = cnpy::npy_load(filename.str());

            lengthA = arr.shape[0];
            lengthB = arr.shape[1];
            
            double *loaded_data = arr.data<double>();

            std::vector<std::vector<double>> SM_ch_i;

            for(j=0;j<lengthA;j++) {
                std::vector<double> line;
                for(k=0;k<lengthB;k++) {
                    line.push_back(loaded_data[k+j*lengthB]);
                }
                SM_ch_i.push_back(line);
            }
             DS50lowmass::SM_ar39.push_back(SM_ch_i);
        }

         DS50lowmass::y_ar39Init_SM = true;
    }

}

// ---------------------------------------------------------
// Get Kr85 M2 Matrix
// ---------------------------------------------------------
void  DS50lowmass::getkr85_SM() {

    int i,j,k;
    int lengthA, lengthB;

    if (! DS50lowmass::y_kr85Init_SM) {
        for(i=0; i< DS50lowmass::Nch;i++) {
            std::stringstream filename;
            if(irebin) {
                filename << "./datafiles/smearing_matrices/kr85n_400_"<< DS50lowmass::active_ch[i]<<"_rebin.npy";
            } else {
                throw std::runtime_error("Unavailable updated M2 matrices for not-rebinned configuration.");
            }
            cnpy::NpyArray arr = cnpy::npy_load(filename.str());

            lengthA = arr.shape[0];
            lengthB = arr.shape[1];
            
            double *loaded_data = arr.data<double>();

            std::vector<std::vector<double>> SM_ch_i;

            for(j=0;j<lengthA;j++) {
                std::vector<double> line;
                for(k=0;k<lengthB;k++) {
                    line.push_back(loaded_data[k+j*lengthB]);
                }
                SM_ch_i.push_back(line);
            }
             DS50lowmass::SM_kr85.push_back(SM_ch_i);
        }

         DS50lowmass::y_kr85Init_SM = true;
    }

}


// ---------------------------------------------------------
// Get PMT theoretical spectrum
// ---------------------------------------------------------
void  DS50lowmass::getpmt_theo() {

    int i, j, k;
    
    if (! DS50lowmass::y_pmtInit) {
        for(i=0;i< DS50lowmass::Npmt;i++) {
            std::vector<std::vector<double>> y_pmt_i;
            for(j=0; j< DS50lowmass::Nch;j++) {
                std::stringstream filename;
                
                filename << "./datafiles/th_spectra/"<< DS50lowmass::pmt_names[i]<<"_spectrum_theo_"<< DS50lowmass::active_ch[j]<<"_new.npy";
                
                cnpy::NpyArray arr = cnpy::npy_load(filename.str());

                double length = arr.shape[0];
                double *loaded_data = arr.data<double>();

                std::vector<double> y_ch_j;

                for(k=0;k<length;k++) {
                    y_ch_j.push_back(loaded_data[k]);
                }

                y_pmt_i.push_back(y_ch_j);
            }
             DS50lowmass::y_pmt.push_back(y_pmt_i);
        }

         DS50lowmass::y_pmtInit = true;
    }

}

// ---------------------------------------------------------
// Get PMT M2 Matrix
// ---------------------------------------------------------
void  DS50lowmass::getpmt_SM() {
    int i,j,k,l;
    int lengthA, lengthB;

    if (! DS50lowmass::y_pmtInit_SM) {
        for(i=0;i< DS50lowmass::Npmt;i++) {
            std::vector<std::vector<std::vector<double>>> SM_pmt_i;
            for(j=0; j< DS50lowmass::Nch;j++) {
                std::stringstream filename;
                if(irebin) {
                    filename << "./datafiles/smearing_matrices/"<< DS50lowmass::pmt_names[i]<<"_400_"<< DS50lowmass::active_ch[j]<<"_rebin.npy";
                } else {
                    throw std::runtime_error("Unavailable updated M2 matrices for not-rebinned configuration.");
                }
                cnpy::NpyArray arr = cnpy::npy_load(filename.str());

                lengthA = arr.shape[0];
                lengthB = arr.shape[1];

                double *loaded_data = arr.data<double>();

                std::vector<std::vector<double>> SM_ch_j;

                for(k=0;k<lengthA;k++) {
                    std::vector<double> line;
                    for(l=0;l<lengthB;l++) {
                        line.push_back(loaded_data[l+k*lengthB]);
                    }
                    SM_ch_j.push_back(line);
                }
                SM_pmt_i.push_back(SM_ch_j);
            }
             DS50lowmass::SM_pmt.push_back(SM_pmt_i);
        }
        
         DS50lowmass::y_pmtInit_SM = true;
    }
}

// ---------------------------------------------------------
// Get Cryostats theoretical spectrum
// ---------------------------------------------------------
void  DS50lowmass::getcryo_theo() {

    int i, j, k;
    
    if (! DS50lowmass::y_cryoInit) {
        for(i=0;i< DS50lowmass::Ncryo;i++) {
            std::vector<std::vector<double>> y_cryo_i;
            for(j=0; j< DS50lowmass::Nch;j++) {
                std::stringstream filename;
                
                filename << "./datafiles/th_spectra/"<< DS50lowmass::cryo_names[i]<<"_spectrum_theo_"<< DS50lowmass::active_ch[j]<<"_new.npy";
                
                cnpy::NpyArray arr = cnpy::npy_load(filename.str());

                double length = arr.shape[0];
                double *loaded_data = arr.data<double>();

                std::vector<double> y_ch_j;

                for(k=0;k<length;k++) {
                    y_ch_j.push_back(loaded_data[k]);
                }

                y_cryo_i.push_back(y_ch_j);
            }
             DS50lowmass::y_cryo.push_back(y_cryo_i);
        }

         DS50lowmass::y_cryoInit = true;
    }

}

// ---------------------------------------------------------
// Get Cryostats M2 Matrix
// ---------------------------------------------------------
void  DS50lowmass::getcryo_SM() {
    int i,j,k,l;
    int lengthA, lengthB;

    if (! DS50lowmass::y_cryoInit_SM) {
        for(i=0;i< DS50lowmass::Ncryo;i++) {
            std::vector<std::vector<std::vector<double>>> SM_cryo_i;
            for(j=0; j< DS50lowmass::Nch;j++) {
                std::stringstream filename;
                if(irebin) {
                    filename << "./datafiles/smearing_matrices/"<< DS50lowmass::cryo_names[i]<<"_400_"<< DS50lowmass::active_ch[j]<<"_rebin.npy";
                } else {
                    throw std::runtime_error("Unavailable updated M2 matrices for not-rebinned configuration.");
                }
                cnpy::NpyArray arr = cnpy::npy_load(filename.str());

                lengthA = arr.shape[0];
                lengthB = arr.shape[1];

                double *loaded_data = arr.data<double>();

                std::vector<std::vector<double>> SM_ch_j;

                for(k=0;k<lengthA;k++) {
                    std::vector<double> line;
                    for(l=0;l<lengthB;l++) {
                        line.push_back(loaded_data[l+k*lengthB]);
                    }
                    SM_ch_j.push_back(line);
                }
                SM_cryo_i.push_back(SM_ch_j);
            }
             DS50lowmass::SM_cryo.push_back(SM_cryo_i);
        }
        
         DS50lowmass::y_cryoInit_SM = true;
    }
}

// ---------------------------------------------------------
// Get x values for the ER theretical spectra
// ---------------------------------------------------------
void  DS50lowmass::getx_theo() {

    int i;
    
    if (! DS50lowmass::x_theoInit) {

        std::stringstream filename;
        filename << "./datafiles/th_spectra/x_theo_bins.npy";
        cnpy::NpyArray arr = cnpy::npy_load(filename.str());

        double length = arr.shape[0];
        double *loaded_data = arr.data<double>();

        double bs;
        
        for(i=0;i<(length-1);i++) {
            bs = loaded_data[i+1] - loaded_data[i];
             DS50lowmass::x_theo.push_back(loaded_data[i+1] - 0.5 * bs);
        }

        if( DS50lowmass::debug) {
            std::cout<<"DEBUG "<< DS50lowmass::x_theo.size()<<"  "<< DS50lowmass::x_theo[0]<<std::endl;
        }
        
         DS50lowmass::x_theoInit = true;
    }
}

// ---------------------------------------------------------
// Get data
// ---------------------------------------------------------
void  DS50lowmass::getdata() {

    std::string x,y,thrash;
    
    if (! DS50lowmass::dataInit) {
        std::ifstream inFile;
        if(irebin) {
            if(expected) {
                inFile.open("./datafiles/data/latest_asimov.txt");
            } else {
                inFile.open("./datafiles/data/data_rebinned_latest.txt");// data
            }
        } else {
            throw std::runtime_error("Unavailable updated M2 matrices for not-rebinned configuration.");
        }
        
        while ( !inFile.eof () ) {
            
            inFile>> x;
            inFile>> y;
            inFile>> thrash;

            double exp_factor = 1.;
	    
            DS50lowmass::y_data.push_back(std::stod(y)*exp_factor);
        }
        
         DS50lowmass::dataInit = true;
    }
}

void  DS50lowmass::getPeakEff() {
    std::string line, word;
    std::vector<double> row;
    std::ifstream myfile("./datafiles/peak_time_eff_central.txt");
    if (myfile.is_open()) {
        while (getline(myfile,line)) {
            row.clear();
            std::stringstream str(line);
            int c = 0;
            while(std::getline(str, word, '\t')) {
                    if(c!=0) {
                        peak_eff_central.push_back(std::stod(word));
                    }
                    c++;
                }
        }
        myfile.close();
    } else {
        throw std::runtime_error("Unable to open file './datafiles/peak_time_eff_central.txt'");
    }

    std::ifstream myfile2("./datafiles/peak_time_eff_ring.txt");
    if (myfile2.is_open()) {
        while (getline(myfile2,line)) {
            row.clear();
            std::stringstream str(line);
            int c = 0;
            while(std::getline(str, word, '\t')) {
                    if(c!=0) {
                        peak_eff_ring.push_back(std::stod(word));
                    }
                    c++;
                }
        }
        myfile2.close();
    } else {
        throw std::runtime_error("Unable to open file './datafiles/peak_time_eff_central.txt'");
    }
    
}



void  DS50lowmass::getNormNR() {
    std::string line, word;
    std::vector<double> row;
    std::ifstream myfile("./datafiles/wimp_norms.txt");
    if (myfile.is_open()) {
        while (getline(myfile,line)) {
            row.clear();
            std::stringstream str(line);
            int c = 0;
            while(std::getline(str, word, '\t')) {
                    if(c!=0) {
		      norm_nr.push_back(std::stod(word));
                    }
                    c++;
                }
        }
        myfile.close();
    } else {
        throw std::runtime_error("Unable to open file './datafiles/wimp_norms.txt'");
    }
}





//==========================================================

// ---------------------------------------------------------
// Initialization of the GPU variables
// ---------------------------------------------------------
void  DS50lowmass::init_cuda() {
    int dimA, dimB, dimC;
    int l, k;
    int i, j;
    
    dimA = int( DS50lowmass::SM_ar39[0].size());
    dimB = int( DS50lowmass::y_ar39[0].size());
    dimC = int( DS50lowmass::x_WIMP.size());
   
    ///////// tmp mask vectors for sigma_scr and Q-val implementation /////////
    yAr_00=(float **)malloc(Nch*sizeof(float *));
    yAr_01=(float **)malloc(Nch*sizeof(float *));
    yAr_02=(float **)malloc(Nch*sizeof(float *));
    yKr_00=(float **)malloc(Nch*sizeof(float *));
    yKr_01=(float **)malloc(Nch*sizeof(float *));
    yKr_02=(float **)malloc(Nch*sizeof(float *));
    
    
    cAr_00=(float **)malloc(Nch*sizeof(float *));
    cAr_01=(float **)malloc(Nch*sizeof(float *));
    cAr_02=(float **)malloc(Nch*sizeof(float *));
    cKr_00=(float **)malloc(Nch*sizeof(float *));
    cKr_01=(float **)malloc(Nch*sizeof(float *));
    cKr_02=(float **)malloc(Nch*sizeof(float *));
    /////////
    
        
    yar39ch=(float **)malloc(Nch*sizeof(float *));
    car39ch=(float **)malloc(Nch*sizeof(float *));
    
    ykr85ch=(float **)malloc(Nch*sizeof(float *));
    ckr85ch=(float **)malloc(Nch*sizeof(float *));
    
    ypmtch =(float **)malloc(Nch*Npmt*sizeof(float *));
    cpmtch =(float **)malloc(Nch*Npmt*sizeof(float *));
    
    ycryoch =(float **)malloc(Nch*Ncryo*sizeof(float *));
    ccryoch =(float **)malloc(Nch*Ncryo*sizeof(float *));
    
    ywimpch=(float **)malloc(Nch*sizeof(float *));
    cwimpch=(float **)malloc(Nch*sizeof(float *));
    
    
    for(l=0;l<Nch;l++) {
        yar39ch[l] = (float *)malloc (dimB*sizeof(float));
        /////////
        yAr_00[l] = (float *)malloc (dimB*sizeof(float));
        yAr_01[l] = (float *)malloc (dimB*sizeof(float));
        yAr_02[l] = (float *)malloc (dimB*sizeof(float));
        /////////
        for(j=0;j<dimB;j++) {
            yar39ch[l][j]=(float)y_ar39[l][j];
            //////////
            if(j<=20) {
                yAr_00[l][j] = 0.1 * (float)y_ar39[l][j] / 2.8;
            } else {
                yAr_00[l][j] = 0.0 * (float)y_ar39[l][j];
            }
            yAr_01[l][j] = 0.0 * (float)y_ar39[l][j];
            yAr_02[l][j] = (float)y_ar39_Q[j];
            //////////
        }
    }
    
    ykr85ch=(float **)malloc(Nch*sizeof(float *));
    ckr85ch=(float **)malloc(Nch*sizeof(float *));
    for(l=0;l<Nch;l++) {
        ykr85ch[l] = (float *)malloc (dimB*sizeof(float));
        /////////
        yKr_00[l] = (float *)malloc (dimB*sizeof(float));
        yKr_01[l] = (float *)malloc (dimB*sizeof(float));
        yKr_02[l] = (float *)malloc (dimB*sizeof(float));
        /////////
        for(j=0;j<dimB;j++) {
            ykr85ch[l][j]=(float)y_kr85[l][j];
            //////////
            if(j<=20) {
                yKr_00[l][j] = 0.1 * (float)y_kr85[l][j] / 2.8;
            } else {
                yKr_00[l][j] = 0.0 * (float)y_kr85[l][j];
            }
            yKr_01[l][j] = 0.0 * (float)y_kr85[l][j];
            yKr_02[l][j] = (float)y_kr85_Q[j];
            //////////
        }
    }
    
    
    for(l=0;l<Nch;l++) {
        for(k=0;k<Npmt;k++) {
            ypmtch[k*Nch+l]   = (float *)malloc(dimB*sizeof(float));
            for(j=0;j<dimB;j++) {
                ypmtch[k*Nch+l][j]  = (float)y_pmt[k][l][j];
            }
        }
        for(k=0;k<Ncryo;k++) {
            ycryoch[k*Nch+l] = (float *)malloc(dimB*sizeof(float));
            for(j=0;j<dimB;j++) {
                ycryoch[k*Nch+l][j]  = (float)y_cryo[k][l][j];
            }
        }
    }
    
    
    for(l=0;l<Nch;l++) {
        ywimpch[l] = (float *)malloc (dimC*sizeof(float));
        for(j=0;j<dimC;j++) {
            ywimpch[l][j]=(float)y_WIMP[l][j];
        }
    }
                              
    cM1     =(float*)malloc(dimA*dimB*sizeof(float));
    cM1mig  =(float*)malloc(dimA*dimB*sizeof(float));
    cM1NR   =(float*)malloc(dimA*dimC*sizeof(float));
    
    for(i=0;i<7;i++) {
        cache_pars.push_back(0.0);
    }
    for(i=0;i<dimA;i++) {
        std::vector<double> tmpM1;
        for(j=0;j<dimB;j++) {
            tmpM1.push_back(0.0);
        }
        cache_M1.push_back(tmpM1);
        cache_M1mig.push_back(tmpM1);
        std::vector<double> tmpM1NR;
        for(j=0;j<dimC;j++) {
            tmpM1NR.push_back(0.0);
        }
        cache_M1NR.push_back(tmpM1NR);
    }
    
    
    ymigch=(float **)malloc(Nch*sizeof(float *));
    cmigch=(float **)malloc(Nch*sizeof(float *));
    for(l=0;l<Nch;l++) {
        ymigch[l]   = (float *)malloc (dimC*dimB*sizeof(float));
        for(i=0;i<dimC;i++) {
            for(j=0;j<dimB;j++) {
                (ymigch[l])[IDX2C(i, j, dimC)]  =  (float)y_migdal[l][i][j];
            }
        }
    }
    
    cSMWIMP = (float **)malloc (Nch*sizeof(float *));
    dSMWIMP = (float **)malloc (Nch*sizeof(float *));
    for(l=0;l<Nch;l++) {
        cSMWIMP[l] = (float *)malloc (dimA*dimA*sizeof(float));
        for(i=0;i<dimA;i++) {
            for(j=0;j<dimA;j++) {
                (cSMWIMP[l])[IDX2C(i, j, dimA)] = (float)SM_WIMP[l][i][j];
            }
        }
    }
    
    
    cSMmig = (float **)malloc (Nch*sizeof(float *));
    dSMmig = (float **)malloc (Nch*sizeof(float *));
    for(l=0;l<Nch;l++) {
        cSMmig[l] = (float *)malloc (dimA*dimA*sizeof(float));
        for(i=0;i<dimA;i++) {
            for(j=0;j<dimA;j++) {
                (cSMmig[l])[IDX2C(i, j, dimA)] = (float)SM_migdal[l][i][j];
            }
        }
    }
    
    
    cSMar39 = (float **)malloc (Nch*sizeof(float *));
    dSMar39 = (float **)malloc (Nch*sizeof(float *));
    for(l=0;l<Nch;l++) {
        cSMar39[l] = (float *)malloc (dimA*dimA*sizeof(float));
        for(i=0;i<dimA;i++) {
            for(j=0;j<dimA;j++) {
                (cSMar39[l])[IDX2C(i, j, dimA)] = (float)SM_ar39[l][i][j];
            }
        }
    }
    
    cSMkr85 = (float **)malloc (Nch*sizeof(float *));
    dSMkr85 = (float **)malloc (Nch*sizeof(float *));
    for(l=0;l<Nch;l++) {
        cSMkr85[l] = (float *)malloc (dimA*dimA*sizeof(float));
        for(i=0;i<dimA;i++) {
            for(j=0;j<dimA;j++) {
                (cSMkr85[l])[IDX2C(i, j, dimA)] = (float)SM_kr85[l][i][j];
            }
        }
    }
    
    cSMpmt = (float **)malloc (Nch*Npmt*sizeof(float *));
    dSMpmt = (float **)malloc (Nch*Npmt*sizeof(float *));
    for(l=0;l<Nch;l++) {
        for(k=0;k<Npmt;k++) {
            cSMpmt[k*Nch + l] = (float *)malloc (dimA*dimA*sizeof(float));
            for(i=0;i<dimA;i++) {
                for(j=0;j<dimA;j++) {
                    (cSMpmt[k*Nch + l])[IDX2C(i, j, dimA)] = (float)SM_pmt[k][l][i][j];
                }
            }
        }
    }
    
    cSMcryo = (float **)malloc (Nch*Ncryo*sizeof(float *));
    dSMcryo = (float **)malloc (Nch*Ncryo*sizeof(float *));
    for(l=0;l<Nch;l++) {
        for(k=0;k<Ncryo;k++) {
            cSMcryo[k*Nch + l] = (float *)malloc (dimA*dimA*sizeof(float));
            for(i=0;i<dimA;i++) {
                for(j=0;j<dimA;j++) {
                    (cSMcryo[k*Nch + l])[IDX2C(i, j, dimA)] = (float)SM_cryo[k][l][i][j];
                }
            }
        }
    }
    
    
    PeakEffCent_c  =(float*)malloc(dimA*dimA*sizeof(float));
    PeakEffRing_c  =(float*)malloc(dimA*dimA*sizeof(float));
    for(i=0;i<dimA;i++) {
        for(j=0; j<dimA;j++) {
            if(i==j) {
                if(i < (int)peak_eff_central.size()) {
                    PeakEffCent_c[IDX2C(i, j, dimA)] = (float)peak_eff_central[i];
                    PeakEffRing_c[IDX2C(i, j, dimA)] = (float)peak_eff_ring[i];
                } else {
                    PeakEffCent_c[IDX2C(i, j, dimA)] = (float)1.0;
                    PeakEffRing_c[IDX2C(i, j, dimA)] = (float)1.0;
                }
            } else {
                PeakEffCent_c[IDX2C(i, j, dimA)] = (float)0.0;
                PeakEffRing_c[IDX2C(i, j, dimA)] = (float)0.0;
            }
        }
    }
    
    
    tmp1ch = (float **)malloc (Nch*sizeof(float *));
    tmp2ch = (float **)malloc (Nch*sizeof(float *));
    tmp3ch = (float **)malloc (Nch*Npmt*sizeof(float *));
    tmp4ch = (float **)malloc (Nch*Ncryo*sizeof(float *));
    
    tmp5ch = (float **)malloc (Nch*sizeof(float *));
    tmp6ch = (float **)malloc (Nch*sizeof(float *));
    tmp7ch = (float **)malloc (Nch*sizeof(float *));
    tmp8ch = (float **)malloc (Nch*sizeof(float *));
    
    ///////////
    zerov = (float *)malloc(dimB*sizeof(float));
    for( j=0;j<dimB;j++){
        zerov[j] = 0.0;
    }
    cudaStat=cudaMalloc((void**)&czerov, dimB*sizeof(float));
    ///////////
    
    for(l=0;l<Nch;l++) {
        cudaStat=cudaMalloc((void**)&car39ch[l],dimB*sizeof(float));
        cudaStat=cudaMalloc((void**)&ckr85ch[l],dimB*sizeof(float));
        ////////
        cudaStat=cudaMalloc((void**)&cAr_00[l],dimB*sizeof(float));
        cudaStat=cudaMalloc((void**)&cAr_01[l],dimB*sizeof(float));
        cudaStat=cudaMalloc((void**)&cAr_02[l],dimB*sizeof(float));
        
        cudaStat=cudaMalloc((void**)&cKr_00[l],dimB*sizeof(float));
        cudaStat=cudaMalloc((void**)&cKr_01[l],dimB*sizeof(float));
        cudaStat=cudaMalloc((void**)&cKr_02[l],dimB*sizeof(float));
        ////////
        cudaStat=cudaMalloc((void**)&cwimpch[l],dimC*sizeof(float));
        cudaStat=cudaMalloc((void**)&dSMWIMP[l],dimA*dimA*sizeof(float));
        cudaStat=cudaMalloc((void**)&dSMar39[l],dimA*dimA*sizeof(float));
        cudaStat=cudaMalloc((void**)&dSMkr85[l],dimA*dimA*sizeof(float));
        cudaStat=cudaMalloc((void**)&tmp1ch[l] ,dimA*sizeof(float));
        cudaStat=cudaMalloc((void**)&tmp2ch[l] ,dimA*sizeof(float));
        cudaStat=cudaMalloc((void**)&tmp5ch[l] ,dimA*sizeof(float));
        
        
        cudaStat=cudaMalloc((void**)&cmigch[l], dimC*dimB*sizeof(float));
        cudaStat=cudaMalloc((void**)&dSMmig[l] ,dimA*dimA*sizeof(float));
        
        
        
        cudaStat=cudaMalloc((void**)&tmp6ch[l] ,dimC*dimA*sizeof(float));
        cudaStat=cudaMalloc((void**)&tmp7ch[l] ,dimA*dimA*sizeof(float));
        cudaStat=cudaMalloc((void**)&tmp8ch[l] ,dimA*sizeof(float));
        
        for(k=0;k<Npmt;k++) {
            cudaStat=cudaMalloc((void**)&cpmtch[k*Nch+l], dimB*sizeof(float));
            cudaStat=cudaMalloc((void**)&tmp3ch[k*Nch+l], dimA*sizeof(float));
            cudaStat=cudaMalloc((void**)&dSMpmt[k*Nch+l], dimA*dimA*sizeof(float));
        }
        for(k=0;k<Ncryo;k++) {
            cudaStat=cudaMalloc((void**)&ccryoch[k*Nch+l], dimB*sizeof(float));
            cudaStat=cudaMalloc( (void**)&tmp4ch[k*Nch+l], dimA*sizeof(float));
            cudaStat=cudaMalloc((void**)&dSMcryo[k*Nch+l], dimA*dimA*sizeof(float));
        }
        
    }
    
    cudaStat=cudaMalloc((void**)&totar, dimA*sizeof(float));
    cudaStat=cudaMalloc((void**)&totkr, dimA*sizeof(float));
    cudaStat=cudaMalloc((void**)&totpmt, dimA*sizeof(float));
    cudaStat=cudaMalloc((void**)&totcryo, dimA*sizeof(float));
    cudaStat=cudaMalloc((void**)&totwimp, dimA*sizeof(float));
    cudaStat=cudaMalloc((void**)&totmig, dimA*sizeof(float));
    cudaStat=cudaMalloc((void**)&dM1,dimA*dimB*sizeof(*cM1));
    cudaStat=cudaMalloc((void**)&dM1mig,dimA*dimB*sizeof(*cM1mig));
    cudaStat=cudaMalloc((void**)&M1_tr, dimB*dimA*sizeof(*cM1));
    cudaStat=cudaMalloc((void**)&M1mig_tr, dimB*dimA*sizeof(*cM1));
    
    cudaStat=cudaMalloc((void**)&PeakEffCent_d, dimA*dimA*sizeof(*PeakEffCent_c));
    cudaStat=cudaMalloc((void**)&PeakEffRing_d, dimA*dimA*sizeof(*PeakEffRing_c));
    
    cudaStat=cudaMalloc((void**)&dM1NR,dimA*dimC*sizeof(*cM1NR));
    stat = cublasCreate(&handle);
    
    float* tmp_AB;
    tmp_AB = (float *)malloc(dimB*dimA*sizeof(float));
    for(i = 0;i<dimA*dimB;i++) {
        tmp_AB[i] = (float)0.0;
    }
    float* tmp_CA;
    tmp_CA= (float *)malloc(dimA*dimC*sizeof(float));
    for(i = 0;i<dimA*dimC;i++) {
        tmp_CA[i] = (float)0.0;
    }
    float* tmp_AA;
    tmp_AA= (float *)malloc(dimA*dimA*sizeof(float));
    for(i = 0;i<dimA*dimA;i++) {
        tmp_AA[i] = (float)0.0;
    }
    
    stat = cublasSetMatrix(dimB, dimA, sizeof(float), tmp_AB, dimB, M1_tr, dimB);
    stat = cublasSetMatrix(dimB, dimA, sizeof(float), tmp_AB, dimB, M1mig_tr, dimB);
    
    stat = cublasSetMatrix(dimA, dimA, sizeof(float), PeakEffCent_c, dimA, PeakEffCent_d, dimA);
    stat = cublasSetMatrix(dimA, dimA, sizeof(float), PeakEffRing_c, dimA, PeakEffRing_d, dimA);
    
    ////////
    stat = cublasSetVector(dimB,sizeof(float),zerov,1,czerov,1);
    ////////
    for(l=0;l<Nch;l++) {
        stat = cublasSetMatrix(dimC, dimA, sizeof(float), tmp_CA, dimC, tmp6ch[l], dimC);
        stat = cublasSetMatrix(dimA, dimA, sizeof(float), tmp_AA, dimA, tmp7ch[l], dimA);
        
        stat = cublasSetVector(dimB,sizeof(float),yar39ch[l],1,car39ch[l],1);
        stat = cublasSetVector(dimB,sizeof(float),ykr85ch[l],1,ckr85ch[l],1);
        ///////
        stat = cublasSetVector(dimB,sizeof(float),yAr_00[l],1,cAr_00[l],1);
        stat = cublasSetVector(dimB,sizeof(float),yAr_01[l],1,cAr_01[l],1);
        stat = cublasSetVector(dimB,sizeof(float),yAr_02[l],1,cAr_02[l],1);
        stat = cublasSetVector(dimB,sizeof(float),yKr_00[l],1,cKr_00[l],1);
        stat = cublasSetVector(dimB,sizeof(float),yKr_01[l],1,cKr_01[l],1);
        stat = cublasSetVector(dimB,sizeof(float),yKr_02[l],1,cKr_02[l],1);
        ///////
        stat = cublasSetVector(dimC,sizeof(float),ywimpch[l],1,cwimpch[l],1);
        stat = cublasSetMatrix(dimA,dimA,sizeof(float),cSMWIMP[l],dimA,dSMWIMP[l],dimA);
        stat = cublasSetMatrix(dimA,dimA,sizeof(float),cSMar39[l],dimA,dSMar39[l],dimA);
        stat = cublasSetMatrix(dimA,dimA,sizeof(float),cSMkr85[l],dimA,dSMkr85[l],dimA);
        
        
        stat = cublasSetMatrix(dimC,dimB,sizeof(float),ymigch[l],dimC,cmigch[l],dimC);
        stat = cublasSetMatrix(dimA,dimA,sizeof(float),cSMmig[l],dimA,dSMmig[l],dimA);
    }
    
    for(l=0; l<Nch; l++) {
        for(k=0;k<Npmt;k++) {
            stat=cublasSetVector(dimB, sizeof(float), ypmtch[k*Nch+l], 1, cpmtch[k*Nch+l], 1);
            stat=cublasSetMatrix(dimA,dimA,sizeof(float), cSMpmt[k*Nch +l],dimA, dSMpmt[k*Nch+ l],dimA);
        }
    }
    for(l=0; l<Nch; l++) {
        for(k=0;k<Ncryo;k++) {
            stat=cublasSetVector(dimB, sizeof(float), ycryoch[k*Nch+l], 1, ccryoch[k*Nch+l], 1);
            stat=cublasSetMatrix(dimA,dimA,sizeof(float),cSMcryo[k*Nch+l],dimA,dSMcryo[k*Nch+l],dimA);
        }
    }
    
    
}










//==========================================================

// ---------------------------------------------------------
// Functions for matrices manipulation in LogPrior
// ---------------------------------------------------------



void  DS50lowmass::print_mat_contents(gsl_matrix *matrix)
{
    size_t i, j;
    double element;

    for (i = 0; i < 5; ++i) {
        for (j = 0; j < 5; ++j) {
            element = gsl_matrix_get(matrix, i, j);
            printf("%.9f ", element);
        }
        printf("\n");
    }
}




gsl_matrix * DS50lowmass::getFromDoubleArray(const double original_matrix[5][5])
{
    gsl_matrix *mat = gsl_matrix_alloc(5, 5);
    int i, j;

    for(i=0; i<5; i++) {
        for(j=0; j<5; j++) {
            gsl_matrix_set(mat, i, j,  original_matrix[i][j]);
        }
    }

    return mat;

}
gsl_matrix * DS50lowmass::getFromDoubleArray(const double original_matrix[2][2])
{
    gsl_matrix *mat = gsl_matrix_alloc(2, 2);
    int i, j;

    for(i=0; i<2; i++) {
        for(j=0; j<2; j++) {
            gsl_matrix_set(mat, i, j,  original_matrix[i][j]);
        }
    }

    return mat;

}

gsl_matrix * DS50lowmass::invert_a_matrix(gsl_matrix *matrix)
{
    gsl_permutation *p = gsl_permutation_alloc(5);
    int s;

    // Compute the LU decomposition of this matrix
    gsl_linalg_LU_decomp(matrix, p, &s);

    // Compute the  inverse of the LU decomposition
    gsl_matrix *inv = gsl_matrix_alloc(5, 5);
    gsl_linalg_LU_invert(matrix, p, inv);

    gsl_permutation_free(p);

    return inv;
}
gsl_matrix * DS50lowmass::invert_a_matrix2(gsl_matrix *matrix)
{
    gsl_permutation *p = gsl_permutation_alloc(2);
    int s;

    // Compute the LU decomposition of this matrix
    gsl_linalg_LU_decomp(matrix, p, &s);

    // Compute the  inverse of the LU decomposition
    gsl_matrix *inv = gsl_matrix_alloc(2, 2);
    gsl_linalg_LU_invert(matrix, p, inv);

    gsl_permutation_free(p);

    return inv;
}

double  DS50lowmass::matrix_det(gsl_matrix *matrix)
{
    gsl_permutation *p = gsl_permutation_alloc(5);
    int s;

    // Compute the LU decomposition of this matrix
    gsl_linalg_LU_decomp(matrix, p, &s);

    // Compute the determinant of the LU decomposition
    double det;
    det = gsl_linalg_LU_det(matrix, s);

    gsl_permutation_free(p);

    return det;
}
double  DS50lowmass::matrix_det2(gsl_matrix *matrix)
{
    gsl_permutation *p = gsl_permutation_alloc(2);
    int s;

    // Compute the LU decomposition of this matrix
    gsl_linalg_LU_decomp(matrix, p, &s);

    // Compute the determinant of the LU decomposition
    double det;
    det = gsl_linalg_LU_det(matrix, s);

    gsl_permutation_free(p);

    return det;
}


void  DS50lowmass::getMatrixFromGsl(gsl_matrix *matrix, double mat[5][5]) {
    int i,j;

    for(i=0;i<5;i++) {
        for(j=0;j<5;j++) {
            mat[i][j] = gsl_matrix_get(matrix, i, j);
        }
    }
}

void  DS50lowmass::getMatrixFromGsl(gsl_matrix *matrix, double mat[2][2]) {
    int i,j;

    for(i=0;i<2;i++) {
        for(j=0;j<2;j++) {
            mat[i][j] = gsl_matrix_get(matrix, i, j);
        }
    }
}

