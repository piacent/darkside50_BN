// ***************************************************************
// This file was created using the bat-project script
// for project darkside50_BN.
// bat-project is part of Bayesian Analysis Toolkit (BAT).
// BAT can be downloaded from http://mpp.mpg.de/bat
//
// Author: Stefano Piacentini
// Date: 03/02/2023
// ***************************************************************

#include <BAT/BCLog.h>
#include "DS50lowmass.h"

#include <chrono>
#include <stdexcept>

int main(int argc, char *argv[])
{
    
    // Initial input checks
    if(argc != 2) {
        throw std::invalid_argument("Wrong number of arguments.");
    }
    std::string mass(argv[1]);
    std::string allowed_m[35] = {"0.030", "0.063", "0.070", "0.079", "0.089", "0.100",
                                 "0.117", "0.137", "0.161",
                                 "0.189", "0.221", "0.259", "0.304", "0.356",
                                 "0.418", "0.489", "0.574", "0.672", "0.788",
                                 "0.924", "1.08", "1.27", "1.49", "1.74",
                                 "2.04", "2.40", "2.81", "3.29", "3.86",
                                 "4.52", "5.30", "6.21", "7.28", "8.53", "10.0"};
    int i;
    bool compare_masses = false;
    for(i=0; i<35; i++) {
        if(mass.compare(allowed_m[i]) == 0) {
            compare_masses = true;
        }
    }
    if(compare_masses) {
        std::cout<<"Correct mass input."<<std::endl;
    } else {
        throw std::invalid_argument("Wrong mass input.");
    }
    
    

    // N_e fitting window
    float Nmin, Nmax;
    Nmin = 4;
    Nmax = 170;

    int Nch, NIter;
    Nch = 12;          //number of parallel MCMC chains
    NIter = 10*10000;   //number of step per chain
    
    bool expected = true;

    // CHOOSING THE MODEL
    std::string model    = "expected_"+mass;
    std::string addinfos = "latest";
    std::string title    = "datafit";

    DS50lowmass m(title, Nmin, Nmax, Nch, mass, false, false, true, expected);
    
    //Creating ./results/ sub-directory
    std::string res_dir;
    res_dir = "./results_" + std::to_string(Nmin) +"-"+std::to_string(Nmax) +
              "_"+addinfos+"_"+model+"_"+std::to_string(Nch*NIter)+"/";

    int com = std::system(("mkdir "+res_dir).c_str());
    if(com != 0) {
        std::cout<<com<<std::endl;
    }

    // Creating Logfile
    BCLog::OpenLog(res_dir + title + "_log.txt", BCLog::detail, BCLog::detail);

    // Setting MCMC algorithm and precision
    m.SetMarginalizationMethod(BCIntegrate::kMargMetropolis);
    m.SetPrecision(BCEngineMCMC::kMedium);

    BCLog::OutSummary("Model created");

    // Setting prerun iterations to 10^8 (basically infinity)
    m.SetNIterationsPreRunMax(100000000);

    // Setting MC run iterations and number of parallel chains
    m.SetNIterationsRun(NIter);
    m.SetNChains(Nch);

    
    // SETTING STARTING POINT TO CALIBRATION VALUES 
    
    std::vector<double> x0;
    
    x0.push_back(1.0); //rB_Int
    x0.push_back(23.0);                //g2
    x0.push_back(2.49);                //p0
    x0.push_back(21.84);               //p1
    x0.push_back(0.1141);              //p2
    x0.push_back(1.7089);              //p3
    x0.push_back(8.05);                //CboxNR
    x0.push_back(0.67);                //fB
    x0.push_back(1e-7);                //rS
    x0.push_back(0.0);                 //sigma_scr
    x0.push_back(1.0);                 //rB_Ext 
    x0.push_back(1.0);                 //r_exp
    x0.push_back(0.0);                 //Q_Ar
    x0.push_back(0.0);                 //Q_Kr
    x0.push_back(1.0);                 //rBAr
    x0.push_back(1.0);                 //rBcryo
    x0.push_back(0.0);                 //sigma_scr
    
    m.SetInitialPositions(x0);
    
    // Write MCMC on root file
    m.WriteMarkovChain(res_dir+m.GetSafeName() + "_mcmc.root", "RECREATE");//, true, true);
    
    
    // Run MCMC, marginalizing posterior
    m.MarginalizeAll();

    // Run mode finding; by default using Minuit
    m.FindMode(m.GetBestFitParameters());

    // Draw all marginalized distributions into a PDF file
    m.PrintAllMarginalized(res_dir+m.GetSafeName() + "_plots.pdf");

    // Print summary plots
    m.PrintParameterPlot(res_dir+m.GetSafeName() + "_parameters.pdf");
    m.PrintCorrelationPlot(res_dir+m.GetSafeName() + "_correlation.pdf");
    m.PrintCorrelationMatrix(res_dir+m.GetSafeName() + "_correlationMatrix.pdf");
    m.PrintKnowledgeUpdatePlots(res_dir+m.GetSafeName() + "_update.pdf");

    // Print results of the analysis into a text file
    m.PrintSummary();
    
    // Close log file
    BCLog::OutSummary("Exiting");
    BCLog::CloseLog();

    return 0;
}
