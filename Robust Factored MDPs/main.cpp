#include "parameter.h"
#include "function.h"

#include <sstream>

extern int     neighbors[number_of_computers][number_of_neighbors];
NumArray3D     weight, weight_beta, weight_gamma;
NumArray2DA    weight_eta;
int type = 1;   // default: single-direct ring
                // type 1:  bi-direct ring
                // type 2:  star
                // type 3:  ring and star
                // type 4:  rings and rings
                // type 5:  3 legs

// radius of uncertainty set, set as 0, 0.03, 0.06, 0.09, 0.12, 0.15
double          epsilon = 0.03;  

// the percentage of test data
double          test_ratio = 0.8;  

// the number of history observations, set as 100, 200, 300, 400, 500, 600
int             number_of_history_data = 500;

int             state_constr[number_of_computers][number_of_constraints];
int             original_constr[number_of_computers][number_of_constraints];
int             action_constr[number_of_computers][number_of_constraints];
int             time_constr[number_of_constraints];
int             constr_num = 0;


int main(int argc, const char* argv[], const char* envi[]) {

    // create transition probability
    create_tran_prob();

    // create the network for different type of topology
    create_network_topology(type);

    // create basis functions for each scope
    create_basis_functions(type);

    unsigned int seed = job_number; //static_cast<unsigned int>(time(nullptr));
    srand(seed);

    // initial state: all one
    int state[number_of_computers];
    for (int k = 0; k < number_of_computers; k++) state[k] = 1;

    // generate historical data via some greedy strategy
    monto_carlo_history();

    // Ture formulation: oracle knows the true transition probability
    cout << "True formulation:" << endl;
    
    // Cutting plane method to solve the true formulation
    cutting_plane_true_formulation(weight); 
   
    // objective value for all-one state
    double value_all_one = 0;
    for (int k = 0; k < number_of_computers; k++) value_all_one += weight[1][1][k];

    cout << "Objective value for all-one state: " << value_all_one << endl;

    double value_mc_true = monto_carlo_MIP_true(weight, state);


    cout << "True formualtion Monte Carlo value: " << value_mc_true << endl << endl;

    // Non-Robust formulation with estimated transition probability
    cout << endl << "Non-robust formulation: " << endl;

    // Cutting plane method to solve the nominal formulation
    cutting_plane_nominal_formulation(weight);

    value_all_one = 0;
    for (int k = 0; k < number_of_computers; k++) value_all_one += weight[1][1][k];

    cout << "Objective value for all-one state: " << value_all_one << endl;

    double value_mc_non_in_sample = monto_carlo_MIP_in_sample(weight, state);

    cout << "In of Sample -- Non Robust case Monte Carlo value: " << value_mc_non_in_sample << endl;

    double value_mc_non = monto_carlo_MIP(weight, state);

    cout << "Out of Sample -- Non Robust case Monte Carlo value: " << value_mc_non << endl << endl;

    // Robust formulation with the infinite-norm ambiguity set
    cout << endl << "Robust formulation: " << endl;

    auto startTime = std::chrono::steady_clock::now();

    double value_mc;

    constr_num = 0;

    // Cutting plane method to solve the robust formulation
    cutting_plane_robust_formulation(weight, weight_beta, weight_gamma, weight_eta);

    auto endTime = std::chrono::steady_clock::now();

    double solvingTime = std::chrono::duration<double>(endTime - startTime).count();

    cout << endl << "Type of topology: " << type << "; number of computers: " << number_of_computers << endl << endl;

    // Use the other 20% historical data for validation to determine the best radii 
    value_mc = monto_carlo_MIP_validation(weight, weight_beta, weight_eta, state);

    cout << endl << endl;

    cout << "Robust case Monte Carlo value for validation: " << value_mc << endl;

    double value_mc_in_sample = monto_carlo_MIP_robust_in_sample(weight, weight_beta, weight_eta, state);

    cout << "In sample -- Robust case Monte Carlo value: " << value_mc_in_sample << endl;

    value_mc = monto_carlo_MIP_robust(weight, weight_beta, weight_eta, state);

    cout << "Out of sample -- Robust case Monte Carlo value: " << value_mc << endl;

    cout << "In Sample -- Non Robust case Monte Carlo value: " << value_mc_non_in_sample << endl;

    cout << "Out of Sample Non Robust case Monte Carlo value: " << value_mc_non << endl;

    cout << "True case Monte Carlo value: " << value_mc_true << endl;

    return 0;

}