#include "parameter.h"
#include "function.h"

using namespace std;

extern int     neighbors[number_of_computers][number_of_neighbors];
extern double  prob[number_of_states][number_of_states][number_of_states][number_of_states];
NumArray2D     weight2;   // decision variables for scope-one formulation
NumArray3D     weight3;   // decision variables for scope-two formulation
NumArray4D     weight4;   // decision variables for scope-three formulation
int type = 1;   // default: single-direct ring
                // type 1:  bi-direct ring
                // type 2:  star
                // type 3:  ring and star
                // type 4:  rings and rings
                // type 5:  3 legs

int main(int argc, const char* argv[], const char* envi[]) {

    // create transition probability
    create_tran_prob();

    // create the network for different type of topology
    create_network_topology(type);

    // create basis functions for each scope
    create_basis_functions(type);

    // set the initial state (all-one state) for Monte carlo simulations
    int init_state[number_of_computers];
    for (int k = 0; k < number_of_basis; k++) init_state[k] = 1;

    // Naive strategy 1: random
    double monte_carlo_value_mean = monto_carlo_MIP_random(init_state);  
    cout << "Monte carlo simulation value of random strategy: " << monte_carlo_value_mean << endl;

    // Naive strategy 2: priority
    monte_carlo_value_mean = monto_carlo_MIP_priority(init_state);
    cout << endl << "Monte carlo simulation value of priority strategy: " << monte_carlo_value_mean << endl;

    // Naive strategy 3: level
    monte_carlo_value_mean = monto_carlo_MIP_level(init_state);
    cout << endl << "Monte carlo simulation value of level strategy: " << monte_carlo_value_mean << endl;

    // Cutting plane scheme: scope one formulation
    cout << endl << "Cutting plane scheme scope-one formulation :" << endl << endl;

    auto startTime_se_1 = std::chrono::steady_clock::now();

    // Cutting plane method to solve the scope-one formulation
    cutting_plane_scope_one(weight2);

    auto endTime_se_1 = std::chrono::steady_clock::now();

    double solvingTime_se_1 = std::chrono::duration<double>(endTime_se_1 - startTime_se_1).count();
    std::cout << "Solving the whole problem time: " << solvingTime_se_1 << " seconds" << std::endl;

    double value_all_one = 0;
    for (int k = 0; k < number_of_basis; k++) value_all_one += weight2[1][k];

    cout << "Objective value for all one state:" << value_all_one << endl;

    monte_carlo_value_mean = monto_carlo_MIP_scope_one(weight2, init_state);

    cout << "Monte Carlo simulation value: " << monte_carlo_value_mean << endl;

    cout << "Optimality gap: " << (value_all_one - monte_carlo_value_mean) / monte_carlo_value_mean << endl;

    // Cutting plane scheme: scope two formulation
    cout << endl << "Cutting plane scheme scope-two formulation :" << endl << endl;

    auto startTime_se_2 = std::chrono::steady_clock::now();

    // Cutting plane method to solve the scope-two formulation
    cutting_plane_scope_two(weight3);

    auto endTime_se_2 = std::chrono::steady_clock::now();

    double solvingTime_se_2 = std::chrono::duration<double>(endTime_se_2 - startTime_se_2).count();
    std::cout << "Solving the whole problem time: " << solvingTime_se_2 << " seconds" << std::endl;
    

    value_all_one = 0;
    for (int k = 0; k < number_of_basis; k++) value_all_one += weight3[1][1][k];

    cout << "Objective value for all one state:" << value_all_one << endl;

    monte_carlo_value_mean = monto_carlo_MIP_scope_two(weight3, init_state);

    cout << "Monte Carlo simulation value: " << monte_carlo_value_mean << endl;

    cout << "Optimality gap: " << (value_all_one - monte_carlo_value_mean) / monte_carlo_value_mean << endl;

    // Cutting plane scheme: scope three formulation
    cout << endl << "Cutting plane scheme scope-three formulation :" << endl << endl;

    auto startTime_se_3 = std::chrono::steady_clock::now();

    // Cutting plane method to solve the scope-three formulation
    cutting_plane_scope_three(weight4);

    auto endTime_se_3 = std::chrono::steady_clock::now();

    double solvingTime_se_3 = std::chrono::duration<double>(endTime_se_3 - startTime_se_3).count();
    std::cout << "Solving the whole problem time: " << solvingTime_se_3 << " seconds" << std::endl;

    value_all_one = 0;
    for (int k = 0; k < number_of_basis; k++) value_all_one += weight4[1][1][1][k];
    
    cout << "Objective value for all one state:" << value_all_one << endl;

    monte_carlo_value_mean = monto_carlo_MIP_scope_three(weight4, init_state);

    cout << "Monte Carlo simulation value: " << monte_carlo_value_mean << endl;

    cout << "Optimality gap: " << (value_all_one - monte_carlo_value_mean) / monte_carlo_value_mean << endl;

    // Variable elimination method (VE): scope one
    cout << endl << "VE: scope one formulation :" << endl;

    auto startTime_1 = std::chrono::steady_clock::now();

    solve_overall_problem_scope_1_VE_two_actions();

    auto endTime_1 = std::chrono::steady_clock::now();

    double solvingTime_1 = std::chrono::duration<double>(endTime_1 - startTime_1).count();
    std::cout << "Solving the whole problem time: " << solvingTime_1 << " seconds" << std::endl;

    value_all_one = 0;
    for (int k = 0; k < number_of_basis; k++) value_all_one += weight2[1][k];

    cout << "upper bound of scope 1: " << value_all_one << endl << endl;

    monte_carlo_value_mean = monto_carlo_MIP_scope_one(weight2, init_state);

    cout << "Monte Carlo simulation value: " << monte_carlo_value_mean << endl;

    cout << "Optimality gap: " << (value_all_one - monte_carlo_value_mean) / monte_carlo_value_mean << endl;

    // Variable elimination method (VE): scope one
    cout << endl << "VE: scope two formulation :" << endl;

    auto startTime = std::chrono::steady_clock::now();

    solve_overall_problem_scope_2_VE_two_actions();

    auto endTime = std::chrono::steady_clock::now();

    double solvingTime = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "Solving the whole problem time: " << solvingTime << " seconds" << std::endl;

    value_all_one = 0;
    for (int k = 0; k < number_of_basis; k++) value_all_one += weight3[1][1][k];
 
    cout << "upper bound of scope 2: " << value_all_one << endl << endl;

    monte_carlo_value_mean = monto_carlo_MIP_scope_two(weight3, init_state);

    cout << "Monte Carlo simulation value: " << monte_carlo_value_mean << endl;

    cout << "Optimality gap: " << (value_all_one - monte_carlo_value_mean) / monte_carlo_value_mean << endl;

    return 0;
}