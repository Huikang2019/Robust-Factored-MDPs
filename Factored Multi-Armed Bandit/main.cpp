#include <ilcplex/ilocplex.h>
#include <gurobi_c++.h>
#include <list>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <time.h>
#include <chrono>
#include <thread>
#include <atomic>

#include <sstream>

using namespace std;

#define ILOUSESTL

ILOSTLBEGIN

int number_of_arms;
int number_of_states;
int number_of_states_shared_signal = 5;
double pulled_ratio;
int number_of_arms_pulled_once;
double discount_factor = 0.95;
double percentage_specail_arms;

int number_of_repeat = 200;
const int number_of_time_periods = 150;

double* init_prob_shared_signal;
double** init_prob_arms;
double** trans_prob_shared_signal;
double**** trans_prob_arms;
double*** rewards;

void generate_dist(int dimension, double dist[], double lower_bound) {
    double prob_sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        dist[i] = (double)rand() / (RAND_MAX);
        prob_sum += dist[i];
    }
    for (int i = 0; i < dimension; i++) dist[i] /= prob_sum;
}

void initialization() {  

    // the distribution of initial states of shared signal
    init_prob_shared_signal = new double[number_of_states_shared_signal];
    generate_dist(number_of_states_shared_signal, init_prob_shared_signal, 0);

    // the distribution of initial states 
    init_prob_arms = new double* [number_of_arms];
    for (int k = 0; k < number_of_arms; k++) {
        init_prob_arms[k] = new double[number_of_states];
        generate_dist(number_of_states, init_prob_arms[k], 0 );
    }

    // the shared signal z = i, it transfers to h with prob = trans_prob_shared_signal[i][j]
    trans_prob_shared_signal = new double* [number_of_states_shared_signal];
    for (int i = 0; i < number_of_states_shared_signal; i++) {
        trans_prob_shared_signal[i] = new double[number_of_states_shared_signal];
        generate_dist(number_of_states_shared_signal, trans_prob_shared_signal[i], 0);
    }

    // for k-th arm at state j and the shared signal z = i, it transfers to state h with prob = trans_prob[k][i][j][h]
    trans_prob_arms = new double*** [number_of_arms];
    for (int k = 0; k < number_of_arms; k++) {
        trans_prob_arms[k] = new double** [number_of_states_shared_signal];
        for (int i = 0; i < number_of_states_shared_signal; i++) {
            trans_prob_arms[k][i] = new double* [number_of_states];
            for (int j = 0; j < number_of_states; j++) {
                trans_prob_arms[k][i][j] = new double[number_of_states];
                generate_dist(number_of_states, trans_prob_arms[k][i][j], 0);
            }
        }
    }
    // for k-th arm at state j and the shared signal z = i, the reward is rewards[k][i][j]
    rewards = new double** [number_of_arms];
    for (int k = 0; k < number_of_arms; k++) {
        rewards[k] = new double* [number_of_states_shared_signal];
        for (int i = 0; i < number_of_states_shared_signal; i++) {
            rewards[k][i] = new double[number_of_states];
            for (int j = 0; j < number_of_states; j++) {
                rewards[k][i][j] = (double)rand() / (RAND_MAX);
            }
        }
    }
    for (int k = 0; k < number_of_arms; k++) {
        if (((double)rand() / (RAND_MAX)) < percentage_specail_arms) {
            for (int i = 0; i < number_of_states_shared_signal; i++) {
                rewards[k][i][0] = (1.0 * (number_of_states - 1) + 1) * ((double)rand() / (RAND_MAX));
                for (int j = 1; j < number_of_states; j++) {
                    rewards[k][i][j] = 0 * (double)rand() / (RAND_MAX);
                }
            }
        }
        
    }

}

void creat_mastered_problem(GRBModel& model, vector < vector< vector<GRBVar>>>& alpha) {
    GRBEnv env = model.getEnv();
    GRBLinExpr obj_func;

    for (int k = 0; k < number_of_arms; k++) {
        for (int i = 0; i < number_of_states_shared_signal; i++) {
            for (int j = 0; j < number_of_states; j++) {
                alpha[k][i][j] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);               
                obj_func += alpha[k][i][j] * init_prob_arms[k][j] * init_prob_shared_signal[i];
            }
        }
    }
   
    model.setObjective(obj_func, GRB_MINIMIZE);
    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_Threads, 4);
    model.update();
}

void generate_state_action(vector<int>& current_state, vector<int>& current_action) {
    double random_number;
    for (int j = 0; j < number_of_arms; ++j) {
        random_number = 0.999999 * ((double)rand() / (RAND_MAX));
        current_state[j] = int(random_number * number_of_states);
        current_action[j] = 0;
    }
    for (int j = 0; j < number_of_arms_pulled_once; ++j) {
        random_number = 0.999999 * ((double)rand() / (RAND_MAX));
        int action_i = (int)(number_of_arms * random_number);
        current_action[action_i] = 1;
    }
}

void add_constraint_to_master_problem(GRBModel& model, vector < vector< vector<GRBVar>>>& alpha, const vector<int>& current_state,
    const vector<int>& current_action, int state_shared_signal) {
    GRBEnv env = model.getEnv();

    GRBLinExpr curr_LHS;
    for (int k = 0; k < number_of_arms; ++k) {
        if (current_action[k] == 1) curr_LHS -= rewards[k][state_shared_signal][current_state[k]];
    }
    for (int k = 0; k < number_of_arms; k++) {
        curr_LHS += alpha[k][state_shared_signal][current_state[k]];
    }

    GRBLinExpr curr_RHS;
    double prob_k, prob_z;
    for (int z = 0; z < number_of_states_shared_signal; z++) {
        prob_z = trans_prob_shared_signal[state_shared_signal][z];
        for (int k = 0; k < number_of_arms; k++) {
            if (current_action[k] == 1) {
                for (int state_k = 0; state_k < number_of_states; ++state_k) {
                    prob_k = trans_prob_arms[k][state_shared_signal][current_state[k]][state_k];
                    curr_RHS += prob_z * prob_k * alpha[k][z][state_k];
                }
            }
            else curr_RHS += prob_z * alpha[k][z][current_state[k]];
        }
    }       
        
    model.addConstr(discount_factor * curr_RHS - curr_LHS <= 0);
    curr_LHS.clear(); curr_RHS.clear();

    //env.end();
}

double compare_two_states(vector < vector< vector<double>>>& weight, int k, int current_state_k, int next_state_k, 
    int current_action_k, int state_shared_signal) {

    double current_LHS = 0, next_LHS = 0;
    current_LHS += weight[k][state_shared_signal][current_state_k];
    next_LHS += weight[k][state_shared_signal][next_state_k];
    if (current_action_k == 1) {
        current_LHS -= rewards[k][state_shared_signal][current_state_k];
        next_LHS -= rewards[k][state_shared_signal][next_state_k];
    }

    double current_RHS = 0, next_RHS = 0;
    double prob_k, prob_z;
    for (int z = 0; z < number_of_states_shared_signal; z++) {
        prob_z = trans_prob_shared_signal[state_shared_signal][z];
        if (current_action_k == 1) {
            for (int state_k = 0; state_k < number_of_states; ++state_k) {
                prob_k = trans_prob_arms[k][state_shared_signal][current_state_k][state_k];
                current_RHS += prob_z * prob_k * weight[k][z][state_k];
                prob_k = trans_prob_arms[k][state_shared_signal][next_state_k][state_k];
                next_RHS += prob_z * prob_k * weight[k][z][state_k];
            }
        }
        else {
            current_RHS += prob_z * weight[k][z][current_state_k];
            next_RHS += prob_z * weight[k][z][next_state_k];
        }
    }

    return discount_factor * (current_RHS - next_RHS) - (current_LHS - next_LHS);
}

double state_action_value(vector < vector< vector<double>>>& weight, vector<int>& current_state, vector<int>& current_action, 
    int state_shared_signal) {

    double curr_LHS = 0;
    for (int k = 0; k < number_of_arms; ++k) {
        if (current_action[k] == 1) curr_LHS -= rewards[k][state_shared_signal][current_state[k]];
    }
    for (int k = 0; k < number_of_arms; k++) {
        curr_LHS += weight[k][state_shared_signal][current_state[k]];
    }
    double curr_RHS = 0;
    double prob_k, prob_z;
    for (int z = 0; z < number_of_states_shared_signal; z++) {
        prob_z = trans_prob_shared_signal[state_shared_signal][z];
        for (int k = 0; k < number_of_arms; k++) {
            if (current_action[k] == 1) {
                for (int state_k = 0; state_k < number_of_states; ++state_k) {
                    prob_k = trans_prob_arms[k][state_shared_signal][current_state[k]][state_k];
                    curr_RHS += prob_z * prob_k * weight[k][z][state_k];
                }
            }
            else curr_RHS += prob_z * weight[k][z][current_state[k]];
        }
    }

    return discount_factor * curr_RHS - curr_LHS;
}



void random_generate_constraints_local_search(GRBModel& model, vector < vector< vector<double>>>& weight,
    vector < vector< vector<GRBVar>>>& alpha, int number_of_constr) {
    vector<int> current_state(number_of_arms, 0), next_state(number_of_arms, 0), current_action(number_of_arms,0);
    for (int number = 0; number < number_of_constr; number++) {
        for (int z = 0; z < number_of_states_shared_signal; z++) {
            generate_state_action(current_state, current_action);        
            for (int i = 0; i < number_of_arms; i++) {
                next_state[i] = current_state[i];
            }               
           for (int i = 0; i < number_of_arms; i++) {
                for (int j = 0; j < number_of_states; j++) {
                    if (j != current_state[i]) {
                        next_state[i] = j;
                        if (compare_two_states(weight, i, current_state[i], next_state[i], current_action[i], z) < 0) {
                        //if (state_action_value(weight, current_state, current_action, z) < state_action_value(weight, next_state, current_action, z)) {
                            current_state[i] = j;                           
                        }
                        
                    }
                }
                next_state[i] = current_state[i];
            }
            add_constraint_to_master_problem(model, alpha, current_state, current_action, z);
        }       
    }

}

double action_expected_value(vector < vector< vector<double>>>& weight, const vector<int>& current_state, 
    vector<int>& current_action, int state_shared_signal) {

    double curr_LHS = 0;
    for (int k = 0; k < number_of_arms; ++k) {
        if (current_action[k] == 1) curr_LHS -= rewards[k][state_shared_signal][current_state[k]];
    }

    double curr_RHS = 0;
    double prob_k, prob_z;
    for (int z = 0; z < number_of_states_shared_signal; z++) {
        prob_z = trans_prob_shared_signal[state_shared_signal][z];
        for (int k = 0; k < number_of_arms; k++) {
            if (current_action[k] == 1) {
                for (int state_k = 0; state_k < number_of_states; ++state_k) {
                    prob_k = trans_prob_arms[k][state_shared_signal][current_state[k]][state_k];
                    curr_RHS += prob_z * prob_k * weight[k][z][state_k];
                }
            }
            else curr_RHS += prob_z * weight[k][z][current_state[k]];
        }
    }

    return discount_factor * curr_RHS - curr_LHS;
}

void find_best_action(vector < vector< vector<double>>>& weight, const vector<int>& current_state, vector<int>& best_action, int state_shared_signal) {

    for (int j = 0; j < number_of_arms; ++j) {
        best_action[j] = 0;
    }
    vector<double> value_action;
    double prob_k, prob_z, value_k;
    for (int k = 0; k < number_of_arms; k++) {
        value_k = 0;
        for (int z = 0; z < number_of_states_shared_signal; z++) {
            prob_z = trans_prob_shared_signal[state_shared_signal][z];                   
            for (int state_k = 0; state_k < number_of_states; ++state_k) {
                prob_k = trans_prob_arms[k][state_shared_signal][current_state[k]][state_k];
                value_k += prob_z * prob_k * weight[k][z][state_k];
            }
            value_k -= prob_z * weight[k][z][current_state[k]];
        }
        value_action.push_back(discount_factor * value_k + rewards[k][state_shared_signal][current_state[k]] 
            + 0.00001 * ((double)rand() / (RAND_MAX)));
    }
    vector<int> indices(value_action.size());
    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&](int i1, int i2) { return value_action[i1] < value_action[i2]; });
    for (int j = 0; j < number_of_arms_pulled_once; j++) {
        best_action[indices[number_of_arms - 1 - j]] = 1;
    }
    /*vector <double> value_action_copy = value_action;
    std::sort(value_action_copy.begin(), value_action_copy.end());
    for (int j = 0; j < number_of_arms; ++j) {
        if (value_action[j] > value_action_copy[number_of_arms - number_of_arms_pulled_once - 1])  best_action[j] = 1;
    }*/
}

void find_best_action_random(const vector<int>& current_state, vector<int>& best_action, int state_shared_signal) {

    for (int j = 0; j < number_of_arms; ++j) {
        best_action[j] = 0;
    }
    vector<double> value_action;
    double prob_k, prob_z, value_k;
    for (int k = 0; k < number_of_arms; k++) {
        value_action.push_back(rand());
    }
    vector<int> indices(value_action.size());
    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&](int i1, int i2) { return value_action[i1] < value_action[i2]; });
    for (int j = 0; j < number_of_arms_pulled_once; j++) {
        best_action[indices[number_of_arms - 1 - j]] = 1;
    }
}

void find_best_action_greedy(const vector<int>& current_state, vector<int>& best_action, int state_shared_signal) {

    for (int j = 0; j < number_of_arms; ++j) {
        best_action[j] = 0;
    }
    vector<double> value_action;
    double prob_k, prob_z, value_k;
    for (int k = 0; k < number_of_arms; k++) {
        value_action.push_back(rewards[k][state_shared_signal][current_state[k]]);
    }
    vector<int> indices(value_action.size());
    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&](int i1, int i2) { return value_action[i1] < value_action[i2]; });
    for (int j = 0; j < number_of_arms_pulled_once; j++) {
        best_action[indices[number_of_arms - 1 - j]] = 1;
    }
}

void find_best_action_fluidLP(const vector<int>& current_state, vector<int>& best_action, int state_shared_signal,
    const vector< vector< vector< vector<double>>>>& pi,
    const vector< vector< vector< vector<double>>>>& sigma, int time) {

    for (int j = 0; j < number_of_arms; ++j) {
        best_action[j] = 0;
    }
    vector<double> value_action;
    double ratio = 0, random_number;
    for (int j = 0; j < number_of_arms; ++j) {
        ratio = pi[time][j][state_shared_signal][current_state[j]] / sigma[time][j][state_shared_signal][current_state[j]];
        if (ratio <= 1) value_action.push_back(ratio);
        else value_action.push_back(0);
    }
    for (int j = 0; j < number_of_arms; ++j) {
        random_number = (double)rand() / (RAND_MAX);
        if (random_number < value_action[j]) value_action[j] += 1.0 + 0.01 * ((double)rand() / (RAND_MAX));
    }
    vector<int> indices(value_action.size());
    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&](int i1, int i2) { return value_action[i1] < value_action[i2]; });

    for (int j = 0; j < number_of_arms_pulled_once; j++) {
        best_action[indices[number_of_arms - 1 - j]] = 1;
    }
    std::sort(value_action.begin(), value_action.end());
    for (int j = 0; j < number_of_arms_pulled_once + 2; j++) {
        //cout << value_action[number_of_arms - 1 - j] << " , ";
    }
    //cout << endl;
}


double monto_carlo_simulation(vector < vector< vector<double>>>& weight, vector< vector< vector< vector<double>>>>& pi, vector< vector< vector< vector<double>>>>& sigma, int strategy) {
    double value_sum = 0;
    double random_number, value_opt = 0, value_k = 0, value;
    vector<int> current_state(number_of_arms, 0), action_opt(number_of_arms, 0);
    int state_shared_signal;
    for (int repeat = 0; repeat < number_of_repeat; ++repeat) {
        for (int i = 0; i < number_of_arms; i++) {
            random_number = 0.9999 * ((double)rand() / (RAND_MAX));
            for (int j = 0; j < number_of_states; j++) {
                if (random_number <= init_prob_arms[i][j]) {
                    current_state[i] = j;
                    break;
                }
                else random_number -= init_prob_arms[i][j];
            }
        }
        random_number = 0.99999 * ((double)rand() / (RAND_MAX));
        for (int j = 0; j < number_of_states_shared_signal; j++) {
            if (random_number <= init_prob_shared_signal[j]) {
                state_shared_signal = j;
                break;
            }
            else random_number -= init_prob_shared_signal[j];
        }
        value = 0;
        for (int iter = 0; iter < number_of_time_periods; ++iter) {
            switch (strategy) {
            default:
                find_best_action(weight, current_state, action_opt, state_shared_signal);
                break;
            case 1:
                find_best_action_random(current_state, action_opt, state_shared_signal);
                break;
            case 2:
                find_best_action_greedy(current_state, action_opt, state_shared_signal);
                break;
            case 3:
                find_best_action_fluidLP(current_state, action_opt, state_shared_signal, pi, sigma, iter);
                break;
            }
            
            for (int i = 0; i < number_of_arms; ++i) {
                if (action_opt[i] == 1) {
                    value += rewards[i][state_shared_signal][current_state[i]] * pow(discount_factor, iter);
                    random_number = ((double)rand() / (RAND_MAX));
                    for (int j = 0; j < number_of_states; j++) {
                        if (random_number <= trans_prob_arms[i][state_shared_signal][current_state[i]][j]) {
                            current_state[i] = j; 
                            break;
                        }
                        else random_number -= trans_prob_arms[i][state_shared_signal][current_state[i]][j];
                    }

                }
            }
            random_number = ((double)rand() / (RAND_MAX));
            for (int j = 0; j < number_of_states_shared_signal; j++) {
                if (random_number <= trans_prob_shared_signal[state_shared_signal][j]) {
                    state_shared_signal = j;
                    break;
                }
                else random_number -= trans_prob_shared_signal[state_shared_signal][j];
            }
        }

        value_sum += value;
    }
    return value_sum / number_of_repeat;
}

double fluid_LP(vector< vector< vector< vector<double>>>> & pi_out, vector< vector< vector< vector<double>>>>& sigma_out) {

    IloEnv env;
    IloModel model(env);

    typedef IloArray<IloNumVarArray> NumVarMatrix;
    typedef IloArray<NumVarMatrix>   NumVarThreeMatrix;
    typedef IloArray<NumVarThreeMatrix>   NumVarFourMatrix;

    NumVarMatrix sigma_z(env, number_of_time_periods);
    NumVarFourMatrix pi(env, number_of_time_periods);
    NumVarFourMatrix sigma(env, number_of_time_periods);
    for (int t = 0; t < number_of_time_periods; ++t) {
        sigma_z[t] = IloNumVarArray(env, number_of_states_shared_signal, 0, +IloInfinity);
        pi[t] = NumVarThreeMatrix(env, number_of_arms);
        sigma[t] = NumVarThreeMatrix(env, number_of_arms);
        for (int j = 0; j < number_of_arms; ++j) {
            pi[t][j] = NumVarMatrix(env, number_of_states_shared_signal);
            sigma[t][j] = NumVarMatrix(env, number_of_states_shared_signal);
            for (int k = 0; k < number_of_states_shared_signal; k++) {
                pi[t][j][k] = IloNumVarArray(env, number_of_states, 0, +IloInfinity);
                sigma[t][j][k] = IloNumVarArray(env, number_of_states, 0, +IloInfinity);
            }
        }
    }

    IloExpr obj_func(env);

    for (int t = 0; t < number_of_time_periods; ++t) {
        for (int j = 0; j < number_of_arms; ++j) {
            for (int k = 0; k < number_of_states_shared_signal; k++) {
                for (int h = 0; h < number_of_states; h++) {
                    obj_func += pi[t][j][k][h] * rewards[j][k][h] * pow(discount_factor, t);
                }
            }
        }
    }

    IloObjective objc = IloMaximize(env, obj_func);
    model.add(objc);
    obj_func.end();

    for (int j = 0; j < number_of_arms; j++) {
        for (int k = 0; k < number_of_states_shared_signal; k++) {
            for (int h = 0; h < number_of_states; h++) {
                model.add(sigma[0][j][k][h] == init_prob_arms[j][h] * init_prob_shared_signal[k]);
            }         
        }
    }
    
    for (int k = 0; k < number_of_states_shared_signal; k++)
        model.add(sigma_z[0][k] == init_prob_shared_signal[k]);

    for (int t = 0; t < number_of_time_periods; t++) {
        for (int z = 0; z < number_of_states_shared_signal; z++) {
            IloExpr curr_LHS(env);
            for (int j = 0; j < number_of_arms; j++) {
                for (int k = 0; k < number_of_states; k++) {
                    model.add(pi[t][j][z][k] <= sigma[t][j][z][k]);
                    curr_LHS += pi[t][j][z][k];
                }
            }
            model.add(curr_LHS <= number_of_arms_pulled_once * sigma_z[t][z]);
            curr_LHS.end();
        }
    }

    for (int t = 1; t < number_of_time_periods; t++) {
        for (int k = 0; k < number_of_states_shared_signal; k++) {
            IloExpr curr_LHS(env);
            for (int h = 0; h < number_of_states_shared_signal; h++) {
                curr_LHS += sigma_z[t - 1][h] * trans_prob_shared_signal[h][k];
            }
            model.add(sigma_z[t][k] == curr_LHS);
            curr_LHS.end();
        }        
    }

    for (int t = 1; t < number_of_time_periods; t++) {
        for (int j = 0; j < number_of_arms; j++) {
            for (int k = 0; k < number_of_states_shared_signal; k++) {
                for (int h = 0; h < number_of_states; h++) {
                    IloExpr curr_LHS(env);
                    for (int k1 = 0; k1 < number_of_states_shared_signal; k1++) {
                        for (int h1 = 0; h1 < number_of_states; h1++) {
                            curr_LHS += trans_prob_shared_signal[k1][k] * trans_prob_arms[j][k1][h1][h] * pi[t - 1][j][k1][h1];
                        }
                        curr_LHS += trans_prob_shared_signal[k1][k] * (sigma[t - 1][j][k1][h] - pi[t - 1][j][k1][h]);
                    }
                    model.add(sigma[t][j][k][h] == curr_LHS);
                    curr_LHS.end();
                }             
            }
        }        
    }

    for (int t = 1; t < number_of_time_periods; t++) {
        for (int k = 0; k < number_of_states_shared_signal; k++) {
            IloExpr curr_LHS(env);
            for (int j = 0; j < number_of_arms; j++) {
                for (int h = 0; h < number_of_states; h++) {
                    curr_LHS += sigma[t][j][k][h];
                }
            }
            // model.add(sigma_z[t][k] * number_of_arms == curr_LHS);
        }
    }
        
    

    for (int t = 1; t < number_of_time_periods; t++) {
        for (int j = 0; j < number_of_arms; j++) {
            for (int k = 0; k < number_of_states_shared_signal; k++) {
                for (int h = 0; h < number_of_states; h++) {                    
                    //model.add(pi[t][j][k][h] == pi[t - 1][j][k][h]);
                }
            }
        }
    }
        
    IloCplex cplex(model);
    cplex.setOut(env.getNullStream());
    cplex.setParam(IloCplex::Param::ClockType, 2);
    cplex.setParam(IloCplex::RootAlg, IloCplex::Barrier);
    cplex.setParam(IloCplex::Param::Simplex::Tolerances::Feasibility, 1e-3); 
    cplex.setParam(IloCplex::Param::Simplex::Tolerances::Optimality, 1e-3); 
    cplex.setParam(IloCplex::Param::MIP::Tolerances::AbsMIPGap, 1e-2); 
    cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, 1e-1); 
    cplex.setParam(IloCplex::Param::Emphasis::Numerical, true); 

    cplex.setParam(IloCplex::Threads, 1);
    cplex.solve();

    double objval = cplex.getObjValue();

    for (int t = 0; t < number_of_time_periods; t++) {
        for (int j = 0; j < number_of_arms; j++) {
            for (int h = 0; h < number_of_states_shared_signal; h++) {
                IloNumArray values_ALP(env), values_BLP(env);
                cplex.getValues(values_ALP, pi[t][j][h]);
                cplex.getValues(values_BLP, sigma[t][j][h]);
                for (int k = 0; k < number_of_states; k++) {
                    pi_out[t][j][h][k] = values_ALP[k];
                    sigma_out[t][j][h][k] = values_BLP[k];
                }
            }          
        }
    }

    env.end();

    cout << "vlaue of fluid LP: " << objval << endl;

    return objval;
}

void semi_infinite_algorithm(vector < vector< vector<double>>>& weight) {
    GRBEnv* env = new GRBEnv();
    GRBModel model = GRBModel(env);

    vector < vector< vector<GRBVar>>> alpha(number_of_arms, vector < vector<GRBVar>>(number_of_states_shared_signal, 
        vector<GRBVar>(number_of_states)));

    creat_mastered_problem(model, alpha);

    double eps = 0.00001, value = 0, slackValue;
    int number_of_constraints = 0, ratio_constr_variables = 5;;
    for (int iter = 0; iter < number_of_arms; iter++) {
        for (int inner_iter = 0; inner_iter < 5; inner_iter++) {

            random_generate_constraints_local_search( model, weight, alpha, number_of_states * 20);

            model.optimize();

            for (int k = 0; k < number_of_arms; k++) {
                for (int i = 0; i < number_of_states_shared_signal; i++) {
                    for (int j = 0; j < number_of_states; j++) {
                        weight[k][i][j] = alpha[k][i][j].get(GRB_DoubleAttr_X);
                    }
                }
            }
        }
        number_of_constraints = model.get(GRB_IntAttr_NumConstrs);
        cout << "number of constraints:" << number_of_constraints << "  ";
        cout << "value of master problem: " << model.get(GRB_DoubleAttr_ObjVal) << endl;

        if ((model.get(GRB_DoubleAttr_ObjVal) < (1 + eps) * value) && (iter > number_of_arms)) {
            break;
        }

        value = model.get(GRB_DoubleAttr_ObjVal);

        if (model.get(GRB_IntAttr_NumConstrs) > ratio_constr_variables * model.get(GRB_IntAttr_NumVars)) {
            GRBConstr* NBen = model.getConstrs();
            for (int i = 0; i < model.get(GRB_IntAttr_NumConstrs); i++) {
                slackValue = NBen[i].get(GRB_DoubleAttr_Slack);
                // Use the slackValue as needed (e.g., check if it's large enough and delete the constraint)
                if ((slackValue >= 0.003 * value) && (slackValue >= 0.1)) {
                    model.remove(NBen[i]);
                }
            }
            if (number_of_constraints > (ratio_constr_variables - 1) * model.get(GRB_IntAttr_NumVars))
                ratio_constr_variables = ratio_constr_variables + 2;
            delete[] NBen;
        }
    }

    
}


int main(int argc, const char* argv[], const char* envi[]) {

    // get job number
    int job_number = 1;
    int repetition_number = -1;

    for (int i = 0; envi[i] != NULL; ++i) {
        string str = envi[i];
        if (str.find("PBS_ARRAY_INDEX") != string::npos) {
            job_number = atoi(str.substr(str.find_last_of("=") + 1).c_str());
            break;
        }
    }
    cout << "job number: " << job_number << endl;
    if (job_number == -1) exit(-1);


    int counter = 0;
    for (int index_number_of_arms = 0; index_number_of_arms < 3; ++index_number_of_arms)
        for (int index_number_of_states = 0; index_number_of_states < 4; ++index_number_of_states)
            for (int index_percentage_specail_arms = 0; index_percentage_specail_arms < 3; ++index_percentage_specail_arms)
                for (int index_percentage_pulled = 0; index_percentage_pulled < 4; ++index_percentage_pulled)
                    for (int index_repetition_number = 0; index_repetition_number < 100; ++index_repetition_number) {
                        if (job_number == counter) {

                            switch (index_percentage_specail_arms) {
                            case 0: percentage_specail_arms = 0.0; break;
                            case 1: percentage_specail_arms = 0.1; break;
                            case 2: percentage_specail_arms = 0.2; break;
                            default: exit(-1); break;
                            }

                            switch (index_number_of_states) {
                            case 0: number_of_states = 5; break;
                            case 1: number_of_states = 3; break;
                            case 2: number_of_states = 4; break;
                            case 3: number_of_states = 5; break;
                            default: exit(-1); break;
                            }

                            switch (index_number_of_arms) {
                            case 0: number_of_arms = 20; break;
                            case 1: number_of_arms = 50;  break;
                            case 2: number_of_arms = 100; break;
                            default: exit(-1); break;
                            }

                            switch (index_percentage_pulled) {
                            case 0: pulled_ratio = 0.05; break;
                            case 1: pulled_ratio = 0.1; break;
                            case 2: pulled_ratio = 0.15; break;
                            case 3: pulled_ratio = 0.20; break;
                            default: exit(-1); break;
                            }

                            number_of_arms_pulled_once = int(number_of_arms * pulled_ratio);

                            repetition_number = index_repetition_number;

                            goto done;
                        }
                        ++counter;
                    }
done:

    srand(job_number);
    initialization();

    cout << "number_of_arms: " << number_of_arms << endl;
    cout << "number_of_states: " << number_of_states << endl;
    cout << "percentage_specail_arms: " << percentage_specail_arms << endl;
    cout << "pulled_ratio: " << pulled_ratio << endl;

    // create output file
    stringstream filename;
    filename << "/rds/general/user/wwiesema/home/HKL/" << number_of_arms << "_" << number_of_states << "_" 
        << (int)(10.0 * percentage_specail_arms) << "_" << (int)(100.0 * pulled_ratio) << "_" << repetition_number << ".txt";
    std::ifstream fileCheck(filename.str().c_str());
    if (!fileCheck.good())
    {
        std::ofstream file(filename.str().c_str());
    }
    std::ofstream file(filename.str().c_str(), std::ios::app);
    if (file.is_open()) file << setw(15) << left << number_of_arms;

    vector< vector < vector< vector<double>>>> pi(number_of_time_periods, vector< vector< vector<double>>>
        (number_of_arms, vector < vector<double>>(number_of_states_shared_signal, vector<double>(number_of_states, 0))));
    vector< vector < vector< vector<double>>>> sigma(number_of_time_periods, vector< vector< vector<double>>>
        (number_of_arms, vector < vector<double>>(number_of_states_shared_signal, vector<double>(number_of_states, 0))));
    vector < vector< vector<double>>> weight(number_of_arms, vector < vector<double>>(number_of_states_shared_signal,
        vector<double>(number_of_states, 0)));

    double upperbound = number_of_arms_pulled_once * 20.0;

    //upperbound = fluid_LP(pi, sigma);

    double mc_random = 100 - 100 * monto_carlo_simulation(weight, pi, sigma, 1) / upperbound;
    cout << "monte carlo value of random strategy:" << mc_random << endl;
    file << setw(15) << left << fixed << std::setprecision(1) << mc_random;

    double mc_greedy = 1 - monto_carlo_simulation(weight, pi, sigma, 2) / upperbound;
    cout << "monte carlo value of greedy strategy:" << mc_greedy << endl;
    file << setw(15) << left << fixed << std::setprecision(1) << mc_greedy;

    double mc_FluidLP = 1 - monto_carlo_simulation(weight, pi, sigma, 3) / upperbound;
    cout << "monte carlo value of fluid LP:" << mc_FluidLP << endl;
    file << setw(15) << left << fixed << std::setprecision(1) << mc_FluidLP;

    semi_infinite_algorithm(weight);

    double mc_ours = 1 - monto_carlo_simulation(weight, pi, sigma, 0) / upperbound;
    cout << "monte carlo value of ours:" << mc_ours << endl;
    file << setw(15) << left << fixed << std::setprecision(1) << mc_ours;

    file.close();

    return 0;
}