
#ifndef FUNCTION_H
#define FUNCTION_H

int decr(int i);

int incr(int i);

void create_tran_prob();

void monto_carlo_history();

void create_basis_functions(int type);

void create_network_topology(int type);

double cutting_plane_true_formulation(NumArray3D& weight);

double cutting_plane_nominal_formulation(NumArray3D& weight);

double cutting_plane_robust_formulation(NumArray3D& weight, NumArray3D& weight_beta,
    NumArray3D& weight_gamma, NumArray2DA& weight_eta);

double mastered_problem_weight(int constr_num, NumArray3D& weight, NumArray3D& weight_beta, NumArray3D& weight_gamma, NumArray2DA& weight_eta);

void generate_state_action(int* current_state, int* current_action);

void generate_state_action_half(int* current_state, int* original_state, int* current_action, int& time);

void random_generate_constraints_local_search(NumArray3D& weight, NumArray3D& weight_beta, NumArray3D& weight_gamma, NumArray2DA& weight_eta,
    int number_of_constr, int& constr_num);

double monto_carlo_MIP(NumArray3D weight, const int* initial_state);

double monto_carlo_MIP_in_sample(NumArray3D weight, const int* initial_state);

void creat_mastered_problem_NonRobust(GRBModel& model, NumVar3D& alpha);

void random_generate_constraints_local_search_NonRobust(GRBModel& model, NumVar3D& alpha, NumArray3D& weight,
    int number_of_constr, int& constr_num);

void random_generate_constraints_local_search_true(GRBModel& model, NumVar3D& alpha, NumArray3D& weight,
    int number_of_constr);

void add_constraint_to_master_problem(GRBModel& model, NumVar3D& alpha,
    const int* current_state, const int* current_action, int& constr_num);

void add_constraint_to_master_problem_true(GRBModel& model, NumVar3D& alpha,
    const int* current_state, const int* current_action);

double mastered_problem_prob(int constr, NumArray3D& weight, NumArray3D& weight_beta);

double prob_out(int* current_state, int* original_state, int i, int action_i, int state_i, int current_time);
double prob_out_robust(int* current_state, int i, int action_i, int state_i, int current_time, int num_constr);

double trans_prob_update(int constr, NumArray3D& weight, NumArray3D& weight_beta, int* current_state, int* current_action,
    int* original_state, int time);

double monto_carlo_MIP_true(NumArray3D weight, const int* initial_state);

double monto_carlo_MIP_robust(NumArray3D& weight, NumArray3D& weight_beta, NumArray2DA& weight_eta, const int* initial_state);

double monto_carlo_MIP_robust_in_sample(NumArray3D& weight, NumArray3D& weight_beta, NumArray2DA& weight_eta, const int* initial_state);

double monto_carlo_MIP_validation(NumArray3D& weight, NumArray3D& weight_beta, NumArray2DA& weight_eta, const int* initial_state);


#endif


