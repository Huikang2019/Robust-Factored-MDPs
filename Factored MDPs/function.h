#ifndef FUNCTION_H
#define FUNCTION_H

int decr(int i);

int incr(int i);

void integer_to_binary(int inte, int* binary_i);

void create_tran_prob();

void create_basis_functions(int type);

void create_network_topology(int type);

int state_to_index(int* current_state);

void solve_overall_problem_scope_1_VE();

void solve_overall_problem_scope_1_VE_two_actions();

void solve_overall_problem_scope_2_VE();

void solve_overall_problem_scope_2_VE_two_actions();

void solve_overall_problem_scope_1_VE_RoR();

double monto_carlo_MIP_random(const int* initial_state);

double monto_carlo_MIP_priority(const int* initial_state);

double monto_carlo_MIP_level(const int* initial_state);

double monto_carlo_scope_one(NumArray2D weight, const int* initial_state, double value_all_one);

void creat_mastered_problem_scope_one(GRBModel& model, NumVar2D& alpha);
void creat_mastered_problem_scope_two(GRBModel& model, NumVar3D& alpha);
void creat_mastered_problem_scope_three(GRBModel& model, NumVar4D& alpha);

void add_constraint_to_master_problem_scope_one(GRBModel& model, NumVar2D& alpha,
    const int* current_state, const int* current_action);
void add_constraint_to_master_problem_scope_two(GRBModel& model, NumVar3D& alpha,
    const int* current_state, const int* current_action);
void add_constraint_to_master_problem_scope_three(GRBModel& model, NumVar4D& alpha,
    const int* current_state, const int* current_action);

double solve_mastered_problem_scope_one(GRBModel& model, NumVar2D& alpha, NumArray2D& weight);
double solve_mastered_problem_scope_two(GRBModel& model, NumVar3D& alpha, NumArray3D& weight);
double solve_mastered_problem_scope_three(GRBModel& model, NumVar4D& alpha, NumArray4D& weight);

//double state_constraint_value(NumArray3D& weight, const int* current_state, const int* current_action);

//double compare_two_states(NumArray3D& weight, const int* current_state_1,const int* current_state_2, const int* current_action);

//double compare_two_states_naive(NumArray3D& weight, const int* current_state_1,const int* current_state_2, const int* current_action);

void random_generate_constraints_local_search_scope_one(GRBModel& model,
    NumVar2D& alpha, NumArray2D& weight, int number_of_constr);
void random_generate_constraints_local_search_scope_two(GRBModel& model, 
    NumVar3D& alpha, NumArray3D& weight, int number_of_constr);
void random_generate_constraints_local_search_scope_three(GRBModel& model,
    NumVar4D& alpha, NumArray4D& weight, int number_of_constr);

void generate_state_action(int* current_state, int* current_action);

double solve_subed_problem_GRB_scope_one(NumArray2D weight, double eps, double time_limit_least, double time_limit_most,
    int* worst_state, int* worst_action);
double solve_subed_problem_GRB_scope_two(NumArray3D weight, double eps, double time_limit_least, double time_limit_most,
    int* worst_state, int* worst_action);
double solve_subed_problem_GRB_scope_three(NumArray4D weight, double eps, double time_limit_least, double time_limit_most,
    int* worst_state, int* worst_action);

void fixed_constraints_local_search_scope_one(GRBModel& model, NumVar2D& alpha, NumArray2D& weight, int* state, int* action);
void fixed_constraints_local_search_scope_two(GRBModel& model, NumVar3D& alpha, NumArray3D& weight, int* state, int* action);
void fixed_constraints_local_search_scope_three(GRBModel& model, NumVar4D& alpha, NumArray4D& weight, int* state, int* action);

double cutting_plane_scope_one(NumArray2D& weight);
double cutting_plane_scope_two(NumArray3D& weight);
double cutting_plane_scope_three(NumArray4D& weight);

double monto_carlo_MIP_scope_one(NumArray2D weight, const int* initial_state);
double monto_carlo_MIP_scope_two(NumArray3D weight, const int* initial_state);
double monto_carlo_MIP_scope_three(NumArray4D weight, const int* initial_state);


#endif

