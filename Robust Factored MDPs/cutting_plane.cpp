#include "parameter.h"
#include "function.h"

extern int             constr_num;

double cutting_plane_true_formulation(NumArray3D& weight) {
    GRBEnv* env_true = new GRBEnv();

    GRBModel model_true = GRBModel(env_true);

    NumVar3D alpha;

    creat_mastered_problem_NonRobust(model_true, alpha);

    double eps = 0.0001, value = 0;
    for (int iter = 0; iter < 100 * number_of_computers; iter++) {
        for (int inner_iter = 0; inner_iter < 5; inner_iter++) {
            random_generate_constraints_local_search_true(model_true, alpha, weight, 2 * number_of_computers);
            model_true.optimize();

            for (int i = 0; i < number_of_states; i++) {
                for (int j = 0; j < number_of_states; j++) {
                    for (int k = 0; k < number_of_computers; k++) {
                        weight[i][j][k] = alpha[i][j][k].get(GRB_DoubleAttr_X);
                    }
                }
            }
        }

        // cout << "number of constraints:" << model_true.get(GRB_IntAttr_NumConstrs) << "  ";
        // cout << "value of master problem: " << model_true.get(GRB_DoubleAttr_ObjVal) << endl;

        if ((model_true.get(GRB_DoubleAttr_ObjVal) < (1 + eps) * value) && (iter > number_of_computers)) {
            break;
        }

        value = model_true.get(GRB_DoubleAttr_ObjVal);
    }

    delete env_true;

    return value;
}

double cutting_plane_nominal_formulation(NumArray3D& weight) {
    GRBEnv* env = new GRBEnv();

    GRBModel model = GRBModel(env);

    NumVar3D alpha;

    creat_mastered_problem_NonRobust(model, alpha);

    double eps = 0.0001, value = 0;
    for (int iter = 0; iter < 100 * number_of_computers; iter++) {
        for (int inner_iter = 0; inner_iter < 5; inner_iter++) {
            random_generate_constraints_local_search_NonRobust(model, alpha, weight, 2 * number_of_computers, constr_num);
            model.optimize();

            for (int i = 0; i < number_of_states; i++) {
                for (int j = 0; j < number_of_states; j++) {
                    for (int k = 0; k < number_of_computers; k++) {
                        weight[i][j][k] = alpha[i][j][k].get(GRB_DoubleAttr_X);
                    }
                }
            }
        }

        // cout << "number of constraints:" << model.get(GRB_IntAttr_NumConstrs) << "  ";
        // cout << "value of master problem: " << model.get(GRB_DoubleAttr_ObjVal) << endl;

        if ((model.get(GRB_DoubleAttr_ObjVal) < (1 + eps) * value) && (iter > number_of_computers)) {
            break;
        }

        value = model.get(GRB_DoubleAttr_ObjVal);
    }

    delete env;

    return value;
}

double cutting_plane_robust_formulation(NumArray3D& weight, NumArray3D& weight_beta, 
    NumArray3D& weight_gamma, NumArray2DA& weight_eta) {

    double value_old = 0, value_new;
    for (int iter = 0; iter < 100; iter++) {
        cout << "round: " << iter << endl;
        random_generate_constraints_local_search(weight, weight_beta, weight_gamma, weight_eta, 10 * number_of_computers, constr_num);
        if (constr_num >= number_of_constraints) break;
        //mastered_problem_prob(constr_num, weight, weight_beta);
        value_new = mastered_problem_weight(constr_num, weight, weight_beta, weight_gamma, weight_eta);
        if ((value_new > 0) && (value_new < value_old * 0)) break;
        else value_old = value_new;
        cout << "master problem value : " << value_old << endl;
        for (int i = 0; i < 5; i++) {
            mastered_problem_prob(constr_num, weight, weight_beta);
            value_new = mastered_problem_weight(constr_num, weight, weight_beta, weight_gamma, weight_eta);
            cout << "master problem value : " << value_new << endl;
            if (value_new >= 0.98 * value_old) {
                value_old = value_new;
                break;
            }
            else value_old = value_new;
            if (value_new < 0) break;
        }

    }

    return value_new;
}

