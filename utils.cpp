#include "utils.hpp"

MatrixXd one_hot(VectorXd& y, int num_labels){
    int uniq;
    int n_samples = y.size();
    if(num_labels<=0){
        uniq =  y.maxCoeff() + 1;
    }else{
        uniq = num_labels;
    }

    MatrixXd ary = MatrixXd::Zero(n_samples, uniq);
    for(int i=0; i<y.size(); i++){
        ary(i, static_cast<int>(y(i))) = 1;
    }
    return ary;
}


// int main(int argc, char**argv){
    
//     VectorXd y(6);
//     y << 0, 1, 3, 3, 2, 1; 
//     MatrixXd y_enc = one_hot(y);
//     cout << "y encoded >>> \n" << y_enc << endl;
//     return 0;
// }