#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace std;
using namespace Eigen;

bool CheckSolve(const MatrixXd& A);
VectorXd SolveSystemPALU(const MatrixXd& A, const VectorXd& b);
VectorXd SolveSystemQR(const MatrixXd& A, const VectorXd& b);
void CheckSolutions(const MatrixXd& A, const VectorXd& b, VectorXd& solPALU, VectorXd& solQR);
void CheckError(const VectorXd& solPALU, const VectorXd& solQR, const VectorXd& solution, double& errRelPALU, double& errRelQR);

int main()
{
    Vector2d x(-1.0000e+0, -1.0000e+00);

    Matrix2d A1 {{5.547001962252291e-01, -3.770900990025203e-02}, {8.320502943378437e-01, -9.992887623566787e-01}};
    Vector2d b1(-5.169911863249772e-01, 1.672384680188350e-01);

    Matrix2d A2 {{5.547001962252291e-01, -5.540607316466765e-01}, {8.320502943378437e-01,-8.324762492991313e-01}};
    Vector2d b2(-6.394645785530173e-04, 4.259549612877223e-04);


    Matrix2d A3 {{5.547001962252291e-01, -5.547001955851905e-01}, {8.320502943378437e-01,-8.320502947645361e-01}};
    Vector2d b3(-6.400391328043042e-10, 4.266924591433963e-10);

    cout << scientific << setprecision(16);

    if (CheckSolve(A1))
    {
        VectorXd solPALU1, solQR1;
        double  errPALU1, errQR1;
        CheckSolutions(A1, b1, solPALU1, solQR1);
        CheckError(solPALU1, solQR1, x, errPALU1, errQR1);
        cout << "System 1:" << endl;
        cout << "PALU decomposition solution: " << solPALU1.transpose() << ", QR decomposition solution: " << solQR1.transpose() << endl;
        cout << "PALU decomposition relative error: " << errPALU1 << ", QR decomposition relative error: " << errQR1 << endl;
    }
    else
    {
        cout << "Matrix 1 is singular, system 1 is unsolvable" << endl;
    }

    if (CheckSolve(A2))
    {
        VectorXd solPALU2, solQR2;
        double  errPALU2, errQR2;
        CheckSolutions(A2, b2, solPALU2, solQR2);
        CheckError(solPALU2, solQR2, x, errPALU2, errQR2);
        cout << "System 2:" << endl;
        cout << "PALU decomposition solution: " << solPALU2.transpose() << ", QR decomposition solution: " << solQR2.transpose() << endl;
        cout << "PALU decomposition relative error: " << errPALU2 << ", QR decomposition relative error: " << errQR2 << endl;
    }
    else
    {
        cout << "Matrix 2 is singular, system 2 is unsolvable" << endl;
    }

    if (CheckSolve(A3))
    {
        VectorXd solPALU3, solQR3;
        double  errPALU3, errQR3;
        CheckSolutions(A3, b3, solPALU3, solQR3);
        CheckError(solPALU3, solQR3, x, errPALU3, errQR3);
        cout << "System 3:" << endl;
        cout << "PALU decomposition solution: " << solPALU3.transpose() << ", QR decomposition solution: " << solQR3.transpose() << endl;
        cout << "PALU decomposition relative error: " << errPALU3 << ", QR decomposition relative error: " << errQR3 << endl;
    }
    else
    {
        cout << "Matrix 3 is singular, system 3 is unsolvable" << endl;
    }


    return 0;

}

bool CheckSolve(const MatrixXd& A)
{
    JacobiSVD<MatrixXd> svd(A);
    VectorXd singValA = svd.singularValues();
    if(singValA.minCoeff() < 1e-16)
    {
        return false;
    }
    return true;
}

VectorXd SolveSystemPALU(const MatrixXd& A, const VectorXd& b)
{
    VectorXd solPALU = A.fullPivLu().solve(b);
    return solPALU;
}

VectorXd SolveSystemQR(const MatrixXd& A, const VectorXd& b)
{
    VectorXd solQR = A.fullPivHouseholderQr().solve(b);
    return solQR;
}

void CheckSolutions(const MatrixXd& A, const VectorXd& b, VectorXd& solPALU, VectorXd& solQR)
{
    solPALU = SolveSystemPALU(A, b);
    solQR = SolveSystemQR(A, b);
}

void CheckError(const VectorXd& solPALU, const VectorXd& solQR, const VectorXd& solution, double& errPALU, double& errQR)
{

    errPALU = (solPALU - solution).norm()/solution.norm();
    errQR = (solQR - solution).norm()/solution.norm();

}

