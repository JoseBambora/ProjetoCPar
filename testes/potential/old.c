#include <stdio.h>
#include <math.h>

int N = 2160;
double sigma = 1.;
double r[5001*3] __attribute__((aligned (32)));
double epsilon = 1.;

double PotentialAux(double sub1, double sub2, double sub3)
{
    double quot = sigma / sqrt(2);
    double quot6 = quot * quot * quot * quot * quot * quot;
    double quot12 = quot6*quot6;
    return (quot12 - quot6);
}

// Function to calculate the potential energy of the system
double Potential() {
    double Pot=0.;
    int j, i;
    for (i=0; i<3*N; i+=3) {
        for (j=0; j<i; j+=3)
            Pot+=PotentialAux(r[i]-r[j],r[i+1]-r[j+1],r[i+2]-r[j+2]);
        for (j+=3; j < 3*N; j+=3)
            Pot+=PotentialAux(r[i]-r[j],r[i+1]-r[j+1],r[i+2]-r[j+2]);
    }
    return 4 * epsilon * Pot;
}

int main()
{
    printf("%f\n",Potential());
    return 0;
}