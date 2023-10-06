#include <stdio.h>
int N = 2160;

double PotentialAux(int i, int j)
{
    return 1;
}

// Function to calculate the potential energy of the system
double Potential() {
    double Pot=0.;
    int j, i;
    for (i=0; i<N; i++) {
        for (j=0; j<i; j++)
            Pot+=PotentialAux(i,j);
        for (j+=1; j < N; j++)
            Pot+=PotentialAux(i,j);
    }
    return Pot;
}

int main()
{
    printf("%f\n",Potential());
    return 0;
}