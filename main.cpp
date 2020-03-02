#include <iostream>
#include "Neural_Network.cpp"

int main() 
{
    Neural_Network network {"./iris.txt"};

    std::cout << std::endl << std::fixed << std::setprecision(2);

    network.train(5);

    std::cout << std::endl;

    return 0;
}
