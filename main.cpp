#include <iostream>
#include "Neural_Network.cpp"

int main() 
{
    Neural_Network network {"./iris.txt"};
    // Initializing MLP with dataset file path

    // network.display();

    std::cout << std::endl << std::fixed << std::setprecision(2);

    network.train(20);

    std::cout << std::endl;    

    // network.display();

    return 0;
}
