#include <iostream>
#include "Perceptron.cpp"

int main() 
{
    Perceptron network {"./iris.txt"};
    // Initializing MLP with dataset file path

    network.display();

    std::cout << std::endl;

    network.train(5);

    std::cout << std::endl;    

    network.display();

    return 0;
}
