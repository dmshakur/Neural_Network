#include <iostream>
#include "Perceptron.cpp"

int main() 
{
    Perceptron network {"./iris.txt"};

    network.display();

    std::cout << std::endl;

    network.train(1);

    std::cout << std::endl;    

    network.display();

    return 0;
}
