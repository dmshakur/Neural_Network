#include <iostream>
#include "Perceptron.cpp"

int main() 
{
    Perceptron test {"./iris.txt"};

    // test.display();

    test.train(5);

    return 0;
}