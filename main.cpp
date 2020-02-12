#include <iostream>
#include <memory>
#include "Perceptron.cpp"

int main() 
{
    Perceptron test {"../iris.txt"};

    test.display();

    return 0;
}