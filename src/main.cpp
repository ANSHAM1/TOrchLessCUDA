#include "kernel.cuh"

#include <iostream>


int main()
{
    constexpr int size = 5;


    int a[size] = {
        1,2,3,4,5
    };


    int b[size] = {
        10,20,30,40,50
    };


    int c[size] = { 0 };



    vectorAdd(
        a,
        b,
        c,
        size
    );


    std::cout << "Result: ";

    for (int i = 0; i < size; i++)
    {
        std::cout << c[i] << " ";
    }


    std::cout << "\n";


    return 0;
}