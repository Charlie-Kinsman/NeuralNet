#include <vector>
#include <iostream>
#include "neural.h"

int main(){
    std::cout<<"here"<<std::endl;
    return 0;
}

Neuron::Update(int x, int y){
    inputs = x;
    value = y;
}