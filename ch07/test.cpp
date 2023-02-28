#include<iostream>
using namespace std;

int main(int argc, char **argv){
    int d, k;
    sscanf(argv[1], "%d", &d);
    sscanf(argv[2], "%d", &k);
    cout << "计算前****************d： " << d << "**************k: " <<
         k << "******************" << endl;
    d |= 1 << k;
    cout << "计算后****************d： " << d << "**************k: " <<
    k << "******************" <<  endl;
}