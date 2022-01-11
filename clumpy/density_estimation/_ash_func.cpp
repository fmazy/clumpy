#include <iostream>
#include "ndarray.h"

extern "C" {

int myfunc(numpyArray<double> array1, numpyArray<double> array2)
{
    Ndarray<double,3> a(array1);
    Ndarray<double,3> b(array2);

    double sum=0.0;

    for (int i = 0; i < a.getShape(0); i++)
    {
        for (int j = 0; j < a.getShape(1); j++)
        {
            for (int k = 0; k < a.getShape(2); k++)
            {
                a[i][j][k] = 2.0 * b[i][j][k];
                sum += a[i][j][k];
           }
        }
    }
    return sum;
}
} // end extern "C"

//#include<bits/stdc++.h>
//using namespace std;
//
//int multiply(double *arr_in, float factor, double *arr_out, unsigned int *shape) {
//
//  unsigned int num_rows, num_cols, row, col;
//
//  num_rows = shape[0];
//  num_cols = shape[1];
//
//  for (row=0; row<num_rows; row++) {
//    for (col=0; col<num_cols; col++) {
//      arr_out[row*num_cols + col] = factor*arr_in[row*num_cols + col];
//    }
//  }
//
//  return 0;
//}
//
//int uniques(double *arr_in, double *arr_out, unsigned int *shape) {
//
//  int num_rows, num_cols
//  num_rows = shape[0];
//  num_cols = shape[1];
//
//  unordered_set<string> uset;
//
//  for(int i = 0; i < num_rows; i++)
//    {
//        string s = "";
//
//        for(int j = 0; j < num_cols; j++)
//            s += to_string(arr[i][j]);
//
//        if(uset.count(s) == 0)
//        {
//            uset.insert(s);
//            cout << s << endl;
//
//        }
//    }
//
//  return 0;
//}
