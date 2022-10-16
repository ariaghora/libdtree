/*
    This example demonstrates the usage of decision tree for XOR gate dataset.
*/

#include <stdio.h>
#include "libdtree.h"

int main()
{
    // features
    float data[8] = {
        1, 1,
        0, 1,
        1, 0,
        0, 0};

    // targets
    float target[4] = {0, 1, 1, 0};

    int nrow = 4;
    int ncol = 2;
    Tree *tree = dtree_fit(data, target, ncol, nrow);

    /*
        make bulk predictions on the original data (which is expected to
        reproduce the original target)
    */
    // allocate prediction output buffer
    float predictions[nrow];
    dtree_predict(tree, data, ncol, nrow, predictions);

    for (int i = 0; i < nrow; i++)
    {
        printf("result %d: %.2f\n", i, predictions[i]);
    }

    /*
        make a single class prediction
    */
    float test[2] = {1, 0};
    float class = dtree_predict_single(tree, test);

    printf("result single: %.2f\n", class); // should print 1.00

    dtree_free(tree);
    return 0;
}