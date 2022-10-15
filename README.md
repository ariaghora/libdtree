# libdtree

This is a single-header-file C library to create and grow decision tree classifiers.
The implementation follows C99 standard so it is compatible with most architectures.
It is small (< 500 SLOC) and suitable for embedded applications.
It is also easy for the other languages to interface with libdtree.

## API

## Example

This example demonstrates the usage of decision tree on XOR gate dataset.

```C
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
    float target[4] = {
        0,
        1,
        1,
        0};

    int nrow = 4;
    int ncol = 2;
    Tree *tree = dtree_grow(data, target, ncol, nrow);

    // make bulk predictions on the original data (which is expected to
    // reproduce the original target)
    float *res_bulk = dtree_predict(tree, data, ncol, nrow);
    for (int i = 0; i < nrow; i++)
    {
        printf("result %d: %.2f\n", i, res_bulk[i]);
    }

    // make a single prediction
    float test[2] = {1, 0};
    float res = dtree_predict_single(tree, test);
    printf("result single: %.2f\n", res); // should print 1.00

    free(res_bulk);

    dtree_free(tree);
    return 0;
}
```

## Notes
- This library only provides support for training decision tree classifiers.
    The input data is assumed to be ALL numerical.
- The data loading, data preprocessing, and other auxiliary functionalities
    are out of the scope of this library.