# libdtree

This is a single-header-file C library to create and grow decision tree classifiers.
The implementation follows C99 standard so it is compatible with most architectures.
It is small (< 500 SLOC) and suitable for embedded applications.
It is also easy for the other languages to interface with libdtree.

## API

This library only provides few main APIs:
### `dtree_fit`

```C
Tree dtree_fit(float *data, float *target, int ncol, int nrow);
```

Train a decision tree classifier with the given data (feature array) and target. The data represents (flatten) feature matrix in row-major order. It returns a `Tree` struct to make predictions later.

**Arguments**
- data: flatten numeric values following row-major matrix
        order
- target: target classes, encoded from 0, 1, ..., nclass-1
- ncol: number of columns (or features)
- nrow: number of samples

### `dtree_fit_with_param`
```C
Tree dtree_fit_with_param(
  float *data, float *target, int ncol, int nrow, TreeParam param
);
```
    
Same as `dtree_fit`, but instead of using default tree parameter, we should pass `param` that we define ourselves.

**Arguments**
- data: flatten numeric values following row-major matrix
        order
- target: target classes, encoded from 0, 1, ..., nclass-1
- ncol: number of columns (or features)
- nrow: number of samples
- param: the struct containing tree parameters

### `dtree_predict_single`
```C
float *dtree_predict_single(Tree tree, float *data);
```

Given a grown tree, make a single categorical prediction on the given data.

**Arguments**
- data:
        flatten numeric values following row-major matrix order
- target:
        target classes, encoded from 0, 1, ..., nclass-1

### `dtree_predict`
```C 
void dtree_predict(Tree tree, float *data, int ncol, int nrow, float *out);
```
    
Given a grown tree, make categorical predictions on the given data.

**Arguments**
- data:
    flatten numeric values following row-major matrix order
- target:
    target classes, encoded from 0, 1, ..., nclass-1
- ncol:
    number of columns (or features)
- nrow:
    number of samples
- out: output array buffer to hold the prediction result

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
```

## Notes
- This library only provides support for training decision tree classifiers.
    The input data is assumed to be ALL numerical.
- The data loading, data preprocessing, and other auxiliary functionalities
    are out of the scope of this library.