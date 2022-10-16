/*  Decision tree classifier - Aria Ghora Prabono 2022

    This is a single-header-file C library to create and grow decision tree
    classifier. The implementation follows C99 standard so it is compatible
    with most architectures. It is small and suitable for embedded applications.
    It is also easy to interface from other languages.

    Algorithm variants:
        - ID3
        - C4.5 [TODO]

DOCUMENTATION

    Functions (macros):

        dtree_fit
            Tree dtree_fit(float *data, float *target, int ncol, int nrow);
                Train a decision tree classifier with the given data (feature
                array) and target. The data represents (flatten) feature matrix
                in row-major order.

                Arguments
                ---------
                    data: flatten numeric values following row-major matrix
                        order
                    target: target classes, encoded from 0, 1, ..., nclass-1
                    ncol: number of columns (or features)
                    nrow: number of samples


        dtree_predict_single
            float *dtree_predict(Tree tree, float *data);
                Given a grown tree, make categorical predictions on the given
                data.

                Arguments
                ---------
                    data:
                        flatten numeric values following row-major matrix order
                    target:
                        target classes, encoded from 0, 1, ..., nclass-1


        dtree_predict
            void dtree_predict(
                Tree tree, float *data, int ncol, int nrow, float *out
            );
                Given a grown tree, make categorical predictions on the given
                data.

                Arguments
                ---------
                    data:
                        flatten numeric values following row-major matrix order
                    target:
                        target classes, encoded from 0, 1, ..., nclass-1
                    ncol:
                        number of columns (or features)
                    nrow:
                        number of samples
                    out: output array buffer to hold the prediction result

NOTES

    * This library only provides support for training decision tree classifier.
      The input data is assumed to be ALL numerical.
    * The data loading, data preprocessing, and other auxillary functionalities
      are out the scope of this library.

 */

#ifndef LIBDTREE_H_
#define LIBDTREE_H_

////////////////////////////////////////////////////////////////////////////////
//
// Internal consts
//
////////////////////////////////////////////////////////////////////////////////

// Aliases
#define dtree_fit(data, target, ncol, nrow) \
    dtree_grow(data, target, ncol, nrow);
#define dtree_fit_with_param(data, target, ncol, nrow, param) \
    dtree_grow_with_param(data, target, ncol, nrow, param);

// Consts
#define MIN_LIST_CAP 16

// API prototypes

typedef struct Tree Tree;
typedef struct TreeParam TreeParam;

Tree* dtree_grow(float* data, float* target, int ncol, int nrow);
Tree* dtree_grow_with_param(float* data, float* target, int ncol, int nrow,
                            TreeParam param);
float dtree_predict_single(Tree* tree, float* data);
void dtree_predict(Tree* tree, float* data, int ncol, int nrow, float* out);

////////////////////////////////////////////////////////////////////////////////
//
// The rest of this file is the implementation
//
////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Some helper data structure for dynamic array

typedef struct {
    int cap;
    int len;
    float* data;
} List;

List ldt_listalloc() {
    float* data = (float*)malloc(MIN_LIST_CAP * sizeof(*data));
    List l = {.cap = MIN_LIST_CAP, .len = 0, .data = data};
    return l;
}

void ldt_listpush(List* l, float val) {
    l->data[l->len] = val;
    ++l->len;
    if (l->len >= l->cap / 2) {
        l->cap *= 2;
        l->data = (float*)realloc(l->data, l->cap * sizeof(*l->data));
    }
}

void ldt_listfree(List* l) {
    free(l->data);
    l->data = NULL;
}

int ldt_listcontains(List l, float x) {
    for (int i = 0; i < l.len; i++)
        if (l.data[i] == x) return 1;
    return 0;
}

List ldt_listunique(float* arr, int arrlen) {
    List found = ldt_listalloc();
    for (int i = 0; i < arrlen; i++)
        if (!ldt_listcontains(found, arr[i])) ldt_listpush(&found, arr[i]);
    return found;
}

//
// Decision tree implementations

struct Tree {
    int isleaf;
    float value;
    int featidx;
    float thresh;
    float gain;
    Tree* lnode;
    Tree* rnode;
};

struct TreeParam {
    int maxdepth;
    int min_sample_split;
};

typedef struct Split {
    int featidx;
    float thresh;
    float* ldata;
    float* ltarget;
    int lnrow;
    float* rdata;
    float* rtarget;
    int rnrow;
    float gain;
} Split;

// in-place element-wise array division
inline void ldt_arrdiv(float* x, float div, long arrlen) {
    for (long i = 0; i < arrlen; i++) x[i] /= div;
}

inline float ldt_arrmax(float* x, long arrlen) {
    int cmax = 0;
    for (long i = 0; i < arrlen; i++) cmax = x[i] > cmax ? x[i] : cmax;
    return cmax;
}

float* ldt_bincount(float* x, long arrlen) {
    int max = (int)(ldt_arrmax(x, arrlen)) + 1;
    float* res = (float*)calloc(max, sizeof(float));
    for (int i = 0; i < arrlen; i++) res[(int)(x[i])] += 1;
    return res;
}

Tree* classify_asleaf(float* target, int arrlen) {
    int max = (int)ldt_arrmax(target, arrlen);
    int cnt[max + 1];
    for (int i = 0; i < max + 1; i++) cnt[i] = 0;

    for (int i = 0; i < arrlen; i++) {
        cnt[(int)target[i]]++;
    }

    int idxmax = 0;
    int currmax = cnt[idxmax];
    for (int i = 0; i < max + 1; i++) {
        if (cnt[i] > currmax) {
            idxmax = i;
            currmax = cnt[i];
        }
    }

    Tree* n = (Tree*)malloc(sizeof(*n));
    n->isleaf = 1;
    n->value = (float)idxmax;
    n->lnode = NULL;
    n->rnode = NULL;
    return n;
}

static inline float entropy(float* x, long arrlen) {
    float* bc = ldt_bincount(x, arrlen);
    int max = (int)ldt_arrmax(x, arrlen);
    ldt_arrdiv(bc, (float)arrlen, max + 1);

    float entropy = 0.0;
    List u = ldt_listunique(x, arrlen);
    for (int i = 0; i < u.len; i++) {
        if (bc[(int)u.data[i]] > 0)
            entropy += bc[(int)u.data[i]] * log2f(bc[(int)u.data[i]]);
    }

    free(bc);
    ldt_listfree(&u);
    return -entropy;
}

static inline float gain(float* parent, float* left, float* right, int nparent,
                         int nleft, int nright) {
    float lprop = nleft / (float)nparent;
    float rprop = nright / (float)nparent;
    return (entropy(parent, nparent) -
            (lprop * entropy(left, nleft) + rprop * entropy(right, nright)));
}

inline void ldt_getcol(float* data, int idxcol, int ncol, int nrow,
                       float* dst) {
    for (int i = 0; i < nrow; i++) dst[i] = data[idxcol + ncol * i];
}

int ispure(float* data, int arrlen) {
    List unique = ldt_listunique(data, arrlen);
    int res = unique.len == 1;
    ldt_listfree(&unique);
    return res;
}

Split best_split(float* data, float* target, int ncol, int nrow) {
    Split split;
    split.ldata = NULL;
    split.rdata = NULL;
    split.ltarget = NULL;
    split.rtarget = NULL;
    split.featidx = 0;
    split.thresh = 0;
    split.lnrow = 0;
    split.rnrow = 0;
    split.gain = 0;
    float bestgain = -1;

    // buffer to get each column data (the f-th) in the following iteration
    float xcol[nrow];

    for (int f = 0; f < ncol; f++) {
        ldt_getcol(data, f, ncol, nrow, xcol);
        List unique = ldt_listunique(xcol, nrow);

        // iterater over theslholds (i.e., the unique values) and take
        // the best one
        for (int i = 0; i < unique.len; i++) {
            float leftdata[ncol * nrow];
            float lefttarget[nrow];
            float rightdata[ncol * nrow];
            float righttarget[nrow];
            int lnrow = 0;
            int rnrow = 0;
            int loffset = 0;
            int roffset = 0;

            // split data
            for (int row = 0; row < nrow; row++) {
                if (data[f + ncol * row] <= unique.data[i]) {
                    for (int c = 0; c < ncol; c++)
                        leftdata[loffset++] = data[c + ncol * row];
                    lefttarget[lnrow++] = target[row];
                } else {
                    for (int c = 0; c < ncol; c++)
                        rightdata[roffset++] = data[c + ncol * row];
                    righttarget[rnrow++] = target[row];
                }
            }

            // obtain a split with best gain
            if ((lnrow > 0) && (rnrow > 0)) {
                float g = gain(target, lefttarget, righttarget, nrow, lnrow, rnrow);

                if (g > bestgain) {
                    // set the current best split
                    split.gain = g;
                    split.featidx = f;
                    split.thresh = unique.data[i];
                    split.lnrow = lnrow;
                    split.rnrow = rnrow;

                    // free the buffers before (re)allocating
                    free(split.ldata), free(split.ltarget);
                    free(split.rdata), free(split.rtarget);

                    split.ldata = (float*)malloc(loffset * sizeof(*leftdata));
                    split.ltarget = (float*)malloc(lnrow * sizeof(*lefttarget));
                    split.rdata = (float*)malloc(roffset * sizeof(*rightdata));
                    split.rtarget = (float*)malloc(rnrow * sizeof(*lefttarget));
                    memcpy(split.ldata, leftdata, loffset * sizeof(*leftdata));
                    memcpy(split.ltarget, lefttarget, lnrow * sizeof(*lefttarget));
                    memcpy(split.rdata, rightdata, roffset * sizeof(*rightdata));
                    memcpy(split.rtarget, righttarget, rnrow * sizeof(*righttarget));
                }
            }
        }
        ldt_listfree(&unique);
    }

    return split;
}

Tree* ldt_grow(float* data, float* target, int ncol, int nrow, int depth,
               TreeParam param) {
    if (ispure(target, nrow) || (nrow < param.min_sample_split) ||
        (depth == param.maxdepth)) {
        Tree* res = classify_asleaf(target, nrow);
        return res;
    } else {
        Split best = best_split(data, target, ncol, nrow);
        Tree* left =
            ldt_grow(best.ldata, best.ltarget, ncol, best.lnrow, depth + 1, param);
        Tree* right =
            ldt_grow(best.rdata, best.rtarget, ncol, best.rnrow, depth + 1, param);

        Tree* n = (Tree*)malloc(sizeof(*n));
        n->featidx = best.featidx;
        n->thresh = best.thresh;
        n->isleaf = 0;
        n->gain = best.gain;
        n->lnode = left;
        n->rnode = right;

        free(best.ldata), free(best.ltarget);
        free(best.rdata), free(best.rtarget);

        return n;
    }
}

void dtree_free(Tree* tree) {
    if (!tree->isleaf) {
        dtree_free(tree->lnode);
        dtree_free(tree->rnode);
    }
    free(tree);
}

Tree* dtree_grow_with_param(float* data, float* target, int ncol, int nrow,
                            TreeParam param) {
    return ldt_grow(data, target, ncol, nrow, 0, param);
}

Tree* dtree_grow(float* data, float* target, int ncol, int nrow) {
    TreeParam defaultparam = {
        .maxdepth = 5,
        .min_sample_split = 1,
    };
    return ldt_grow(data, target, ncol, nrow, 0, defaultparam);
}

float dtree_predict_single(Tree* tree, float* data) {
    if (tree->isleaf) return tree->value;

    float feat = data[tree->featidx];

    // recurse to the left direction
    if (feat <= tree->thresh) return dtree_predict_single(tree->lnode, data);
    // recurse to the right direction
    return dtree_predict_single(tree->rnode, data);
}

void dtree_predict(Tree* tree, float* data, int ncol, int nrow, float* out) {
    long cnt = 0;
    for (int i = 0; i < nrow * ncol; i += ncol) {
        float row[ncol];
        for (int j = i; j < ncol + i; j++) {
            row[j - i] = data[j];
        }
        float pred = dtree_predict_single(tree, row);
        out[cnt++] = pred;
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// Unit testing
//
////////////////////////////////////////////////////////////////////////////////

#ifdef LIBDTREE_TEST_

#include <stdio.h>

void assert_eq_int(int x, int y, char* title) {
    if (x == y)
        printf("\033[32m[PASS] %s\033[0m\n", title);
    else
        printf("\033[31m[FAIL] %s: left=%d right=%d\033[31m\n", title, x, y);
}

void assert_eq_float(float x, float y, char* title) {
    if (x == y)
        printf("\033[32m[PASS] %s\033[0m\n", title);
    else
        printf("\033[31m[FAIL] %s: left=%f right=%f\033[31m\n", title, x, y);
}

void test_list() {
    List l = ldt_listalloc();
    ldt_listpush(&l, 1), ldt_listpush(&l, 2), ldt_listpush(&l, 3);
    assert_eq_int(l.len, 3, "list_len_equals_to_3");
    assert_eq_float(l.data[1], 2, "2nd_list_elem_equals_to_2");
    ldt_listfree(&l);
}

void test_arrunique() {
    float arr[7] = {1, 2, 3, 3, 4, 5, 4};
    float* bc = ldt_bincount(arr, 7);
    assert_eq_float(bc[0], 0, "test_bincount_0");
    assert_eq_float(bc[1], 1, "test_bincount_1");
    assert_eq_float(bc[2], 1, "test_bincount_2");
    assert_eq_float(bc[3], 2, "test_bincount_3");
    assert_eq_float(bc[4], 2, "test_bincount_4");
    assert_eq_float(bc[5], 1, "test_bincount_5");
    free(bc);
}

void test_ispure() {
    float arr_impure[7] = {1, 2, 3, 3, 4, 5, 4};
    float arr_pure[7] = {1, 1, 1, 1, 1, 1, 1};
    assert_eq_int(ispure(arr_impure, 7), 0, "test_impure");
    assert_eq_int(ispure(arr_pure, 7), 1, "test_pure");
}

void test_classify() {
    float case1[5] = {1, 1, 2, 2, 2};
    Tree* res1 = classify_asleaf(case1, 5);
    assert_eq_int(res1->isleaf, 1, "test_classification_res_is_a_leaf");
    assert_eq_int(res1->value, 2, "test_classification_case1_is_correct");

    float case2[5] = {1, 1, 1, 1, 1};
    Tree* res2 = classify_asleaf(case2, 5);
    assert_eq_int(res2->value, 1, "test_classification_case2_is_correct");

    float case3[1] = {8};
    Tree* res3 = classify_asleaf(case3, 1);
    assert_eq_int(res3->value, 8, "test_classification_case3_is_correct");
}

void test_arrdiv() {
    float arr[2] = {4, 4};
    ldt_arrdiv(arr, 2, 2);
    assert_eq_int(arr[0], 2, "test_4/4=2");
    assert_eq_int(arr[1], 2, "test_4/4=2");
    ldt_arrdiv(arr, 2, 2);
    assert_eq_int(arr[0], 1, "test_2/2=1");
    assert_eq_int(arr[1], 1, "test_2/2=1");
}

void run_tests() {
    test_list();
    test_arrunique();
    test_ispure();
    test_classify();
    test_arrdiv();
}

#endif

#endif