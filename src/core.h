#ifndef CORE_H
#define CORE_H

/* Commonly used utils */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Usage: FAIL(("Something %i", 10)) */
#define FAIL(args) do { printf("INTERNAL FAILURE: "); printf args; printf("\n"); abort(); } while(0)

typedef enum { false, true } bool;

#define INTERNAL static
#define LOCAL_PERSIST static

#define JOIN2_IMPL(A, B) A##B
#define JOIN2(A, B) JOIN2_IMPL(A, B)

#define JOIN3_IMPL(A, B, C) A##B##C
#define JOIN3(A, B, C) JOIN3_IMPL(A, B, C)

/* Dynamic array */

#define Array(V) JOIN2(V, _Array)
#define create_array(V) JOIN3(create_, V, _array)
#define destroy_array(V) JOIN3(destroy_, V, _array)
#define push_array(V) JOIN3(push_, V, _array)
#define pop_array(V) JOIN3(pop_, V, _array)
#define copy_array(V) JOIN3(copy_, V, _array)

#define DECLARE_ARRAY(V)\
typedef struct Array(V) {\
	V *data;\
	int size;\
	int capacity;\
} Array(V);\
\
Array(V) create_array(V)(int capacity);\
void destroy_array(V)(Array(V) *arr);\
void push_array(V)(Array(V) *arr, V value);\
V pop_array(V)(Array(V) *arr);\
Array(V) copy_array(V)(Array(V) *arr);\

#define DEFINE_ARRAY(V)\
Array(V) create_array(V)(int capacity)\
{\
	Array(V) arr = {0};\
	arr.data = (V*)malloc(capacity*sizeof(*arr.data));\
	arr.capacity = capacity;\
	return arr;\
}\
void destroy_array(V)(Array(V) *arr)\
{\
	assert(arr);\
	free(arr->data);\
}\
void push_array(V)(Array(V) *arr, V value)\
{\
	assert(arr);\
	if (arr->size >= arr->capacity) {\
		if (arr->capacity == 0)\
			arr->capacity = 1;\
		else\
			arr->capacity *= 2;\
		arr->data = (V*)realloc(arr->data, arr->capacity*sizeof(*arr->data));\
	}\
	arr->data[arr->size++] = value;\
}\
V pop_array(V)(Array(V) *arr)\
{\
	assert(arr);\
	assert(arr->size > 0);\
	--arr->size;\
	return arr->data[arr->size];\
}\
Array(V) copy_array(V)(Array(V) *arr)\
{\
	Array(V) copy = {0};\
	copy.data = (V*)malloc(arr->capacity*sizeof(*arr->data));\
	memcpy(copy.data, arr->data, arr->size*sizeof(*arr->data));\
	copy.size = arr->size;\
	copy.capacity = arr->capacity;\
	return copy;\
}\

#endif
