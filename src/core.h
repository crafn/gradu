#ifndef CORE_H
#define CORE_H

/* Commonly used utils */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>

typedef uint32_t U32;
typedef uint64_t U64;
typedef void *Void_Ptr; /* Just for some macro fiddling */

/* Usage: FAIL(("Something %i", 10)) */
#define FAIL(args) do { printf("INTERNAL FAILURE: "); printf args; printf("\n"); abort(); } while(0)
#define ASSERT(x) assert(x)

#define NONULL(x) nonull_impl(x)
void *nonull_impl(void *ptr) { if (!ptr) abort(); return ptr; }

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef enum { false, true } bool;

#define INTERNAL static
#define LOCAL_PERSIST static

#define JOIN2_IMPL(A, B) A##B
#define JOIN2(A, B) JOIN2_IMPL(A, B)

#define JOIN3_IMPL(A, B, C) A##B##C
#define JOIN3(A, B, C) JOIN3_IMPL(A, B, C)

/* Not terminated by NULL! */
typedef struct Buf_Str {
	const char *buf;
	int len;
} Buf_Str;

bool buf_str_equals(Buf_Str a, Buf_Str b);
Buf_Str c_str_to_buf_str(const char* str);

/* Args for printf %.*s specifier */
#define BUF_STR_ARGS(str) str.len, str.buf


/* Dynamic array */

#define Array(V) JOIN2(V, _Array)
#define create_array(V) JOIN3(create_, V, _array)
#define destroy_array(V) JOIN3(destroy_, V, _array)
#define release_array(V) JOIN3(release_, V, _array)
#define push_array(V) JOIN3(push_, V, _array)
#define pop_array(V) JOIN3(pop_, V, _array)
#define insert_array(V) JOIN3(insert_, V, _array)
#define copy_array(V) JOIN3(copy_, V, _array)
/* Internal */
#define increase_array_capacity(V) JOIN3(increase_array_capacity, V, _array)

#define DECLARE_ARRAY(V)\
typedef struct Array(V) {\
	V *data;\
	int size;\
	int capacity;\
} Array(V);\
\
Array(V) create_array(V)(int init_capacity);\
void destroy_array(V)(Array(V) *arr);\
V *release_array(V)(Array(V) *arr);\
void push_array(V)(Array(V) *arr, V value);\
V pop_array(V)(Array(V) *arr);\
void insert_array(V)(Array(V) *arr, int at_place, V *values, int value_count);\
Array(V) copy_array(V)(Array(V) *arr);\

#define DEFINE_ARRAY(V)\
Array(V) create_array(V)(int init_capacity)\
{\
	Array(V) arr = {0};\
	if (init_capacity > 0) {\
		arr.data = (V*)malloc(init_capacity*sizeof(*arr.data));\
		arr.capacity = init_capacity;\
	}\
	return arr;\
}\
void destroy_array(V)(Array(V) *arr)\
{\
	ASSERT(arr);\
	free(arr->data);\
}\
V *release_array(V)(Array(V) *arr)\
{\
	V *data = arr->data;\
	arr->data = NULL;\
	arr->size = 0;\
	arr->capacity = 0;\
	return data;\
}\
INTERNAL void increase_array_capacity(V)(Array(V) *arr, int min_size)\
{\
	if (min_size <= arr->capacity)\
		return;\
	if (arr->capacity == 0)\
		arr->capacity = MAX(min_size, 1);\
	else\
		arr->capacity = MAX(min_size, arr->capacity*2);\
	arr->data = (V*)realloc(arr->data, arr->capacity*sizeof(*arr->data));\
}\
void push_array(V)(Array(V) *arr, V value)\
{\
	ASSERT(arr);\
	increase_array_capacity(V)(arr, arr->size + 1);\
	arr->data[arr->size++] = value;\
}\
void insert_array(V)(Array(V) *arr, int at_place, V *values, int value_count)\
{\
	int move_count = arr->size - at_place;\
	ASSERT(arr);\
	ASSERT(at_place >= 0 && at_place <= arr->size);\
	ASSERT(move_count >= 0);\
	increase_array_capacity(V)(arr, arr->size + value_count);\
	memmove(arr->data + at_place + value_count, arr->data + at_place, sizeof(*arr->data)*move_count);\
	memcpy(arr->data + at_place, values, sizeof(*arr->data)*value_count);\
	arr->size += value_count;\
}\
V pop_array(V)(Array(V) *arr)\
{\
	ASSERT(arr);\
	ASSERT(arr->size > 0);\
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


/* Hashing */

/* Hash "template" */
#define hash(V) JOIN2(hash_, V)

/* Hash functions should avoid generating neighbouring hashes easily (linear probing) */
static U32 hash(Void_Ptr)(Void_Ptr value) { return (U32)(((U64)value)/2); }



/* Hash table */

/* Key_Value */
#define KV(K, V) JOIN3(K, _, V)

#define create_tbl(K, V) JOIN3(create_, KV(K, V), _tbl)
#define destroy_tbl(K, V) JOIN3(destroy_, KV(K, V), _tbl)
#define get_tbl(K, V) JOIN3(get_, KV(K, V), _tbl)
#define set_tbl(K, V) JOIN3(set_, KV(K, V), _tbl)
#define null_tbl_entry(K, V) JOIN3(null_, KV(K, V), _tbl_entry)
#define Hash_Table(K, V) JOIN2(KV(K, V), _Tbl)
#define Hash_Table_Entry(K, V) JOIN2(KV(K, V), _Tbl_Entry)

#define DECLARE_HASH_TABLE(K, V)\
struct Hash_Table_Entry(K, V);\
\
typedef struct Hash_Table(K, V) {\
	struct Hash_Table_Entry(K, V) *array;\
	int array_size;\
	int count;\
	K null_key;\
	V null_value;\
} Hash_Table(K, V);\
typedef struct Hash_Table_Entry(K, V) {\
	K key;\
	V value;\
} Hash_Table_Entry(K, V);\
\
Hash_Table(K, V) create_tbl(K, V)(	K null_key, V null_value, int max_size);\
void destroy_tbl(K, V)(Hash_Table(K, V) *tbl);\
\
V get_tbl(K, V)(Hash_Table(K, V) *tbl, K key);\
void set_tbl(K, V)(Hash_Table(K, V) *tbl, K key, V value);\

#define DEFINE_HASH_TABLE(K, V)\
Hash_Table_Entry(K, V) null_tbl_entry(K, V)(Hash_Table(K, V) *tbl)\
{\
	Hash_Table_Entry(K, V) e = {0};\
	e.key = tbl->null_key;\
	e.value = tbl->null_value;\
	return e;\
}\
\
Hash_Table(K, V) create_tbl(K, V)(	K null_key, V null_value, int max_size)\
{\
	int i;\
	Hash_Table(K, V) tbl = {0};\
	tbl.null_key = null_key;\
	tbl.null_value = null_value;\
	tbl.array_size = max_size;\
	tbl.array = malloc(sizeof(*tbl.array)*max_size);\
	for (i = 0; i < max_size; ++i)\
		tbl.array[i] = null_tbl_entry(K, V)(&tbl);\
	return tbl;\
}\
\
void destroy_tbl(K, V)(Hash_Table(K, V) *tbl)\
{\
	free(tbl->array);\
	tbl->array = NULL;\
}\
\
V get_tbl(K, V)(Hash_Table(K, V) *tbl, K key)\
{\
	int ix = hash(K)(key) % tbl->array_size;\
	/* Linear probing */\
	/* Should not be infinite because set_id_handle_tbl asserts if table is full */\
	while (tbl->array[ix].key != key && tbl->array[ix].key != tbl->null_key)\
		ix= (ix + 1) % tbl->array_size;\
\
	if (tbl->array[ix].key == tbl->null_key)\
		ASSERT(tbl->array[ix].value == tbl->null_value);\
\
	return tbl->array[ix].value;\
}\
\
void set_tbl(K, V)(Hash_Table(K, V) *tbl, K key, V value)\
{\
	int ix = hash(K)(key) % tbl->array_size;\
	ASSERT(key != tbl->null_key);\
\
	/* Linear probing */\
	while (tbl->array[ix].key != key && tbl->array[ix].key != tbl->null_key)\
		ix = (ix + 1) % tbl->array_size;\
\
	{\
		Hash_Table_Entry(K, V) *entry = &tbl->array[ix];\
		bool modify_existing = 	value != tbl->null_value && entry->key != tbl->null_key;\
		bool insert_new =		value != tbl->null_value && entry->key == tbl->null_key;\
		bool remove_existing =	value == tbl->null_value && entry->key != tbl->null_key;\
		bool remove_new =		value == tbl->null_value && entry->key == tbl->null_key;\
	\
		if (modify_existing) {\
			entry->value = value;\
		} else if (insert_new) {\
			entry->key = key;\
			entry->value = value;\
			++tbl->count;\
		} else if (remove_existing) {\
			entry->key = key;\
			entry->key = tbl->null_key;\
			entry->value = tbl->null_value;\
			ASSERT(tbl->count > 0);\
			--tbl->count;\
	\
			/* Rehash */\
			ix= (ix + 1) % tbl->array_size;\
			while (tbl->array[ix].key != tbl->null_key) {\
				Hash_Table_Entry(K, V) e = tbl->array[ix];\
				tbl->array[ix] = null_tbl_entry(K, V)(tbl);\
				--tbl->count;\
				set_tbl(K, V)(tbl, e.key, e.value);\
	\
				ix= (ix + 1) % tbl->array_size;\
			}\
		} else if (remove_new) {\
			/* Nothing to be removed */\
		} else {\
			FAIL(("Hash table logic failed"));\
		}\
	}\
\
	ASSERT(tbl->count < tbl->array_size);\
}\

DECLARE_ARRAY(char)
DECLARE_ARRAY(int)

/* @todo Make this safe.. */
void safe_vsprintf(Array(char) *buf, const char *fmt, va_list args);
void append_str(Array(char) *buf, const char *fmt, ...);

#endif
