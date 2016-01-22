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
typedef struct QC_Buf_Str {
	const char *buf;
	int len;
} QC_Buf_Str;

bool buf_str_equals(QC_Buf_Str a, QC_Buf_Str b);
QC_Buf_Str c_str_to_buf_str(const char* str);

/* Args for printf %.*s specifier */
#define BUF_STR_ARGS(str) str.len, str.buf


/* Dynamic array */

#define QC_Array(V) JOIN2(V, _Array)
#define qc_create_array(V) JOIN3(qc_create_, V, _array)
#define qc_destroy_array(V) JOIN3(qc_destroy_, V, _array)
#define release_array(V) JOIN3(release_, V, _array)
#define qc_push_array(V) JOIN3(qc_push_, V, _array)
#define pop_array(V) JOIN3(pop_, V, _array)
#define insert_array(V) JOIN3(insert_, V, _array)
#define erase_array(V) JOIN3(erase_, V, _array)
#define qc_copy_array(V) JOIN3(qc_copy_, V, _array)
#define clear_array(V) JOIN3(clear_, V, _array)
/* Internal */
#define increase_array_capacity(V) JOIN3(increase_array_capacity_, V, _array)

#define QC_DECLARE_ARRAY(V)\
typedef struct QC_Array(V) {\
	V *data;\
	int size;\
	int capacity;\
} QC_Array(V);\
\
QC_Array(V) qc_create_array(V)(int init_capacity);\
void qc_destroy_array(V)(QC_Array(V) *arr);\
V *release_array(V)(QC_Array(V) *arr);\
void qc_push_array(V)(QC_Array(V) *arr, V value);\
V pop_array(V)(QC_Array(V) *arr);\
void insert_array(V)(QC_Array(V) *arr, int at_place, V *values, int value_count);\
void erase_array(V)(QC_Array(V) *arr, int at_place, int erase_count);\
QC_Array(V) qc_copy_array(V)(QC_Array(V) *arr);\
void clear_array(V)(QC_Array(V) *arr);\

#define DEFINE_ARRAY(V)\
QC_Array(V) qc_create_array(V)(int init_capacity)\
{\
	QC_Array(V) arr = {0};\
	if (init_capacity > 0) {\
		arr.data = (V*)malloc(init_capacity*sizeof(*arr.data));\
		arr.capacity = init_capacity;\
	}\
	return arr;\
}\
void qc_destroy_array(V)(QC_Array(V) *arr)\
{\
	ASSERT(arr);\
	free(arr->data);\
}\
V *release_array(V)(QC_Array(V) *arr)\
{\
	V *data = arr->data;\
	arr->data = NULL;\
	arr->size = 0;\
	arr->capacity = 0;\
	return data;\
}\
INTERNAL void increase_array_capacity(V)(QC_Array(V) *arr, int min_size)\
{\
	if (min_size <= arr->capacity)\
		return;\
	if (arr->capacity == 0)\
		arr->capacity = MAX(min_size, 1);\
	else\
		arr->capacity = MAX(min_size, arr->capacity*2);\
	arr->data = (V*)realloc(arr->data, arr->capacity*sizeof(*arr->data));\
}\
void qc_push_array(V)(QC_Array(V) *arr, V value)\
{\
	ASSERT(arr);\
	increase_array_capacity(V)(arr, arr->size + 1);\
	arr->data[arr->size++] = value;\
}\
void insert_array(V)(QC_Array(V) *arr, int at_place, V *values, int value_count)\
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
void erase_array(V)(QC_Array(V) *arr, int at_place, int erase_count)\
{\
	ASSERT(arr);\
	ASSERT(at_place >= 0 && at_place < arr->size);\
	ASSERT(at_place + erase_count <= arr->size);\
	ASSERT(erase_count >= 0);\
	memmove(arr->data + at_place, arr->data + at_place + erase_count, sizeof(*arr->data)*(arr->size - at_place - erase_count));\
	arr->size -= erase_count;\
}\
V pop_array(V)(QC_Array(V) *arr)\
{\
	ASSERT(arr);\
	ASSERT(arr->size > 0);\
	--arr->size;\
	return arr->data[arr->size];\
}\
QC_Array(V) qc_copy_array(V)(QC_Array(V) *arr)\
{\
	QC_Array(V) copy = {0};\
	copy.data = (V*)malloc(arr->capacity*sizeof(*arr->data));\
	memcpy(copy.data, arr->data, arr->size*sizeof(*arr->data));\
	copy.size = arr->size;\
	copy.capacity = arr->capacity;\
	return copy;\
}\
void clear_array(V)(QC_Array(V) *arr)\
{\
	ASSERT(arr);\
	arr->size = 0;\
}\


/* Hashing */

/* Hash "template" */
#define qc_hash(V) JOIN2(qc_hash_, V)

/* Hash functions should avoid generating neighbouring qc_hashes easily (linear probing) */
static U32 qc_hash(Void_Ptr)(Void_Ptr value) { return (U32)(((U64)value)/2); }



/* Hash table */
/* @todo Automatic resizing */

/* Key_Value */
#define KV(K, V) JOIN3(K, _, V)

#define qc_create_tbl(K, V) JOIN3(qc_create_, KV(K, V), _tbl)
#define qc_destroy_tbl(K, V) JOIN3(qc_destroy_, KV(K, V), _tbl)
#define get_tbl(K, V) JOIN3(get_, KV(K, V), _tbl)
#define set_tbl(K, V) JOIN3(set_, KV(K, V), _tbl)
#define null_tbl_entry(K, V) JOIN3(null_, KV(K, V), _tbl_entry)
#define QC_Hash_Table(K, V) JOIN2(KV(K, V), _Tbl)
#define QC_Hash_Table_Entry(K, V) JOIN2(KV(K, V), _Tbl_Entry)

#define QC_DECLARE_HASH_TABLE(K, V)\
struct QC_Hash_Table_Entry(K, V);\
\
typedef struct QC_Hash_Table(K, V) {\
	struct QC_Hash_Table_Entry(K, V) *array;\
	int array_size;\
	int count;\
	K null_key;\
	V null_value;\
} QC_Hash_Table(K, V);\
typedef struct QC_Hash_Table_Entry(K, V) {\
	K key;\
	V value;\
} QC_Hash_Table_Entry(K, V);\
\
QC_Hash_Table(K, V) qc_create_tbl(K, V)(	K null_key, V null_value, int max_size);\
void qc_destroy_tbl(K, V)(QC_Hash_Table(K, V) *tbl);\
\
V get_tbl(K, V)(QC_Hash_Table(K, V) *tbl, K key);\
void set_tbl(K, V)(QC_Hash_Table(K, V) *tbl, K key, V value);\

#define DEFINE_HASH_TABLE(K, V)\
QC_Hash_Table_Entry(K, V) null_tbl_entry(K, V)(QC_Hash_Table(K, V) *tbl)\
{\
	QC_Hash_Table_Entry(K, V) e = {0};\
	e.key = tbl->null_key;\
	e.value = tbl->null_value;\
	return e;\
}\
\
QC_Hash_Table(K, V) qc_create_tbl(K, V)(K null_key, V null_value, int max_size)\
{\
	int i;\
	QC_Hash_Table(K, V) tbl = {0};\
	tbl.null_key = null_key;\
	tbl.null_value = null_value;\
	tbl.array_size = max_size;\
	tbl.array = malloc(sizeof(*tbl.array)*max_size);\
	for (i = 0; i < max_size; ++i)\
		tbl.array[i] = null_tbl_entry(K, V)(&tbl);\
	return tbl;\
}\
\
void qc_destroy_tbl(K, V)(QC_Hash_Table(K, V) *tbl)\
{\
	free(tbl->array);\
	tbl->array = NULL;\
}\
\
V get_tbl(K, V)(QC_Hash_Table(K, V) *tbl, K key)\
{\
	int ix = qc_hash(K)(key) % tbl->array_size;\
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
void set_tbl(K, V)(QC_Hash_Table(K, V) *tbl, K key, V value)\
{\
	int ix = qc_hash(K)(key) % tbl->array_size;\
	ASSERT(key != tbl->null_key);\
\
	/* Linear probing */\
	while (tbl->array[ix].key != key && tbl->array[ix].key != tbl->null_key)\
		ix = (ix + 1) % tbl->array_size;\
\
	{\
		QC_Hash_Table_Entry(K, V) *entry = &tbl->array[ix];\
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
				QC_Hash_Table_Entry(K, V) e = tbl->array[ix];\
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

QC_DECLARE_ARRAY(char)
QC_DECLARE_ARRAY(int)

/* @todo Make this safe.. */
void safe_vsprintf(QC_Array(char) *buf, const char *fmt, va_list args);
void append_str(QC_Array(char) *buf, const char *fmt, ...);

#endif
