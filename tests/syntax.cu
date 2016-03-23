#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* memcpy */

/* Extended C syntax test */

/**********************/
/* BLOCK COMMENT TEST */
/**********************/
int aaaaa; /* Comment explaining 'aaaaa' */ /* See how comments are preserved in the output */
char bbbbbb;

typedef struct Aggregate2
{
    int c;
} Aggregate2;

typedef struct Aggregate
{
    int a;
    int b;
    Aggregate2 c;
} Aggregate;

/* These will be in global scope in the output code */
typedef struct LocalType
{
    int foo;
} LocalType;
void local_func(int p)
{
    LocalType bug_test; /* Just to test that identifier lookup works */
    bug_test.foo = 123;
    aaaaa = bug_test.foo + p;
}
int main(int argc, const char **argv)
{
    int temp_var;
    int i;

    Aggregate foo1 = {
        0,
        1,
        { 2 }
    };
    temp_var = 1 + 2 * 3;
    local_func(10);

    {
        int test;
        test = 5;
        aaaaa = test;
    }

    if (1) {
        /* Foo */
        for (i = 0; i < 10; i = i + 1) {
            temp_var = temp_var + 1;
        }
    } else if (2) {
        for (i = 0; i < 10; i = i + 1) 
            ;
    } else {
        /* Bar */
        if (1) {
            i = 2;
        }
        while (i) {
            i = i - 1;
        }
    }

    return foo1.a;
}

