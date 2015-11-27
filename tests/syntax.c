#include <stdio.h>
#include <stdlib.h>

/**********************/
/* BLOCK COMMENT TEST */
/**********************/
int aaaaa; /* Comment explaining 'aaaaa' */ /* See how comments are preserved in the output */
char bbbbbb;

/* These will be in global scope in the output code */
typedef struct LocalType
{
    int foo;
} LocalType;
void local_func()
{
    LocalType bug_test; /* Just to test that identifier lookup works */
    (bug_test).foo = 123;
    aaaaa = (bug_test).foo;
}

int main(int argc, const char **argv)
{
    int temp_var;
    int i;

    /* Testing expression parsing */
    temp_var = 1 + 2 * 3;

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
            i = 2; /* Comment */
        }
        while (i) {
            i = i - 1; /* @todo --i and i-- */
        }
    }

    return 0;
}
