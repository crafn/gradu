#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* memcpy */

/* C99 syntax test (currently only a subset is parsed) */

/* @todo Also typedef struct */
typedef struct Car
{
    const char *name;
    int year;
    float max_speed;
} Car; /* @todo Allow unnecessary ; */

int main()
{
    int test_integer;
    int another_test_integer = 1;

    /* @todo Make work */
    /*char test_str[] = "foo"; */

    Car car = { "john", 95, 1995.000000 };
}
