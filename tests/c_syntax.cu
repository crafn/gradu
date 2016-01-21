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
    test_integer = -134;
    int another_test_integer = 1;

    /* @todo Make work */
    /*char test_str[] = "foo"; */

    /* @todo This parses, but needs conversion to C89 (because decl is separated from init) */
    Car car;

    /* @todo This parses, but need to make conversion from compound literals and designated initializers to C89 */
    /*car = (Car) {
		.year = 1,
		.name = "bbb",
		.max_speed = -2.0
	};*/

    /* @todo Rest of C99 */

    /* @todo Make work */
    /*(void)car; */
    /*(void)test_integer; */
    /*(void)another_test_integer; */
    int x = car.year + test_integer + another_test_integer;
    return x;
}
