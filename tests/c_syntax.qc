// C99 syntax test (currently only a subset is parsed)

// @todo Also typedef struct
struct Car {
	const char *name;
	int year;
	float max_speed;
} // @todo Allow unnecessary ;

int main()
{
	int test_integer;
	test_integer = -134;
	int another_test_integer = 1;

	// @todo Make work
	//char test_str[] = "foo";

	// @todo Conversion to C89 (because decl is separated from init)
	Car car = {
		"aaa",
		95,
		1995.0
	};

	// @todo Need to make conversion from compound literals and designated initializers to C89
	car = (Car) {
		.year = 1,
		.name = "bbb",
		.max_speed = -2.0
	};

	car.year = 5;

	bool b;
	b = true;
	b = false;

	/* @todo Rest of C99 */

	/* @todo Make work */
	//(void)car;
	//(void)test_integer;
	//(void)another_test_integer;
	int x = car.year + test_integer + another_test_integer + b;
	return x;
}
