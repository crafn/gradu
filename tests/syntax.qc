// Extended C syntax test

/**********************/
/* BLOCK COMMENT TEST */
/**********************/
int aaaaa; /* Comment explaining 'aaaaa' */ // See how comments are preserved in the output
char bbbbbb;

struct Aggregate2 {
	int c;
}

struct Aggregate {
	int a;
	int b;
	Aggregate2 c;
}


int main(int argc, const char **argv)
{
	int temp_var;
	int i;

	Aggregate foo1 = {0, 1, {2},};

	// Testing expression parsing
	temp_var = 1 + 2 * 3;

	// These will be in global scope in the output code
	struct LocalType {
		int foo;
	}
	void local_func(int p)
	{
		LocalType bug_test; // Just to test that identifier lookup works
		bug_test.foo = 123;
		aaaaa = bug_test.foo + p;
	}

	local_func(10);

	{ /* Useless inner scope */
		int test;
		test = 5;
		aaaaa = test;
	}

	if (1) {
		/* Foo */
		for (i = 0; i < 10; i = i + 1) { // @todo ++i and i++
			temp_var = temp_var + 1;
		}
	} else if (2)
		for (i = 0; i < 10; i = i + 1)
			;
	else {
		/* Bar */
		if (1)
			i = 2; /* Comment */
		while (i)
			i = i - 1; // @todo --i and i--
	}

	return foo1.a;
}

