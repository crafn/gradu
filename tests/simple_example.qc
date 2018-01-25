int printf(const char *fmt, ..); // TODO: Remove

typedef int field(1) Field; // One-dimensional integer field type

int main()
{
    int N = 5; 
	
	host Field a_data = alloc_field(N);
	host Field b_data = alloc_field(N);
	
	// TODO: Initializers
	a_data(0) = 1;
	a_data(1) = 2;
	a_data(2) = 3;
	a_data(3) = 4;
	a_data(4) = 5;
	
	b_data(0) = 10;
	b_data(1) = 20;
	b_data(2) = 30;
	b_data(3) = 40;
	b_data(4) = 50;
	
	device Field a = alloc_field(N);
	device Field b = alloc_field(N);
	
	memcpy_field(a, a_data);
	memcpy_field(b, b_data);
	
	parallel (a; b) {
		a(id(0)) += b(id(0));
	}

	memcpy_field(a_data, a);
	
	for (int i = 0; i < N; ++i) {
		printf("%i ", a_data(i)); // Prints 11 22 33 44 55
	}
	
	free_field(a_data);
	free_field(b_data);
	free_field(a);
	free_field(b);

	return 0;
}

