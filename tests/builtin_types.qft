
int main(int argc, char **argv)
{
	int matrix(2, 2) mat1;
	int matrix(2, 2) mat2;
	float matrix(3, 3) mat3;
	float matrix(3, 3) mat4;
	int i;
	int k;

	mat1(0, 0) = 0;
	mat1(1, 0) = 1;
	mat1(1, 1) = 2;
	mat1(0, 1) = 3;

	mat2(0, 0) = 0;
	mat2(1, 0) = 1;
	mat2(1, 1) = 2;
	mat2(0, 1) = 3;

	for (i = 0; i < 3; i = i + 1) {
		for (k = 0; k < 3; k = k + 1) {
			mat3(i, k) = 1;
			mat4(i, k) = 2;
		}
	}

	// This comment should survive to the generated code
	mat1 = mat1 * mat2;
	mat3 = mat4 * mat3 * mat3 * mat3;

	return 0;
}
