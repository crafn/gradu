int printf(const char *fmt, ..);

typedef float field(2) Field;

int main(int argc, char **argv)
{
	int size_x = 20;
	int size_y = 20;
	host Field host_field = alloc_field(size_x, size_y);
	device Field device_field_1 = alloc_field(size_x, size_y);
	device Field device_field_2 = alloc_field(size_x, size_y);

	// Init field
	{
		for (int x = 0; x < size_x; ++x) {
			for (int y = 0; y < size_y; ++y) {	
				host_field(x, y) = 0;
			}
		}
		host_field(size_x/2, size_y/2) = 1000;
	}

	// Upload contents of the host field to GPU
	memcpy_field(device_field_1, host_field);

	for (int i = 0; i < 5; ++i) {
		device Field *input = &device_field_1;
		device Field *output = &device_field_2;

		// Swap
		if (i % 2 == 1) {
			device Field *tmp = output;
			output = input;
			input = tmp;
		}

		// Diffusion!
		parallel (*output; *input) {
			int x = id(0);
			int y = id(1);
			int nx = (x + 1) % size_x;
			int ny = (y + 1) % size_y;
			int px = (x - 1 + size_x) % size_x;
			int py = (y - 1 + size_y) % size_y;

			output(x, y) =	input(x, y) +
							input(nx, y) +
							input(px, y) +
							input(x, ny) +
							input(x, py);
			output(x, y) /= 5.0;
		}

		// Download contents of GPU data back to host
		memcpy_field(host_field, *output);

		// Print current state
		for (int y = 0; y < size_y; ++y) {
			for (int x = 0; x < size_x; ++x) {	
				const char *ch = " ";
				if (host_field(x, y) > 0.5)
					ch = "#";
				else if (host_field(x, y) > 0.1)
					ch = ".";

				printf("%s", ch);
			}
			printf("\n");
		}
		printf("\n");
	}

	free_field(host_field);
	free_field(device_field_1);
	free_field(device_field_2);
	return 0;
}

