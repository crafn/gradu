#include <assert.h>
#include <stdio.h>

#include "tokenize.h"

int main(int argc, const char **argv)
{
	FILE *file = NULL;
	char *src_buf = NULL;
	int src_size;
	Array(Token) tokens = {0};

	if (argc < 2) {
		printf("Give source file as argument\n");
		goto cleanup;
	}

	file = fopen(argv[1], "rb");
	if (!file) {
		printf("Opening source file failed\n");
		goto cleanup;
	}

	{ /* Read file */
		int size;
		int count;
		fseek(file, 0, SEEK_END);
		size = ftell(file);
		fseek(file, 0, SEEK_SET);

		src_buf = malloc(size);
		count = fread(src_buf, size, 1, file);
		assert(count == 1);
		src_size = size;
	}

	tokens = tokenize(src_buf, src_size);
	print_tokens(tokens.data, tokens.size);

cleanup:
	destroy_array(Token)(&tokens);
	free(src_buf);
	if (file)
		fclose(file);
	return 0;
}
