#include "core.h"
#include "backend_c.h"
#include "backend_cuda.h"
#include "tokenize.h"
#include "parse.h"

/* General todo list
    - accessing matrix and field elements should probably have [x, y, z] syntax instead of (x, y, z)...
	- attributes to AST (like __global__ in cuda)
	- typedef
	- handle (ignore but preserve) C preprocessor
	- handle (ignore but preserve) other C stuff
*/

const char *backends = "[c|cuda]";

/* get_arg({"-flag=foo"}, "-flag") == "foo". If flag == NULL, then first non-flag argument is given */
const char *get_arg(const char **argv, int argc, const char *flag)
{
	int i;
	for (i = 1; i < argc; ++i) {
		if (argv[i][0] == '-') {
			if (flag && !strncmp(flag, argv[i], strlen(flag))) {
				unsigned int index = strlen(flag) + 1;
				if (index > strlen(argv[i]))
					continue;
				return &argv[i][index];
			}
		} else if (!flag) {
			return argv[i];
		}
	}
	return NULL;
}

bool has_arg(const char **argv, int argc, const char *flag)
{
	int i;
	for (i = 1; i < argc; ++i) {
		if (!strcmp(flag, argv[i]))
			return true;
	}
	return false;
}

int main(int argc, const char **argv)
{
	int ret = 0;
	FILE *file = NULL;
	const char *src_path;
	const char *backend_str;
	const char *output_path = NULL;
	bool verbose;
	char *src_buf = NULL;
	int src_size;
	Array(Token) tokens = {0};
	AST_Scope *root = NULL;
	Array(char) gen_code = {0};

	verbose = has_arg(argv, argc, "-verbose");

	src_path = get_arg(argv, argc, NULL);
	output_path = get_arg(argv, argc, "-output");
	if (!src_path) {
		printf("Give source file as an argument\n");
		goto error;
	}

	backend_str = get_arg(argv, argc, "-backend");
	if (!backend_str) {
		printf("Give backend as an argument (-backend=%s)\n", backends);
		goto error;
	}

	file = fopen(src_path, "rb");
	if (!file) {
		printf("Opening source file '%s' failed\n", src_path);
		goto error;
	}

	{ /* Read file */
		int size;
		int count;
		fseek(file, 0, SEEK_END);
		size = ftell(file);
		fseek(file, 0, SEEK_SET);

		src_buf = malloc(size);
		count = fread(src_buf, size, 1, file);
		ASSERT(count == 1);
		src_size = size;
	}

	{ /* Parse */
		tokens = tokenize(src_buf, src_size);
		if (!tokens.data)
			goto error;

		if (verbose) {
			printf("Tokens\n");
			print_tokens(tokens.data, tokens.size);
		}

		root = parse_tokens(tokens.data);
		if (!root)
			goto error;

		if (verbose) {
			printf("AST\n");
			print_ast(&root->b, 2);
		}
	}

	{ /* Output code */
		if (!strcmp(backend_str, "c")) {
			gen_code = gen_c_code(root);
		} else if (!strcmp(backend_str, "cuda")) {
			gen_code = gen_cuda_code(root);
		} else {
			printf("Unknown backend (%s). Supported backends are %s\n", backend_str, backends);	
		}

		if (verbose) {
			printf("Output\n");
			printf("%.*s", gen_code.size, gen_code.data);
		}

		if (output_path) {
			FILE *out = fopen(output_path, "wb");
			fwrite(gen_code.data, strlen(gen_code.data), 1, out);
			fclose(out);
		}
	}

cleanup:
	destroy_array(char)(&gen_code);
	destroy_ast(root);
	destroy_array(Token)(&tokens);
	free(src_buf);
	if (file)
		fclose(file);
	return ret;

error:
	ret = 1;
	goto cleanup;
}
