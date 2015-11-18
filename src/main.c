#include "core.h"
#include "backend_c.h"
#include "backend_cuda.h"
#include "tokenize.h"
#include "parse.h"

/* General todo list
	- if and loops
	- handle (ignore but preserve) C preprocessor
	- handle (ignore but preserve) other C stuff
	- attributes to AST (like __global__ in cuda)
	- unit tests
*/

const char *backends = "[c|cuda]";

const char *get_arg(const char **argv, int argc, const char *flag)
{
	int i;
	for (i = 1; i < argc; ++i) {
		if (argv[i][0] == '-') {
			if (!strcmp(flag, argv[i]) && i + 1 < argc) {
				return argv[i + 1];
			}
			++i;
			continue;
		}

		if (strlen(flag) == 0)
			return argv[i];
	}
	return NULL;
}

int main(int argc, const char **argv)
{
	FILE *file = NULL;
	const char *src_path;
	const char *backend_str;
	const char *output_path = NULL;
	char *src_buf = NULL;
	int src_size;
	Array(Token) tokens = {0};
	AST_Scope *root = NULL;
	Array(char) gen_code = {0};

	src_path = get_arg(argv, argc, "");
	output_path = get_arg(argv, argc, "-o");
	if (!src_path) {
		printf("Give source file as an argument\n");
		goto cleanup;
	}

	backend_str = get_arg(argv, argc, "-b");
	if (!backend_str) {
		printf("Give backend as an argument (-b %s)\n", backends);
		goto cleanup;
	}

	file = fopen(src_path, "rb");
	if (!file) {
		printf("Opening source file '%s' failed\n", src_path);
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
		ASSERT(count == 1);
		src_size = size;
	}

	{ /* Parse */
		tokens = tokenize(src_buf, src_size);
		if (!tokens.data)
			goto cleanup;
		printf("Tokens\n");
		print_tokens(tokens.data, tokens.size);

		root = parse_tokens(tokens.data);
		if (!root)
			goto cleanup;

		printf("AST\n");
		print_ast(&root->b, 2);
	}

	{ /* Output code */
		if (!strcmp(backend_str, "c")) {
			printf("C\n");
			gen_code = gen_c_code(root);
		} else if (!strcmp(backend_str, "cuda")) {
			printf("Cuda\n");
			gen_code = gen_cuda_code(root);
		} else {
			printf("Unknown backend (%s). Supported backends are %s\n", backend_str, backends);	
		}
		printf("%.*s", gen_code.size, gen_code.data);
		
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
	return 0;
}
