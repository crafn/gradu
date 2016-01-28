#include "core.h"
#include "backend_c.h"
#include "backend_cuda.h"
#include "tokenize.h"
#include "parse.h"

/* General todo list
    - tests with different dimensional fields
    - cuda error checking
	- overloading test
	- attributes to QC_AST (like __global__ in cuda)
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

QC_Bool has_arg(const char **argv, int argc, const char *flag)
{
	int i;
	for (i = 1; i < argc; ++i) {
		if (!strcmp(flag, argv[i]))
			return QC_true;
	}
	return QC_false;
}

int main(int argc, const char **argv)
{
	int ret = 0;
	FILE *file = NULL;
	const char *src_path;
	const char *backend_str;
	const char *output_path = NULL;
	QC_Bool verbose;
	char *src_buf = NULL;
	int src_size;
	QC_Array(QC_Token) tokens = {0};
	QC_AST_Scope *root = NULL;
	QC_Array(char) gen_code = {0};

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

		src_buf = QC_MALLOC(size);
		count = fread(src_buf, size, 1, file);
		QC_ASSERT(count == 1);
		src_size = size;
	}

	{ /* Parse */
		tokens = qc_tokenize(src_buf, src_size);
		if (!tokens.data)
			goto error;

		if (verbose) {
			printf("QC_Tokens\n");
			qc_print_tokens(tokens.data, tokens.size);
		}

		root = qc_parse_tokens(tokens.data, QC_false);
		if (!root)
			goto error;

		if (verbose) {
			printf("QC_AST\n");
			qc_print_ast(&root->b, 2);
		}
	}

	{ /* Output code */
		if (!strcmp(backend_str, "c")) {
			gen_code = qc_gen_c_code(root);
		} else if (!strcmp(backend_str, "cuda")) {
			gen_code = qc_gen_cuda_code(root);
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
	qc_destroy_array(char)(&gen_code);
	qc_destroy_ast(root);
	qc_destroy_array(QC_Token)(&tokens);
	QC_FREE(src_buf);
	if (file)
		fclose(file);
	return ret;

error:
	ret = 1;
	goto cleanup;
}
