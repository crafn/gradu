#ifndef TOKENIZE_H
#define TOKENIZE_H

#include "core.h"

typedef enum {
	Token_eof,
	Token_name, /* single_word_like_this */
	Token_number, /* 2538 */
	Token_string, /* "something" */
	Token_assign, /* = */
	Token_semi, /* ; */
	Token_comma, /* , */
	Token_open_paren, /* ( */
	Token_close_paren, /* ) */
	Token_open_brace, /* { */
	Token_close_brace, /* } */
	Token_open_square, /* [ */
	Token_close_square, /* ] */
	Token_rdiv, /* `\` */
	Token_right_arrow, /* -> */
	Token_equals, /* == */
	Token_nequals, /* != */
	Token_less, /* < */
	Token_greater, /* > */
	Token_leq, /* <= */
	Token_geq, /* >= */
	Token_add_assign, /* += */
	Token_sub_assign, /* -= */
	Token_mul_assign, /* *= */
	Token_div_assign, /* /= */
	Token_add, /* + */
	Token_sub, /* - */
	Token_mul, /* * */
	Token_div, /* / */
	Token_mod, /* % */
	Token_dot, /* . */
	Token_amp, /* & */
	Token_hat, /* ^ */
	Token_tilde, /* ~ */
	Token_question, /* ? */
	Token_squote, /* ' */
	Token_line_comment, /* // this is comment */
	Token_block_comment, /* this is block comment */
	Token_kw_struct, /* struct */
	Token_kw_return, /* return */
	Token_kw_goto, /* goto */
	Token_kw_break, /* break */
	Token_kw_continue, /* continue */
	Token_kw_else, /* else */
	Token_kw_null, /* NULL */
	Token_kw_for, /* for */
	Token_kw_if, /* if */
	Token_kw_true, /* true */
	Token_kw_false, /* false */
	Token_unknown
} Token_Type;

const char* tokentype_str(Token_Type type);
const char* tokentype_codestr(Token_Type type);

typedef struct Token {
	Token_Type type;
	const char *text_buf; /* Not terminated! */
	int text_len;
	int line;

	bool empty_line_before;
	bool last_on_line;

	/* Used only for comments */
	int comment_bound_to; /* -1 == prev token, 1 == next_token */
} Token;

/* String args */
#define TOK_ARGS(tok) tok->text_len, tok->text_buf

static bool is_comment_tok(Token_Type type) { return type == Token_line_comment || type == Token_block_comment; }

DECLARE_ARRAY(Token)

/* Tokens will be pointing to the 'src' string */
Array(Token) tokenize(const char* src, int src_size);

void print_tokens(Token *tokens, int token_count);


#endif
