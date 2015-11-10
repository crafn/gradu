#ifndef TOKENIZE_H
#define TOKENIZE_H

#include "core.h"

typedef enum {
	TokenType_eof,
	TokenType_name, /* single_word_like_this */
	TokenType_number, /* 2538 */
	TokenType_string, /* "something" */
	TokenType_assign, /* = */
	TokenType_semi, /* ; */
	TokenType_comma, /* , */
	TokenType_open_paren, /* ( */
	TokenType_close_paren, /* ) */
	TokenType_open_brace, /* { */
	TokenType_close_brace, /* } */
	TokenType_open_square, /* [ */
	TokenType_close_square, /* ] */
	TokenType_rdiv, /* `\` */
	TokenType_right_arrow, /* -> */
	TokenType_equals, /* == */
	TokenType_nequals, /* != */
	TokenType_less, /* < */
	TokenType_greater, /* > */
	TokenType_leq, /* <= */
	TokenType_geq, /* >= */
	TokenType_add_assign, /* += */
	TokenType_sub_assign, /* -= */
	TokenType_mul_assign, /* *= */
	TokenType_div_assign, /* /= */
	TokenType_add, /* + */
	TokenType_sub, /* - */
	TokenType_mul, /* * */
	TokenType_div, /* / */
	TokenType_mod, /* % */
	TokenType_dot, /* . */
	TokenType_amp, /* & */
	TokenType_hat, /* ^ */
	TokenType_tilde, /* ~ */
	TokenType_question, /* ? */
	TokenType_squote, /* ' */
	TokenType_line_comment, /* // this is comment */
	TokenType_block_comment, /* this is block comment */
	TokenType_kw_struct, /* struct */
	TokenType_kw_return, /* return */
	TokenType_kw_goto, /* goto */
	TokenType_kw_break, /* break */
	TokenType_kw_continue, /* continue */
	TokenType_kw_else, /* else */
	TokenType_kw_null, /* NULL */
	TokenType_kw_for, /* for */
	TokenType_kw_if, /* if */
	TokenType_kw_true, /* true */
	TokenType_kw_false, /* false */
	TokenType_unknown
} TokenType;

const char* tokentype_str(TokenType type);
const char* tokentype_codestr(TokenType type);

typedef struct Token {
	TokenType type;
	const char *text_buf; /* Not terminated! */
	int text_len;
	int line;

	bool empty_line_before;
	bool last_on_line;
} Token;

/* String args */
#define TOK_ARGS(tok) tok->text_len, tok->text_buf

static bool is_comment_tok(Token *tok) { return tok->type == TokenType_line_comment || tok->type == TokenType_block_comment; }

DECLARE_ARRAY(Token)

/* Tokens will be pointing to the 'src' string */
Array(Token) tokenize(const char* src, int src_size);

void print_tokens(Token *tokens, int token_count);


#endif
