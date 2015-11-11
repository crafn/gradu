#include "tokenize.h"

DEFINE_ARRAY(Token)

INTERNAL bool whitespace(char ch)
{ return ch == ' ' || ch == '\t' || ch == '\n'; }

INTERNAL bool linebreak(char ch)
{ return ch == '\n'; }

INTERNAL TokenType single_char_tokentype(char ch)
{
	switch (ch) {
		case '=': return TokenType_assign;
		case ';': return TokenType_semi;
		case ',': return TokenType_comma;
		case '(': return TokenType_open_paren;
		case ')': return TokenType_close_paren;
		case '{': return TokenType_open_brace;
		case '}': return TokenType_close_brace;
		case '[': return TokenType_open_square;
		case ']': return TokenType_close_square;
		case '<': return TokenType_less;
		case '>': return TokenType_greater;
		case '+': return TokenType_add;
		case '-': return TokenType_sub;
		case '*': return TokenType_mul;
		case '/': return TokenType_div;
		case '\\': return TokenType_rdiv;
		case '%': return TokenType_mod;
		case '.': return TokenType_dot;
		case '&': return TokenType_amp;
		case '^': return TokenType_hat;
		case '?': return TokenType_question;
		case '~': return TokenType_tilde;
		case '\'': return TokenType_squote;
		default: return TokenType_unknown;
	}
}

INTERNAL TokenType double_char_tokentype(char ch1, char ch2)
{
	if (ch1 == '-' && ch2 == '>')
		return TokenType_right_arrow;
	if (ch1 == '=' && ch2 == '=')
		return TokenType_equals;
	if (ch1 == '!' && ch2 == '=')
		return TokenType_nequals;
	if (ch1 == '/' && ch2 == '/')
		return TokenType_line_comment;
	if (ch1 == '/' && ch2 == '*')
		return TokenType_block_comment;
	if (ch1 == '<' && ch2 == '=')
		return TokenType_leq;
	if (ch1 == '>' && ch2 == '=')
		return TokenType_geq;
	if (ch2 == '=') {
		switch (ch1) {
			case '+': return TokenType_add_assign;
			case '-': return TokenType_sub_assign;
			case '*': return TokenType_mul_assign;
			case '/': return TokenType_div_assign;
			default:;
		}
	}

	return TokenType_unknown;
}

INTERNAL TokenType kw_tokentype(const char *buf, int size)
{
	if (!strncmp(buf, "struct", size))
		return TokenType_kw_struct;
	if (!strncmp(buf, "return", size))
		return TokenType_kw_return;
	if (!strncmp(buf, "goto", size))
		return TokenType_kw_goto;
	if (!strncmp(buf, "break", size))
		return TokenType_kw_break;
	if (!strncmp(buf, "continue", size))
		return TokenType_kw_continue;
	if (!strncmp(buf, "else", size))
		return TokenType_kw_else;
	if (!strncmp(buf, "NULL", size))
		return TokenType_kw_null;
	if (!strncmp(buf, "for", size))
		return TokenType_kw_for;
	if (!strncmp(buf, "if", size))
		return TokenType_kw_if;
	if (!strncmp(buf, "true", size))
		return TokenType_kw_true;
	if (!strncmp(buf, "false", size))
		return TokenType_kw_false;
	if (!strncmp(buf, "true", size))
		return TokenType_kw_true;
	if (!strncmp(buf, "false", size))
		return TokenType_kw_false;
	return TokenType_unknown;
}

typedef enum {
	TokState_none,
	TokState_maybe_single_char,
	TokState_number,
	TokState_number_after_dot,
	TokState_name,
	TokState_str,
	TokState_line_comment,
	TokState_block_comment
} TokState;

typedef struct TokenizeCtx {
	TokState state;
	int block_comment_depth;
	const char *end;
	int cur_line;
	bool last_line_was_empty;
	int tokens_on_line;
	int comments_on_line;
	Array(Token) tokens;
} TokenizeCtx;

INTERNAL void commit_token(TokenizeCtx *t, const char *b, const char *e, TokenType type)
{
	if (e > b) {
		Token tok = {0};
		bool last_on_line = e + 1 < t->end && linebreak(*e);
		if (type == TokenType_name) {
			TokenType kw = kw_tokentype(b, e - b);
			if (kw != TokenType_unknown)
				type = kw;
		}
		tok.type = type;
		tok.text_buf = b;
		tok.text_len = e - b;
		tok.line = t->cur_line;
		tok.empty_line_before = (t->tokens_on_line == 0 && t->last_line_was_empty);
		tok.last_on_line = last_on_line;

		if (is_comment_tok(type)) {
			if (t->tokens_on_line  == t->comments_on_line)
				tok.comment_bound_to = 1; /* If line is only comments, bound to next token */
			else
				tok.comment_bound_to = -1; /* Else bound to token left to comment */
			++t->comments_on_line;
		}

		push_array(Token)(&t->tokens, tok);
		t->state = TokState_none;
		++t->tokens_on_line;
	}
}

INTERNAL void on_linebreak(TokenizeCtx *t)
{
	++t->cur_line;
	t->last_line_was_empty = (t->tokens_on_line == 0);
	t->tokens_on_line = 0;
	t->comments_on_line = 0;
}

Array(Token) tokenize(const char* src, int src_size)
{
	const char *cur = src;
	const char *tok_begin = src;
	TokenizeCtx t = {0};
	t.end = src + src_size;
	t.cur_line = 1;
	t.tokens = create_array(Token)(src_size/4); /* Estimate token count */

	while (cur < t.end && tok_begin < t.end) {
		switch (t.state) {
			case TokState_none:
				if (single_char_tokentype(*cur) != TokenType_unknown) {
					t.state = TokState_maybe_single_char;
				} else if (*cur >= '0' && *cur <= '9') {
					t.state = TokState_number;
				} else if (	(*cur >= 'a' && *cur <= 'z') ||
							(*cur >= 'A' && *cur <= 'Z') ||
							(*cur == '_')) {
					t.state = TokState_name;
				} else if (*cur == '\"') {
					t.state = TokState_str;
				} else if (linebreak(*cur)) {
					on_linebreak(&t);
				}
				tok_begin = cur;
			break;
			case TokState_maybe_single_char: {
				TokenType type = double_char_tokentype(*tok_begin, *cur);
				if (type == TokenType_unknown) {
					commit_token(&t, tok_begin, cur, single_char_tokentype(*tok_begin));
					--cur;
				} else {
					if (type == TokenType_line_comment) {
						t.state = TokState_line_comment;
						tok_begin += 2;
					} else if (type == TokenType_block_comment) {
						t.state = TokState_block_comment;
						t.block_comment_depth = 1;
						tok_begin += 2;
					} else {
						commit_token(&t, tok_begin, cur + 1, type);
					}
				}
			}
			break;
			case TokState_number_after_dot:
			case TokState_number:
				if (	whitespace(*cur) ||
						single_char_tokentype(*cur) != TokenType_unknown) {
					if (t.state == TokState_number_after_dot) {
						/* `123.` <- last dot is detected and removed, */
						/* because `.>` is a token */
						commit_token(&t, tok_begin, cur - 1, TokenType_number);
						cur -= 2;
						break;
					} else if (*cur != '.') {
						commit_token(&t, tok_begin, cur, TokenType_number);
						--cur;
						break;
					}
				}

				if (*cur == '.')
					t.state = TokState_number_after_dot;
				else
					t.state = TokState_number;
			break;
			case TokState_name:
				if (	whitespace(*cur) ||
						single_char_tokentype(*cur) != TokenType_unknown) {
					commit_token(&t, tok_begin, cur, TokenType_name);
					--cur;
				}
			break;
			case TokState_str:
				if (*cur == '\"')
					commit_token(&t, tok_begin + 1, cur, TokenType_string);
			break;
			case TokState_line_comment:
				if (linebreak(*cur)) {
					commit_token(&t, tok_begin, cur, TokenType_line_comment);
					on_linebreak(&t);
				}
			case TokState_block_comment: {
				char a = *(cur - 1);
				char b = *(cur);
				if (double_char_tokentype(a, b) == TokenType_block_comment) {
					++t.block_comment_depth;
				} else if (a == '*' && b == '/') {
					--t.block_comment_depth;
					if (t.block_comment_depth <= 0)
						commit_token(&t, tok_begin, cur - 1, TokenType_block_comment);
				}
			} break;
			default:;
		}
		++cur;
	}

	{ /* Append eof */
		Token eof = {0};
		eof.text_buf = "eof";
		eof.text_len = strlen(eof.text_buf);
		eof.line = t.cur_line;
		eof.last_on_line = true;
		push_array(Token)(&t.tokens, eof);
	}
	return t.tokens;
}

const char* tokentype_str(TokenType type)
{
	switch (type) {
		case TokenType_eof: return "eof";
		case TokenType_name: return "name";
		case TokenType_number: return "number";
		case TokenType_assign: return "assign";
		case TokenType_semi: return "semi";
		case TokenType_comma: return "comma";
		case TokenType_open_paren: return "open_paren";
		case TokenType_close_paren: return "close_paren";
		case TokenType_open_brace: return "open_brace";
		case TokenType_close_brace: return "close_brace";
		case TokenType_open_square: return "open_square";
		case TokenType_close_square: return "close_square";
		case TokenType_right_arrow: return "right_arrow";
		case TokenType_equals: return "equals";
		case TokenType_nequals: return "nequals";
		case TokenType_less: return "less";
		case TokenType_greater: return "greater";
		case TokenType_leq: return "leq";
		case TokenType_geq: return "geq";
		case TokenType_add_assign: return "add_assign";
		case TokenType_sub_assign: return "sub_assign";
		case TokenType_mul_assign: return "mul_assign";
		case TokenType_div_assign: return "div_assign";
		case TokenType_add: return "add";
		case TokenType_sub: return "sub";
		case TokenType_mul: return "mul";
		case TokenType_div: return "div";
		case TokenType_rdiv: return "rdiv";
		case TokenType_mod: return "mod";
		case TokenType_dot: return "dot";
		case TokenType_amp: return "amp";
		case TokenType_hat: return "hat";
		case TokenType_question: return "question";
		case TokenType_tilde: return "tilde";
		case TokenType_squote: return "squote";
		case TokenType_line_comment: return "line_comment";
		case TokenType_block_comment: return "block_comment";
		case TokenType_kw_struct: return "kw_struct";
		case TokenType_kw_return: return "kw_return";
		case TokenType_kw_goto: return "kw_goto";
		case TokenType_kw_break: return "kw_break";
		case TokenType_kw_continue: return "kw_continue";
		case TokenType_kw_else: return "kw_else";
		case TokenType_kw_null: return "kw_null";
		case TokenType_kw_for: return "kw_for";
		case TokenType_kw_if: return "kw_if";
		case TokenType_kw_true: return "kw_true";
		case TokenType_kw_false: return "kw_false";
		case TokenType_unknown:
		default: return "unknown";
	}
}

const char* tokentype_codestr(TokenType type)
{
	switch (type) {
		case TokenType_eof: return "";
		case TokenType_name: return "";
		case TokenType_number: return "";
		case TokenType_assign: return "=";
		case TokenType_semi: return ";";
		case TokenType_comma: return ",";
		case TokenType_open_paren: return "(";
		case TokenType_close_paren: return ")";
		case TokenType_open_brace: return "{";
		case TokenType_close_brace: return "}";
		case TokenType_open_square: return "[";
		case TokenType_close_square: return "]";
		case TokenType_right_arrow: return "->";
		case TokenType_equals: return "==";
		case TokenType_nequals: return "!=";
		case TokenType_less: return "<";
		case TokenType_greater: return ">";
		case TokenType_leq: return "<=";
		case TokenType_geq: return ">=";
		case TokenType_add_assign: return "+=";
		case TokenType_sub_assign: return "-=";
		case TokenType_mul_assign: return "*=";
		case TokenType_div_assign: return "/=";
		case TokenType_add: return "+";
		case TokenType_sub: return "-";
		case TokenType_mul: return "*";
		case TokenType_div: return "/";
		case TokenType_rdiv: return "\\";
		case TokenType_mod: return "%";
		case TokenType_dot: return ".";
		case TokenType_amp: return "&";
		case TokenType_hat: return "^";
		case TokenType_question: return "?";
		case TokenType_tilde: return "~";
		case TokenType_squote: return "'";
		case TokenType_line_comment: return "// ...";
		case TokenType_block_comment: return "/* ... */";
		case TokenType_kw_struct: return "struct";
		case TokenType_kw_return: return "return";
		case TokenType_kw_goto: return "goto";
		case TokenType_kw_break: return "break";
		case TokenType_kw_continue: return "continue";
		case TokenType_kw_else: return "else";
		case TokenType_kw_null: return "NULL";
		case TokenType_kw_for: return "for";
		case TokenType_kw_if: return "if";
		case TokenType_kw_true: return "true";
		case TokenType_kw_false: return "false";
		case TokenType_unknown:
		default: return "???";
	}
}

void print_tokens(Token *tokens, int token_count)
{
	int i;
	for (i = 0; i < token_count; ++i) {
		Token tok = tokens[i];
		int text_len = MIN(tok.text_len, 20);
		printf("%14s: %20.*s %8i last_on_line: %i empty_line_before: %i\n",
				tokentype_str(tok.type), text_len, tok.text_buf, tok.line, tok.last_on_line, tok.empty_line_before);
	}
}
