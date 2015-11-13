#include "tokenize.h"

DEFINE_ARRAY(Token)

INTERNAL bool whitespace(char ch)
{ return ch == ' ' || ch == '\t' || ch == '\n'; }

INTERNAL bool linebreak(char ch)
{ return ch == '\n'; }

INTERNAL Token_Type single_char_tokentype(char ch)
{
	switch (ch) {
		case '=': return Token_assign;
		case ';': return Token_semi;
		case ',': return Token_comma;
		case '(': return Token_open_paren;
		case ')': return Token_close_paren;
		case '{': return Token_open_brace;
		case '}': return Token_close_brace;
		case '[': return Token_open_square;
		case ']': return Token_close_square;
		case '<': return Token_less;
		case '>': return Token_greater;
		case '+': return Token_add;
		case '-': return Token_sub;
		case '*': return Token_mul;
		case '/': return Token_div;
		case '\\': return Token_rdiv;
		case '%': return Token_mod;
		case '.': return Token_dot;
		case '&': return Token_amp;
		case '^': return Token_hat;
		case '?': return Token_question;
		case '~': return Token_tilde;
		case '\'': return Token_squote;
		default: return Token_unknown;
	}
}

INTERNAL Token_Type double_char_tokentype(char ch1, char ch2)
{
	if (ch1 == '-' && ch2 == '>')
		return Token_right_arrow;
	if (ch1 == '=' && ch2 == '=')
		return Token_equals;
	if (ch1 == '!' && ch2 == '=')
		return Token_nequals;
	if (ch1 == '/' && ch2 == '/')
		return Token_line_comment;
	if (ch1 == '/' && ch2 == '*')
		return Token_block_comment;
	if (ch1 == '<' && ch2 == '=')
		return Token_leq;
	if (ch1 == '>' && ch2 == '=')
		return Token_geq;
	if (ch2 == '=') {
		switch (ch1) {
			case '+': return Token_add_assign;
			case '-': return Token_sub_assign;
			case '*': return Token_mul_assign;
			case '/': return Token_div_assign;
			default:;
		}
	}

	return Token_unknown;
}

INTERNAL Token_Type kw_tokentype(const char *buf, int size)
{
	if (!strncmp(buf, "struct", size))
		return Token_kw_struct;
	if (!strncmp(buf, "return", size))
		return Token_kw_return;
	if (!strncmp(buf, "goto", size))
		return Token_kw_goto;
	if (!strncmp(buf, "break", size))
		return Token_kw_break;
	if (!strncmp(buf, "continue", size))
		return Token_kw_continue;
	if (!strncmp(buf, "else", size))
		return Token_kw_else;
	if (!strncmp(buf, "NULL", size))
		return Token_kw_null;
	if (!strncmp(buf, "for", size))
		return Token_kw_for;
	if (!strncmp(buf, "if", size))
		return Token_kw_if;
	if (!strncmp(buf, "true", size))
		return Token_kw_true;
	if (!strncmp(buf, "false", size))
		return Token_kw_false;
	if (!strncmp(buf, "true", size))
		return Token_kw_true;
	if (!strncmp(buf, "false", size))
		return Token_kw_false;
	return Token_unknown;
}

typedef enum {
	Tok_State_none,
	Tok_State_maybe_single_char,
	Tok_State_number,
	Tok_State_number_after_dot,
	Tok_State_name,
	Tok_State_str,
	Tok_State_line_comment,
	Tok_State_block_comment
} Tok_State;

typedef struct Tokenize_Ctx {
	Tok_State state;
	int block_comment_depth;
	const char *end;
	int cur_line;
	bool last_line_was_empty;
	int tokens_on_line;
	int comments_on_line;
	Array(Token) tokens;
} Tokenize_Ctx;

INTERNAL void commit_token(Tokenize_Ctx *t, const char *b, const char *e, Token_Type type)
{
	if (e > b) {
		Token tok = {0};
		bool last_on_line = e + 1 < t->end && linebreak(*e);
		if (type == Token_name) {
			Token_Type kw = kw_tokentype(b, e - b);
			if (kw != Token_unknown)
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
		t->state = Tok_State_none;
		++t->tokens_on_line;
	}
}

INTERNAL void on_linebreak(Tokenize_Ctx *t)
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
	Tokenize_Ctx t = {0};
	t.end = src + src_size;
	t.cur_line = 1;
	t.tokens = create_array(Token)(src_size/4); /* Estimate token count */

	while (cur < t.end && tok_begin < t.end) {
		switch (t.state) {
			case Tok_State_none:
				if (single_char_tokentype(*cur) != Token_unknown) {
					t.state = Tok_State_maybe_single_char;
				} else if (*cur >= '0' && *cur <= '9') {
					t.state = Tok_State_number;
				} else if (	(*cur >= 'a' && *cur <= 'z') ||
							(*cur >= 'A' && *cur <= 'Z') ||
							(*cur == '_')) {
					t.state = Tok_State_name;
				} else if (*cur == '\"') {
					t.state = Tok_State_str;
				} else if (linebreak(*cur)) {
					on_linebreak(&t);
				}
				tok_begin = cur;
			break;
			case Tok_State_maybe_single_char: {
				Token_Type type = double_char_tokentype(*tok_begin, *cur);
				if (type == Token_unknown) {
					commit_token(&t, tok_begin, cur, single_char_tokentype(*tok_begin));
					--cur;
				} else {
					if (type == Token_line_comment) {
						t.state = Tok_State_line_comment;
						tok_begin += 2;
					} else if (type == Token_block_comment) {
						t.state = Tok_State_block_comment;
						t.block_comment_depth = 1;
						tok_begin += 2;
					} else {
						commit_token(&t, tok_begin, cur + 1, type);
					}
				}
			}
			break;
			case Tok_State_number_after_dot:
			case Tok_State_number:
				if (	whitespace(*cur) ||
						single_char_tokentype(*cur) != Token_unknown) {
					if (t.state == Tok_State_number_after_dot) {
						/* `123.` <- last dot is detected and removed, */
						/* because `.>` is a token */
						commit_token(&t, tok_begin, cur - 1, Token_number);
						cur -= 2;
						break;
					} else if (*cur != '.') {
						commit_token(&t, tok_begin, cur, Token_number);
						--cur;
						break;
					}
				}

				if (*cur == '.')
					t.state = Tok_State_number_after_dot;
				else
					t.state = Tok_State_number;
			break;
			case Tok_State_name:
				if (	whitespace(*cur) ||
						single_char_tokentype(*cur) != Token_unknown) {
					commit_token(&t, tok_begin, cur, Token_name);
					--cur;
				}
			break;
			case Tok_State_str:
				if (*cur == '\"')
					commit_token(&t, tok_begin + 1, cur, Token_string);
			break;
			case Tok_State_line_comment:
				if (linebreak(*cur)) {
					commit_token(&t, tok_begin, cur, Token_line_comment);
					on_linebreak(&t);
				}
			case Tok_State_block_comment: {
				char a = *(cur - 1);
				char b = *(cur);
				if (double_char_tokentype(a, b) == Token_block_comment) {
					++t.block_comment_depth;
				} else if (a == '*' && b == '/') {
					--t.block_comment_depth;
					if (t.block_comment_depth <= 0)
						commit_token(&t, tok_begin, cur - 1, Token_block_comment);
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

const char* tokentype_str(Token_Type type)
{
	switch (type) {
		case Token_eof: return "eof";
		case Token_name: return "name";
		case Token_number: return "number";
		case Token_assign: return "assign";
		case Token_semi: return "semi";
		case Token_comma: return "comma";
		case Token_open_paren: return "open_paren";
		case Token_close_paren: return "close_paren";
		case Token_open_brace: return "open_brace";
		case Token_close_brace: return "close_brace";
		case Token_open_square: return "open_square";
		case Token_close_square: return "close_square";
		case Token_right_arrow: return "right_arrow";
		case Token_equals: return "equals";
		case Token_nequals: return "nequals";
		case Token_less: return "less";
		case Token_greater: return "greater";
		case Token_leq: return "leq";
		case Token_geq: return "geq";
		case Token_add_assign: return "add_assign";
		case Token_sub_assign: return "sub_assign";
		case Token_mul_assign: return "mul_assign";
		case Token_div_assign: return "div_assign";
		case Token_add: return "add";
		case Token_sub: return "sub";
		case Token_mul: return "mul";
		case Token_div: return "div";
		case Token_rdiv: return "rdiv";
		case Token_mod: return "mod";
		case Token_dot: return "dot";
		case Token_amp: return "amp";
		case Token_hat: return "hat";
		case Token_question: return "question";
		case Token_tilde: return "tilde";
		case Token_squote: return "squote";
		case Token_line_comment: return "line_comment";
		case Token_block_comment: return "block_comment";
		case Token_kw_struct: return "kw_struct";
		case Token_kw_return: return "kw_return";
		case Token_kw_goto: return "kw_goto";
		case Token_kw_break: return "kw_break";
		case Token_kw_continue: return "kw_continue";
		case Token_kw_else: return "kw_else";
		case Token_kw_null: return "kw_null";
		case Token_kw_for: return "kw_for";
		case Token_kw_if: return "kw_if";
		case Token_kw_true: return "kw_true";
		case Token_kw_false: return "kw_false";
		case Token_unknown:
		default: return "unknown";
	}
}

const char* tokentype_codestr(Token_Type type)
{
	switch (type) {
		case Token_eof: return "";
		case Token_name: return "";
		case Token_number: return "";
		case Token_assign: return "=";
		case Token_semi: return ";";
		case Token_comma: return ",";
		case Token_open_paren: return "(";
		case Token_close_paren: return ")";
		case Token_open_brace: return "{";
		case Token_close_brace: return "}";
		case Token_open_square: return "[";
		case Token_close_square: return "]";
		case Token_right_arrow: return "->";
		case Token_equals: return "==";
		case Token_nequals: return "!=";
		case Token_less: return "<";
		case Token_greater: return ">";
		case Token_leq: return "<=";
		case Token_geq: return ">=";
		case Token_add_assign: return "+=";
		case Token_sub_assign: return "-=";
		case Token_mul_assign: return "*=";
		case Token_div_assign: return "/=";
		case Token_add: return "+";
		case Token_sub: return "-";
		case Token_mul: return "*";
		case Token_div: return "/";
		case Token_rdiv: return "\\";
		case Token_mod: return "%";
		case Token_dot: return ".";
		case Token_amp: return "&";
		case Token_hat: return "^";
		case Token_question: return "?";
		case Token_tilde: return "~";
		case Token_squote: return "'";
		case Token_line_comment: return "// ...";
		case Token_block_comment: return "/* ... */";
		case Token_kw_struct: return "struct";
		case Token_kw_return: return "return";
		case Token_kw_goto: return "goto";
		case Token_kw_break: return "break";
		case Token_kw_continue: return "continue";
		case Token_kw_else: return "else";
		case Token_kw_null: return "NULL";
		case Token_kw_for: return "for";
		case Token_kw_if: return "if";
		case Token_kw_true: return "true";
		case Token_kw_false: return "false";
		case Token_unknown:
		default: return "???";
	}
}

bool tok_text_equals(Token *tok, const char *str)
{
	int i;
	for (i = 0; str[i] && i < tok->text_len; ++i) {
		if (str[i] != tok->text_buf[i])
			return false;
	}
	return true;
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
