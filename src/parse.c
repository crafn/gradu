#include "parse.h"

int str_to_int(Buf_Str text)
{
	const char *c = text.buf;
	const char *end = c + text.len;
	int value = 0;
	int sign = 1;
	if (*c == '+' || *c == '-') {
		if (*c == '-')
			sign = -1;
		c++;
	}
	while (c < end) {
		value *= 10;
		ASSERT(*c >= '0' && *c <= '9');
		value += (int)(*c - '0');
		c++;
	}
	return value*sign;
}

double str_to_float(Buf_Str text)
{
	const char *c = text.buf;
	const char *end = c + text.len;
	double value = 0.0;
	int sign = 1;
	int decimal_div = 1;

	if (*c == '+' || *c == '-') {
		if (*c == '-')
			sign = -1;
		c++;
	}

	while (c < end && *c != '.') {
		value *= 10;
		ASSERT(*c >= '0' && *c <= '9');
		value += (double)(*c - '0');
		++c;
	}

	++c;

	while (c < end) {
		value += (double)(*c - '0')/decimal_div;
		decimal_div *= 10;
		++c;
	}

	return value*sign;
}

#define UOP_PRECEDENCE 100000
/* Not all of the accepted tokens are actually binary operators. They're here for similar precedence handling. */
int biop_prec(Token_Type type)
{
	/* Order should match with C operator precedence */
	int prec = 1;
	switch (type) {

	case Token_open_paren: ++prec; /* Function call */

	case Token_dot:
	case Token_right_arrow: ++prec;

	case Token_mul: 
	case Token_div: ++prec;
	case Token_mod: ++prec;
	case Token_add: 
	case Token_sub: ++prec;

	case Token_leq:
	case Token_geq:
	case Token_less:
	case Token_greater: ++prec;

	case Token_equals: ++prec;

	case Token_and: ++prec;
	case Token_or: ++prec;

	case Token_assign: ++prec;
	case Token_add_assign:
	case Token_sub_assign:
	case Token_mul_assign:
	case Token_div_assign: ++prec;

	break;
	default: return -1;
	}
	return prec;
}

/* -1 left, 1 right */
int biop_assoc(Token_Type type)
{
	switch (type) {
		case Token_assign: return 1; /* a = b = c  <=>  (a = (b = c)) */
		default: return -1;
	}
}

bool is_binary_op(Token_Type type)
{
	switch (type) {

	case Token_mul: 
	case Token_div:
	case Token_mod:
	case Token_add: 
	case Token_sub:

	case Token_leq:
	case Token_geq:
	case Token_less:
	case Token_greater:

	case Token_equals:

	case Token_and:
	case Token_or:

	case Token_assign:
	case Token_add_assign:
	case Token_sub_assign:
	case Token_mul_assign:
	case Token_div_assign:
		return true;
	default: return false;
	}
}

bool is_unary_op(Token_Type type)
{
	switch (type) {
		case Token_add:
		case Token_sub:
		case Token_kw_sizeof:
		case Token_incr:
		case Token_decr:
		case Token_addrof:
		case Token_mul: /* deref */
			return true;
		default: return false;
	}
}


/* Mirrors call stack in parsing */
/* Used in searching, setting parent nodes, and backtracking */
typedef struct Parse_Stack_Frame {
	Token *begin_tok;
	AST_Node *node;
} Parse_Stack_Frame;

DECLARE_ARRAY(Parse_Stack_Frame)
DEFINE_ARRAY(Parse_Stack_Frame)

typedef AST_Type_Decl* AST_Type_Decl_Ptr;
DECLARE_ARRAY(AST_Type_Decl_Ptr)
DEFINE_ARRAY(AST_Type_Decl_Ptr)

typedef struct Parse_Ctx {
	AST_Scope *root;
	Token *first_tok; /* @todo Consider having some Token_sof, corresponding to Token_eof*/
	Token *tok; /* Access with cur_tok */
	int expr_depth;

	/* Builtin type and funcs decls are generated while parsing */

	Array(char) error_msg;
	Token *error_tok;
	Array(Parse_Stack_Frame) parse_stack;

	AST_Parent_Map parent_map; /* Built incrementally during parsing */
} Parse_Ctx;


/* Token manipulation */

INTERNAL Token *cur_tok(Parse_Ctx *ctx)
{ return ctx->tok; }

INTERNAL void advance_tok(Parse_Ctx *ctx)
{
	ASSERT(ctx->tok->type != Token_eof);
	do {
		++ctx->tok;
	} while (is_comment_tok(ctx->tok->type));
}

INTERNAL bool accept_tok(Parse_Ctx *ctx, Token_Type type)
{
	if (ctx->tok->type == type) {
		advance_tok(ctx);
		return true;
	}
	return false;
}


/* Backtracking / stack traversing */

INTERNAL void begin_node_parsing(Parse_Ctx *ctx, AST_Node *node)
{
	Parse_Stack_Frame frame = {0};
	ASSERT(node);

	if (ctx->parse_stack.size > 0) {
		AST_Node *parent = ctx->parse_stack.data[ctx->parse_stack.size - 1].node;
		ASSERT(parent && parent != node);
		set_parent_node(&ctx->parent_map, node, parent);
	}

	frame.begin_tok = cur_tok(ctx);
	frame.node = node;
	push_array(Parse_Stack_Frame)(&ctx->parse_stack, frame);
}

INTERNAL bool can_have_post_comments(AST_Node *node)
{
	/* if or loop without explicit scope can't have post-comments -- only the statement can */
	if (node->type == AST_cond) {
		CASTED_NODE(AST_Cond, cond, node);
		return !cond->implicit_scope;
	} else if (node->type == AST_loop) {
		CASTED_NODE(AST_Loop, loop, node);
		return !loop->implicit_scope;
	}
	return true;
}

INTERNAL void end_node_parsing(Parse_Ctx *ctx)
{
	Parse_Stack_Frame frame = pop_array(Parse_Stack_Frame)(&ctx->parse_stack);
	ASSERT(frame.node);

	/* frame.node is used in end_node_parsing because node might not yet be created at the begin_node_parsing */
	/* That ^ is false now! Can be moved to begin. */
	frame.node->begin_tok = frame.begin_tok;

	/*	Gather comments around node if this is the first time calling end_node_parsing with this node.
		It's possible to have multiple begin...end_node_parsing with the same node because of nesting,
		like when parsing statement: 'foo;', which yields parse_expr(parse_ident()). */
	/* That ^ is false now! Not possible to have multiple begin..end with the same node anymore. */
	if (frame.node->pre_comments.size == 0) {
		Token *tok = frame.begin_tok - 1;
		/* Rewind first */
		while (tok >= ctx->first_tok && is_comment_tok(tok->type))
			--tok;
		++tok;
		while (is_comment_tok(tok->type) && tok->comment_bound_to == 1) {
			push_array(Token_Ptr)(&frame.node->pre_comments, tok);
			++tok;
		}
	}
	if (can_have_post_comments(frame.node) && frame.node->post_comments.size == 0) {
		Token *tok = cur_tok(ctx);
		/* Cursor has been advanced past following comments, rewind */
		--tok;
		while (tok >= ctx->first_tok && is_comment_tok(tok->type))
			--tok;
		++tok;
		while (is_comment_tok(tok->type) && tok->comment_bound_to == -1) {
			push_array(Token_Ptr)(&frame.node->post_comments, tok);
			++tok;
		}
	}
}

INTERNAL void cancel_node_parsing(Parse_Ctx *ctx)
{
	Parse_Stack_Frame frame = pop_array(Parse_Stack_Frame)(&ctx->parse_stack);
	ASSERT(frame.node);

	/* Backtrack */
	ctx->tok = frame.begin_tok;
	set_parent_node(&ctx->parent_map, frame.node, NULL);

	destroy_node(frame.node);
}

INTERNAL void report_error(Parse_Ctx *ctx, const char *fmt, ...)
{
	Array(char) msg;
	va_list args;

	if (cur_tok(ctx) <= ctx->error_tok)
		return; /* Don't overwrite error generated from less succesfull parsing (heuristic) */

	va_start(args, fmt);
	msg = create_array(char)(0);
	safe_vsprintf(&msg, fmt, args);
	va_end(args);

	destroy_array(char)(&ctx->error_msg);
	ctx->error_msg = msg;
	ctx->error_tok = cur_tok(ctx);
}

INTERNAL void report_error_expected(Parse_Ctx *ctx, const char *expected, Token *got)
{
	report_error(ctx, "Expected %s, got '%.*s'", expected, BUF_STR_ARGS(got->text));
}

/* Parsing */

/* @todo AST_Node -> specific node type. <-- not so sure about that.. */
INTERNAL bool parse_ident(Parse_Ctx *ctx, AST_Ident **ret, AST_Node *decl);
INTERNAL bool parse_type_decl(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_typedef(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_var_decl(Parse_Ctx *ctx, AST_Node **ret, bool is_param_decl);
INTERNAL bool parse_type(Parse_Ctx *ctx, AST_Type **ret_type);
INTERNAL bool parse_type_and_ident(Parse_Ctx *ctx, AST_Type **ret_type, AST_Ident **ret_ident, AST_Node *enclosing_decl);
INTERNAL bool parse_func_decl(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_scope(Parse_Ctx *ctx, AST_Scope **ret, bool already_created);
INTERNAL bool parse_literal(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_uop(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_arg_list(Parse_Ctx *ctx, Array(AST_Node_Ptr) *ret, Token_Type ending);
INTERNAL bool parse_expr_inside_parens(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_expr(Parse_Ctx *ctx, AST_Node **ret, int min_prec, AST_Type *type_hint, bool semi);
INTERNAL bool parse_control(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_cond(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_loop(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_parallel(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_element(Parse_Ctx *ctx, AST_Node **ret);

/* If decl is NULL, then declaration is searched. */
/* Parse example: foo */
INTERNAL bool parse_ident(Parse_Ctx *ctx, AST_Ident **ret, AST_Node *decl)
{
	AST_Ident *ident = create_ident_node();

	begin_node_parsing(ctx, AST_BASE(ident));

	/* Consume dot before designated initializer identifier */
	if (accept_tok(ctx, Token_dot))
		ident->designated = true;
	append_str(&ident->text, "%.*s", BUF_STR_ARGS(cur_tok(ctx)->text));

	if (cur_tok(ctx)->type != Token_name) {
		report_error(ctx, "'%.*s' is not an identifier", BUF_STR_ARGS(cur_tok(ctx)->text));
		goto mismatch;
	}

	/* Might be NULL -- should be resolved shortly after */
	ident->decl = decl;

	advance_tok(ctx);

	end_node_parsing(ctx);

	*ret = ident;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse examples: int test -- int func() { } -- struct foo { } */
INTERNAL bool parse_type_decl(Parse_Ctx *ctx, AST_Node **ret)
{
	AST_Type_Decl *decl = create_type_decl_node();
	begin_node_parsing(ctx, AST_BASE(decl));

	if (!accept_tok(ctx, Token_kw_struct))
		goto mismatch;

	if (!parse_ident(ctx, &decl->ident, AST_BASE(decl))) {
		report_error_expected(ctx, "type name", cur_tok(ctx));
		goto mismatch;
	}

	if (parse_scope(ctx, &decl->body, false))
		;
	else {
		report_error_expected(ctx, "'{'", cur_tok(ctx));
		goto mismatch;
	}

	end_node_parsing(ctx);

	*ret = AST_BASE(decl);
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

INTERNAL bool parse_typedef(Parse_Ctx *ctx, AST_Node **ret)
{
	AST_Typedef *def = create_typedef_node();
	begin_node_parsing(ctx, AST_BASE(def));

	if (!accept_tok(ctx, Token_kw_typedef)) {
		report_error_expected(ctx, "typedef", cur_tok(ctx));
		goto mismatch;
	}

	if (!parse_type_and_ident(ctx, &def->type, &def->ident, AST_BASE(def)))
		goto mismatch;

	if (!accept_tok(ctx, Token_semi)) {
		report_error_expected(ctx, ";", cur_tok(ctx));
		goto mismatch;
	}

	end_node_parsing(ctx);

	*ret = AST_BASE(def);
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

AST_Type_Decl *create_builtin_decl(Parse_Ctx *ctx, Builtin_Type bt)
{
	int i;
	/* Search for builtin declaration from existing decls */
	for (i = 0; i < ctx->parent_map.builtin_decls.size; ++i) {
		AST_Node *node = ctx->parent_map.builtin_decls.data[i];
		if (node->type == AST_type_decl) {
			CASTED_NODE(AST_Type_Decl, decl, node);
			Builtin_Type t = decl->builtin_type;
			if (builtin_type_equals(t, bt))
				return decl;
		}
	}

	{ /* Create new builtin decl if not found */
		AST_Type_Decl *tdecl = create_type_decl_node();
		/* Note that the declaration doesn't have ident -- it's up to backend to generate it */
		tdecl->is_builtin = true;
		tdecl->builtin_type = bt;
		push_array(AST_Node_Ptr)(&ctx->parent_map.builtin_decls, AST_BASE(tdecl));

		if (bt.is_field) {
			int type;
			/* Create field alloc funcs */
			for (type = 0; type < 2; ++type) { 
				const char *names[2] = { "alloc_field", "alloc_device_field" };
				AST_Func_Decl *fdecl = create_func_decl_node();
				fdecl->is_builtin = true;

				fdecl->return_type = create_type_node();
				fdecl->return_type->base_type_decl = tdecl;

				fdecl->ident = create_ident_with_text(AST_BASE(fdecl), names[type]);

				for (i = 0; i < bt.field_dim; ++i) {
					AST_Var_Decl *param = create_var_decl_node();
					param->type = create_type_node();
					param->type->base_type_decl = create_builtin_decl(ctx, int_builtin_type());
					param->ident = create_ident_with_text(AST_BASE(param), "size_%i", i);
					push_array(AST_Var_Decl_Ptr)(&fdecl->params, param);
				}

				push_array(AST_Node_Ptr)(&ctx->parent_map.builtin_decls, AST_BASE(fdecl));
			}

			/* Create field free funcs */
			for (type = 0; type < 2; ++type) { 
				const char *names[2] = { "free_field", "free_device_field" };
				AST_Func_Decl *fdecl = create_func_decl_node();
				fdecl->is_builtin = true;

				fdecl->return_type = create_type_node();
				fdecl->return_type->base_type_decl = create_builtin_decl(ctx, void_builtin_type());

				fdecl->ident = create_ident_with_text(AST_BASE(fdecl), names[type]);

				{
					AST_Var_Decl *param = create_var_decl_node();
					param->type = create_type_node();
					param->type->base_type_decl = tdecl;
					param->ident = create_ident_with_text(AST_BASE(param), "field");
					push_array(AST_Var_Decl_Ptr)(&fdecl->params, param);
				}

				push_array(AST_Node_Ptr)(&ctx->parent_map.builtin_decls, AST_BASE(fdecl));
			}

			{ /* Create field memcpy func */
				AST_Func_Decl *fdecl = create_func_decl_node();
				fdecl->is_builtin = true;

				fdecl->return_type = create_type_node();
				fdecl->return_type->base_type_decl = create_builtin_decl(ctx, void_builtin_type());

				fdecl->ident = create_ident_with_text(AST_BASE(fdecl), "memcpy_field");

				{
					AST_Var_Decl *dst = create_simple_var_decl(tdecl, "dst");
					AST_Var_Decl *src = create_simple_var_decl(tdecl, "src");
					push_array(AST_Var_Decl_Ptr)(&fdecl->params, dst);
					push_array(AST_Var_Decl_Ptr)(&fdecl->params, src);
				}

				push_array(AST_Node_Ptr)(&ctx->parent_map.builtin_decls, AST_BASE(fdecl));
			}

			{ /* Create field size query func */
				AST_Func_Decl *fdecl = create_func_decl_node();
				fdecl->is_builtin = true;

				fdecl->return_type = create_simple_type(create_builtin_decl(ctx, int_builtin_type()));
				fdecl->ident = create_ident_with_text(AST_BASE(fdecl), "size");

				{
					AST_Var_Decl *param1 = create_simple_var_decl(tdecl, "field");
					AST_Var_Decl *param2 =
						create_simple_var_decl(
								create_builtin_decl(ctx, int_builtin_type()),
								"index");
					push_array(AST_Var_Decl_Ptr)(&fdecl->params, param1);
					push_array(AST_Var_Decl_Ptr)(&fdecl->params, param2);
				}

				push_array(AST_Node_Ptr)(&ctx->parent_map.builtin_decls, AST_BASE(fdecl));
			}
		}

		return tdecl;
	}
}

INTERNAL bool parse_type(Parse_Ctx *ctx, AST_Type **ret_type)
{
	return parse_type_and_ident(ctx, ret_type, NULL, NULL);
}

/* Type and ident in same function because of cases like 'int (*foo)()' */
INTERNAL bool parse_type_and_ident(Parse_Ctx *ctx, AST_Type **ret_type, AST_Ident **ret_ident, AST_Node *enclosing_decl)
{
	AST_Type *type = create_type_node();
	begin_node_parsing(ctx, AST_BASE(type));

	/* @todo ptr-to-funcs, const (?), types with multiple identifiers... */

	{ /* Type */
		AST_Node *found_decl = NULL;
		bool is_builtin = false;
		Builtin_Type bt = {0};
		bool recognized = true;

		if (accept_tok(ctx, Token_kw_const))
			type->is_const = true;

		/* Gather all builtin type specifiers (like int, matrix(), field()) */
		while (recognized) {
			Token *tok = cur_tok(ctx);

			switch (tok->type) {
			case Token_kw_void:
				bt.is_void = true;
				advance_tok(ctx);
			break;
			case Token_kw_int:
				bt.is_integer = true;
				bt.bitness = 0; /* Not specified */
				advance_tok(ctx);
			break;
			case Token_kw_size_t:
				bt.is_integer = true;
				bt.bitness = sizeof(size_t)*8; /* @todo Assuming target is same architecture than host */
				bt.is_unsigned = true;
				advance_tok(ctx);
			break;
			case Token_kw_char:
				bt.is_char = true;
				bt.bitness = 8;
				advance_tok(ctx);
			break;
			case Token_kw_float:
				bt.is_float = true;
				bt.bitness = 32;
				advance_tok(ctx);
			break;
			case Token_kw_matrix: {
				bt.is_matrix = true;
				advance_tok(ctx);

				if (!accept_tok(ctx, Token_open_paren)) {
					report_error_expected(ctx, "'('", cur_tok(ctx));
					goto mismatch;
				}

				{ /* Parse dimension list */
					/* @todo Support constant expressions */
					while (cur_tok(ctx)->type == Token_int) {
						int dim = str_to_int(cur_tok(ctx)->text);
						bt.matrix_dim[bt.matrix_rank++] = dim;
						advance_tok(ctx);

						if (bt.matrix_rank > MAX_MATRIX_RANK) {
							report_error(ctx, "Too high rank for a matrix. Max is %i", MAX_MATRIX_RANK);
							goto mismatch;
						}

						if (!accept_tok(ctx, Token_comma))
							break;
					}
				}

				if (!accept_tok(ctx, Token_close_paren)) {
					report_error_expected(ctx, "')'", cur_tok(ctx));
					goto mismatch;
				}
			} break;
			case Token_kw_field: {
				AST_Node *dim_expr = NULL;
				AST_Literal dim_value;
				bt.is_field = true;
				advance_tok(ctx);

				if (!parse_expr_inside_parens(ctx, &dim_expr))
					goto mismatch;
				if (!eval_const_expr(&dim_value, dim_expr)) {
					report_error(ctx, "Field dimension must be a constant expression");
					goto mismatch;
				}
				if (dim_value.type != Literal_int) {
					report_error(ctx, "Field dimension must be an integer");
					goto mismatch;
				}
				bt.field_dim = dim_value.value.integer;
				/* Constant expression is removed */
				destroy_node(dim_expr);
			} break;
			default: recognized = false;
			}
			if (recognized)
				is_builtin = true;
		}

		/* Create builtin decls for types included in the type */
		if (is_builtin) {
			AST_Type_Decl *builtin_type = NULL;
			AST_Type_Decl *field_sub_type = NULL;
			AST_Type_Decl *matrix_sub_type = NULL;

			if (bt.is_field) {
				Builtin_Type sub_type = bt;
				sub_type.is_field = false;
				field_sub_type = create_builtin_decl(ctx, sub_type);
			}

			if (bt.is_matrix) {
				Builtin_Type sub_type = bt;
				sub_type.is_matrix = false;
				sub_type.is_field = false;
				matrix_sub_type = create_builtin_decl(ctx, sub_type);

				/* 'int matrix field' generates int, matrix, and field type decls.
				 * This sets matrix->builtin_sub_type_decl to the int type decl. */
				if (field_sub_type)
					field_sub_type->builtin_sub_type_decl = matrix_sub_type;
			}

			builtin_type = create_builtin_decl(ctx, bt);

			if (field_sub_type) {
				builtin_type->builtin_sub_type_decl = field_sub_type;
			} else if (matrix_sub_type) {
				builtin_type->builtin_sub_type_decl = matrix_sub_type;
			}

			found_decl = AST_BASE(builtin_type);
		} else {
			AST_Ident *ident = NULL;
			if (!parse_ident(ctx, &ident, NULL))
				goto mismatch;
			if (!resolve_ident(&ctx->parent_map, ident)) {
				destroy_node(AST_BASE(ident));
				goto mismatch;
			}
			found_decl = ident->decl;
			destroy_node(AST_BASE(ident));
		}
		ASSERT(found_decl);
		if (found_decl->type == AST_type_decl) {
			type->base_type_decl = (AST_Type_Decl*)found_decl;
		} else if (found_decl->type == AST_typedef) {
			CASTED_NODE(AST_Typedef, def, found_decl);
			type->base_typedef = def;
			type->base_type_decl = def->type->base_type_decl;
		} else {
			report_error_expected(ctx, "type name", cur_tok(ctx));
			goto mismatch;
		}
	}

	/* Pointer * */
	while (accept_tok(ctx, Token_mul))
		++type->ptr_depth;

	if (ret_ident && enclosing_decl) {
		/* Variable name */
		if (!parse_ident(ctx, ret_ident, enclosing_decl)) {
			report_error_expected(ctx, "identifier", cur_tok(ctx));
			goto mismatch;
		}
	}

	end_node_parsing(ctx);

	*ret_type = type;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

INTERNAL bool parse_var_decl(Parse_Ctx *ctx, AST_Node **ret, bool is_param_decl)
{
	AST_Var_Decl *decl = create_var_decl_node();
	begin_node_parsing(ctx, AST_BASE(decl));

	if (!parse_type_and_ident(ctx, &decl->type, &decl->ident, AST_BASE(decl)))
		goto mismatch;

	if (!is_param_decl) {
		if (accept_tok(ctx, Token_assign)) {
			if (!parse_expr(ctx, &decl->value, 0, NULL, true))
				goto mismatch;
		} else if (!accept_tok(ctx, Token_semi)) {
			report_error_expected(ctx, "';'", cur_tok(ctx));
			goto mismatch;
		}
	}

	end_node_parsing(ctx);

	*ret = AST_BASE(decl);
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

INTERNAL bool parse_func_decl(Parse_Ctx *ctx, AST_Node **ret)
{
	AST_Func_Decl *decl = create_func_decl_node();
	begin_node_parsing(ctx, AST_BASE(decl));

	if (!parse_type_and_ident(ctx, &decl->return_type, &decl->ident, AST_BASE(decl)))
		goto mismatch;

	if (!accept_tok(ctx, Token_open_paren)) {
		report_error_expected(ctx, "'('", cur_tok(ctx));
		goto mismatch;
	}

	{ /* Parse parameter declaration list */
		while (cur_tok(ctx)->type != Token_close_paren) {
			AST_Var_Decl *param_decl = NULL;
			if (cur_tok(ctx)->type == Token_comma)
				advance_tok(ctx);

			if (accept_tok(ctx, Token_ellipsis)) {
				decl->ellipsis = true;
				continue;
			}

			if (!parse_var_decl(ctx, (AST_Node**)&param_decl, true))
				goto mismatch;

			push_array(AST_Var_Decl_Ptr)(&decl->params, param_decl);
		}
	}

	if (!accept_tok(ctx, Token_close_paren)) {
		report_error_expected(ctx, "')'", cur_tok(ctx));
		goto mismatch;
	}

	if (cur_tok(ctx)->type == Token_semi) {
		/* No body */
		accept_tok(ctx, Token_semi);
	} else if (parse_scope(ctx, &decl->body, false)) {
		;
	} else {
		goto mismatch;
	}

	end_node_parsing(ctx);

	*ret = AST_BASE(decl);
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse example: { .... } */
INTERNAL bool parse_scope(Parse_Ctx *ctx, AST_Scope **ret, bool already_created)
{
	AST_Scope *scope = (already_created ? *ret : create_scope_node());

	begin_node_parsing(ctx, AST_BASE(scope));

	if (!accept_tok(ctx, Token_open_brace)) {
		report_error_expected(ctx, "'{'", cur_tok(ctx));
		goto mismatch;
	}

	while (!accept_tok(ctx, Token_close_brace)) {
		AST_Node *element = NULL;
		if (!parse_element(ctx, &element))
			goto mismatch;
		push_array(AST_Node_Ptr)(&scope->nodes, element);
	}

	end_node_parsing(ctx);

	*ret = scope;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse example: 1234 */
INTERNAL bool parse_literal(Parse_Ctx *ctx, AST_Node **ret)
{
	AST_Literal *literal = create_literal_node();
	Token *tok = cur_tok(ctx);

	begin_node_parsing(ctx, AST_BASE(literal));

	switch (tok->type) {
		case Token_int:
			literal->type = Literal_int;
			literal->value.integer = str_to_int(tok->text);
			literal->base_type_decl = create_builtin_decl(ctx, int_builtin_type());
			advance_tok(ctx);
		break;
		case Token_float:
			literal->type = Literal_float;
			literal->value.floating = str_to_float(tok->text);
			literal->base_type_decl = create_builtin_decl(ctx, float_builtin_type());
			advance_tok(ctx);
		break;
		case Token_string:
			literal->type = Literal_string;
			literal->value.string = tok->text;
			literal->base_type_decl = create_builtin_decl(ctx, char_builtin_type());
			advance_tok(ctx);
		break;
		case Token_kw_null:
			literal->type = Literal_null;
			advance_tok(ctx);
		break;
		case Token_open_paren:
			/* Compound literal */
			literal->type = Literal_compound;
			literal->value.compound.subnodes = create_array(AST_Node_Ptr)(2);

			/* (Type) part */
			advance_tok(ctx);
			if (!parse_type(ctx, &literal->value.compound.type))
				goto mismatch;
			if (!accept_tok(ctx, Token_close_paren)) {
				report_error_expected(ctx, "')'", cur_tok(ctx));
				goto mismatch;
			}

			/* {...} part */
			if (!accept_tok(ctx, Token_open_brace)) {
				report_error_expected(ctx, "'{'", cur_tok(ctx));
				goto mismatch;
			}
			if (!parse_arg_list(ctx, &literal->value.compound.subnodes, Token_close_brace))
				goto mismatch;
		break;
		case Token_open_brace:
			/* Initializer list */

			literal->type = Literal_compound;
			literal->value.compound.subnodes = create_array(AST_Node_Ptr)(2);

			advance_tok(ctx); /* Skip '{' */
			if (!parse_arg_list(ctx, &literal->value.compound.subnodes, Token_close_brace))
				goto mismatch;
		break;
		default:
			report_error_expected(ctx, "literal", cur_tok(ctx));
			goto mismatch;
	}

	end_node_parsing(ctx);

	*ret = AST_BASE(literal);
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

INTERNAL bool parse_uop(Parse_Ctx *ctx, AST_Node **ret)
{
	AST_Biop *biop = create_biop_node();

	begin_node_parsing(ctx, AST_BASE(biop));

	if (!is_unary_op(cur_tok(ctx)->type))
		goto mismatch;

	biop->type = cur_tok(ctx)->type;
	biop->is_top_level = (ctx->expr_depth == 1);
	advance_tok(ctx);

	/* The precedence should be higher than any binary operation, because -1 op 2 != -(1 op 2) */
	if (!parse_expr(ctx, &biop->rhs, UOP_PRECEDENCE, NULL, false)) {
		goto mismatch;
	}

	end_node_parsing(ctx);

	*ret = AST_BASE(biop);
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}


/* Parses 'foo, bar)' */
INTERNAL bool parse_arg_list(Parse_Ctx *ctx, Array(AST_Node_Ptr) *ret, Token_Type ending)
{
	while (cur_tok(ctx)->type != ending) {
		AST_Node *arg = NULL;
		if (cur_tok(ctx)->type == Token_comma)
			advance_tok(ctx);

		/* Normal expression */
		if (!parse_expr(ctx, (AST_Node**)&arg, 0, NULL, false))
			goto mismatch;

		push_array(AST_Node_Ptr)(ret, arg);
	}

	if (!accept_tok(ctx, ending)) {
		report_error(ctx, "List didn't end with '%s', but '%s'",
				tokentype_str(ending), BUF_STR_ARGS(cur_tok(ctx)->text));
		goto mismatch;
	}

	return true;

mismatch:
	return false;
}

/* Parse example: (4 * 2) */
INTERNAL bool parse_expr_inside_parens(Parse_Ctx *ctx, AST_Node **ret)
{
	if (!accept_tok(ctx, Token_open_paren)) {
		report_error_expected(ctx, "'('", cur_tok(ctx));
		goto mismatch;
	}

	if (!parse_expr(ctx, ret, 0, NULL, false))
		goto mismatch;

	if (!accept_tok(ctx, Token_close_paren)) {
		report_error_expected(ctx, "')'", cur_tok(ctx));
		goto mismatch;
	}

	return true;

mismatch:
	return false;
}


/* Functions '*_expr' are part of 'parse_expr', and therefore take lhs as a param */

/* Parse example: pre-parsed-expr + '(a, b, c)' */
INTERNAL bool parse_call_expr(Parse_Ctx *ctx, AST_Node **ret, AST_Node *expr, AST_Type *hint)
{
	AST_Call *call = create_call_node();
	Array(AST_Type) types = {0};
	begin_node_parsing(ctx, AST_BASE(call));

	if (expr->type != AST_ident) {
		goto mismatch;
	}

	if (expr->type == AST_ident)
		call->ident = (AST_Ident*)expr;

	if (!accept_tok(ctx, Token_open_paren)) {
		report_error_expected(ctx, "'('", cur_tok(ctx));
		goto mismatch;
	}

	if (!parse_arg_list(ctx, &call->args, Token_close_paren))
		goto mismatch;

	if (!resolve_call(&ctx->parent_map, call, hint)) {
		goto mismatch;
	}

	end_node_parsing(ctx);
	destroy_array(AST_Type)(&types);

	*ret = AST_BASE(call);
	return true;

mismatch:
	destroy_array(AST_Type)(&types);
	call->ident = NULL;
	cancel_node_parsing(ctx);
	return false;
}

INTERNAL bool parse_var_access_expr(Parse_Ctx *ctx, AST_Node **ret, AST_Node *expr)
{
	AST_Access *access = create_access_node();
	begin_node_parsing(ctx, AST_BASE(access));

	if (expr->type != AST_ident) {
		report_error_expected(ctx, "identifier", expr->begin_tok);
		goto mismatch;
	}

	{
		CASTED_NODE(AST_Ident, ident, expr);

		if (!resolve_ident(&ctx->parent_map, ident))
			goto mismatch;

		if (ident->decl->type == AST_var_decl) {
			access->base = AST_BASE(ident);
		} else {
			ident->decl = NULL;
			goto mismatch;
		}
	}

	end_node_parsing(ctx);

	*ret = AST_BASE(access);
	return true;

mismatch:
	access->base = NULL;
	cancel_node_parsing(ctx);
	return false;
}

INTERNAL bool parse_biop_expr(Parse_Ctx *ctx, AST_Node **ret, AST_Node *lhs)
{
	AST_Biop *biop = create_biop_node();
	AST_Node *rhs = NULL;
	Token *tok = cur_tok(ctx);
	AST_Type new_type_hint;
	AST_Type *new_type_hint_ptr = NULL;
	int prec = biop_prec(tok->type);
	int assoc = biop_assoc(tok->type);
	int next_min_prec;

	begin_node_parsing(ctx, AST_BASE(biop));

	if (!is_binary_op(cur_tok(ctx)->type))
		goto mismatch;

	if (assoc == -1)
		next_min_prec = prec + 1;
	else
		next_min_prec = prec;
	advance_tok(ctx);

	/* E.g. LHS of 'field = alloc_field(1);' chooses the allocation function through type hint */ 
	if (expr_type(&new_type_hint, lhs))
		new_type_hint_ptr = &new_type_hint;
	if (!parse_expr(ctx, &rhs, next_min_prec, new_type_hint_ptr, false))
		goto mismatch;

	/* @todo Type checking */
	/* @todo Operator overloading */
	biop->type = tok->type;
	biop->lhs = lhs;
	biop->rhs = rhs;
	biop->is_top_level = (ctx->expr_depth == 1);

	end_node_parsing(ctx);

	*ret = AST_BASE(biop);
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

INTERNAL bool parse_member_access_expr(Parse_Ctx *ctx, AST_Node **ret, AST_Node *base)
{
	AST_Access *access = create_access_node();
	AST_Type base_type;

	begin_node_parsing(ctx, AST_BASE(access));

	if (!accept_tok(ctx, Token_dot) && !accept_tok(ctx, Token_right_arrow))
		goto mismatch;

	if (!expr_type(&base_type, base)) {
		report_error(ctx, "Expression does not have accessible members");
		goto mismatch;
	}

	{
		AST_Ident *sub = NULL;
		access->base = base;
		access->is_member_access = true;

		{
			AST_Scope *base_type_scope = base_type.base_type_decl->body;
			ASSERT(base_type_scope);
			if (!parse_ident(ctx, &sub, NULL))
				goto mismatch;
			if (!resolve_ident_in_scope(sub, base_type_scope))
				goto mismatch;
			push_array(AST_Node_Ptr)(&access->args, AST_BASE(sub));
		}
	}

	*ret = AST_BASE(access);
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

INTERNAL bool parse_element_access_expr(Parse_Ctx *ctx, AST_Node **ret, AST_Node *base)
{
	AST_Access *access = create_access_node();
	AST_Type base_type;

	begin_node_parsing(ctx, AST_BASE(access));

	if (!accept_tok(ctx, Token_open_paren))
		goto mismatch;

	access->is_element_access = true;
	access->base = base;

	if (!expr_type(&base_type, access->base)) {
		report_error(ctx, "Invalid type");
		goto mismatch;
	}

	if (base_type.ptr_depth == 1) {
		access->implicit_deref = true;
	} else if (base_type.ptr_depth > 1) {
		report_error(ctx, "Trying to access elements of ptr (depth %i). Only 1 level of implicit dereferencing allowed.", base_type.ptr_depth);
		goto mismatch;
	}

	if (!parse_arg_list(ctx, &access->args, Token_close_paren))
		goto mismatch;

	*ret = AST_BASE(access);
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}


/* Parse example: var = 5 + 3 * 2; */
INTERNAL bool parse_expr(Parse_Ctx *ctx, AST_Node **ret, int min_prec, AST_Type *type_hint, bool semi)
{
	AST_Node *expr = NULL;

	++ctx->expr_depth;

	if (	parse_ident(ctx, (AST_Ident**)&expr, NULL) &&
			resolve_ident(&ctx->parent_map, (AST_Ident*)expr)) {
		/* If ident is var, wrap it in AST_Access */
		if (parse_var_access_expr(ctx, &expr, expr)) {
			;
		}
	} else if (parse_uop(ctx, &expr)) {
		;
	} else if (parse_literal(ctx, &expr)) {
		;
	} else if (parse_expr_inside_parens(ctx, &expr)) {
		;
	} else {
		report_error_expected(ctx, "identifier or literal", cur_tok(ctx));
		goto mismatch;
	}

	while (biop_prec(cur_tok(ctx)->type) >= min_prec) {

		if (parse_biop_expr(ctx, &expr, expr)) {
			;
		} else if (parse_call_expr(ctx, &expr, expr, type_hint)) {
			;
		} else if (parse_member_access_expr(ctx, &expr, expr)) {
			;
		} else if (parse_element_access_expr(ctx, &expr, expr)) {
			;
		} else {
			FAIL(("Expression parsing logic failed"));
		}
	}

	if (semi && !accept_tok(ctx, Token_semi)) {
		report_error_expected(ctx, "';'", cur_tok(ctx));
		goto mismatch;
	}

	/* @todo This leaks memory (?) */
	if (expr->type == AST_ident) {
		CASTED_NODE(AST_Ident, ident, expr);
		if (!ident->decl) {
			report_error(ctx, "'%s' not declared in this scope", ident->text.data);
			goto mismatch;
		}
	}

	--ctx->expr_depth;

	ASSERT(expr);
	*ret = expr;
	return true;

mismatch:

	--ctx->expr_depth;
	return false;
}

/* Parse example: return 42; */
INTERNAL bool parse_control(Parse_Ctx *ctx, AST_Node **ret)
{
	Token *tok = cur_tok(ctx);
	AST_Control *control = create_control_node();
	control->type = tok->type;

	begin_node_parsing(ctx, AST_BASE(control));

	switch (tok->type) {
		case Token_kw_return: {
			advance_tok(ctx);
			if (!parse_element(ctx, &control->value)) {
				report_error_expected(ctx, "return value", cur_tok(ctx));
				goto mismatch;
			}
		} break;
		case Token_kw_goto: {
			advance_tok(ctx);
			if (!parse_element(ctx, &control->value)) {
				report_error_expected(ctx, "goto label", cur_tok(ctx));
				goto mismatch;
			}
		} break;
		case Token_kw_continue:
		case Token_kw_break:
			advance_tok(ctx);
		break;
		default:
			report_error_expected(ctx, "control statement", cur_tok(ctx));
			goto mismatch;
	}

	end_node_parsing(ctx);

	*ret = AST_BASE(control);
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse example: if (..) { .. } else { .. } */
INTERNAL bool parse_cond(Parse_Ctx *ctx, AST_Node **ret)
{
	AST_Cond *cond = create_cond_node();
	begin_node_parsing(ctx, AST_BASE(cond));

	if (!accept_tok(ctx, Token_kw_if)) {
		report_error_expected(ctx, "'if'", cur_tok(ctx));
		goto mismatch;
	}

	if (!parse_expr_inside_parens(ctx, &cond->expr))
		goto mismatch;

	if (!accept_tok(ctx, Token_semi)) {
		if (parse_scope(ctx, &cond->body, false)) {
			;
		} else {
			/* Always have scope, because then we can expand one statement to multiple ones without worry */
			AST_Scope *scope = create_scope_node();
			AST_Node *elem = NULL;
			cond->body = scope;
			cond->implicit_scope = true;

			if (parse_element(ctx, &elem)) {
				push_array(AST_Node_Ptr)(&scope->nodes, elem);
			} else {
				goto mismatch;
			}
		}
	}

	if (accept_tok(ctx, Token_kw_else)) {
		if (!accept_tok(ctx, Token_semi)) {
			if (!parse_element(ctx, &cond->after_else))
				goto mismatch;
		}

		if (cond->after_else) {
			if (	cond->after_else->type != AST_scope ||
					cond->after_else->type != AST_cond) {
				report_error_expected(ctx, "'if', '{' or ';'", cur_tok(ctx));
			}
		}
	}

	end_node_parsing(ctx);
	*ret = AST_BASE(cond);
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse example: while(1) { .. } -- for (i = 0; i < 10; ++i) { .. } */
INTERNAL bool parse_loop(Parse_Ctx *ctx, AST_Node **ret)
{
	AST_Loop *loop = create_loop_node();
	begin_node_parsing(ctx, AST_BASE(loop));

	if (accept_tok(ctx, Token_kw_for)) {
		if (!accept_tok(ctx, Token_open_paren)) {
			report_error_expected(ctx, "'('", cur_tok(ctx));
			goto mismatch;
		}

		if (!parse_element(ctx, &loop->init))
			goto mismatch;

		if (!parse_expr(ctx, &loop->cond, 0, NULL, true))
			goto mismatch;

		if (!parse_expr(ctx, &loop->incr, 0, NULL, false)) {
			goto mismatch;
		}

		if (!accept_tok(ctx, Token_close_paren)) {
			report_error_expected(ctx, "')'", cur_tok(ctx));
			goto mismatch;
		}
	} else if (accept_tok(ctx, Token_kw_while)) {
		if (!parse_expr_inside_parens(ctx, &loop->cond))
			goto mismatch;
	} else {
		report_error_expected(ctx, "beginning of a loop", cur_tok(ctx));
		goto mismatch;
	}

	if (!accept_tok(ctx, Token_semi)) {
		if (parse_scope(ctx, &loop->body, false)) {
			;
		} else {
			/* Always have scope, because then we can expand one statement to multiple ones without worry */
			AST_Scope *scope = create_scope_node();
			AST_Node *elem = NULL;
			loop->body = scope;
			loop->implicit_scope = true;

			if (parse_element(ctx, &elem)) {
				push_array(AST_Node_Ptr)(&scope->nodes, elem);
			} else {
				goto mismatch;
			}
		}
	}

	end_node_parsing(ctx);
	*ret = AST_BASE(loop);
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse example: for_field (output; input) { .. } */
INTERNAL bool parse_parallel(Parse_Ctx *ctx, AST_Node **ret)
{
	AST_Parallel *parallel = create_parallel_node();
	begin_node_parsing(ctx, AST_BASE(parallel));

	if (!accept_tok(ctx, Token_kw_parallel))
		goto mismatch;

	if (!accept_tok(ctx, Token_open_paren)) {
		report_error_expected(ctx, "'('", cur_tok(ctx));
		goto mismatch;
	}

	if (!parse_expr(ctx, &parallel->output, 0, NULL, true))
		goto mismatch;

	{ /* Fetch dimension from output type */
		AST_Type out_type;
		if (!expr_type(&out_type, parallel->output) || !out_type.base_type_decl->is_builtin) {
			report_error(ctx, "Parallel loop has invalid output type");
			goto mismatch;
		}

		parallel->dim = out_type.base_type_decl->builtin_type.field_dim;
	}

	if (!parse_expr(ctx, &parallel->input, 0, NULL, false))
		goto mismatch;

	if (!accept_tok(ctx, Token_close_paren)) {
		report_error_expected(ctx, "')'", cur_tok(ctx));
		goto mismatch;
	}

	{
		AST_Scope *scope = create_scope_node();
		{ /* Builtin declaration of 'int matrix(dim) id;' to the beginning of the loop */
			AST_Type_Decl *bt_decl;
			AST_Var_Decl *id_decl;
			Builtin_Type bt = {0};
			bt.is_integer = true;
			bt.is_matrix = true;
			bt.matrix_rank = 1;
			bt.matrix_dim[0] = parallel->dim;
			bt_decl = create_builtin_decl(ctx, bt);

			id_decl = create_var_decl(bt_decl, create_ident_with_text(NULL, "id"), NULL);
			push_array(AST_Node_Ptr)(&scope->nodes, AST_BASE(id_decl));
		}

		if (!parse_scope(ctx, &scope, true))
			goto mismatch;

		parallel->body = scope;
	}

	end_node_parsing(ctx);
	*ret = AST_BASE(parallel);
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse the next self-contained thing - var decl, function decl, statement, expr... */
/* @todo Should this be named "parse_statement"? */
INTERNAL bool parse_element(Parse_Ctx *ctx, AST_Node **ret)
{
	AST_Node *result = NULL;

	/* @todo Heuristic */
	if (accept_tok(ctx, Token_semi)) {
		report_error(ctx, "Unexpected ';'");
		goto mismatch;
	} else if (parse_type_decl(ctx, &result))
		;
	else if (parse_typedef(ctx, &result))
		;
	else if (parse_var_decl(ctx, &result, false))
		;
	else if (parse_func_decl(ctx, &result))
		;
	else if (parse_expr(ctx, &result, 0, NULL, true))
		;
	else if (parse_control(ctx, &result))
		;
	else if (parse_scope(ctx, (AST_Scope**)&result, false))
		;
	else if (parse_cond(ctx, &result))
		;
	else if (parse_loop(ctx, &result))
		;
	else if (parse_parallel(ctx, &result))
		;
	else 
		goto mismatch;

	*ret = result;
	return true;

mismatch:
	return false;
}

AST_Scope *parse_tokens(Token *toks)
{
	bool failure = false;
	Parse_Ctx ctx = {0};
	AST_Scope *root = create_ast();

	ctx.root = root;
	ctx.parse_stack = create_array(Parse_Stack_Frame)(32);
	ctx.tok = ctx.first_tok = toks;
	ctx.parent_map = create_parent_map(NULL);
	if (is_comment_tok(ctx.tok->type))
		advance_tok(&ctx);

	{ /* Create C builtin types (backends should be able to rely that they exist in the AST) */
		create_builtin_decl(&ctx, void_builtin_type());
		create_builtin_decl(&ctx, int_builtin_type());
		create_builtin_decl(&ctx, char_builtin_type());
		/* @todo Rest */
	}

	begin_node_parsing(&ctx, AST_BASE(root));
	while (ctx.tok->type != Token_eof) {
		AST_Node *elem = NULL;
		if (!parse_element(&ctx, &elem)) {
			failure = true;
			break;
		}
		push_array(AST_Node_Ptr)(&root->nodes, elem); 
	}
	end_node_parsing(&ctx);

	{ /* Insert builtin declarations to beginning of root node */
		Array(AST_Node_Ptr) new_nodes = create_array(AST_Node_Ptr)(ctx.parent_map.builtin_decls.size + root->nodes.size);
		int i;
		/* @todo Array insert function */
		for (i = 0; i < ctx.parent_map.builtin_decls.size; ++i) {
			push_array(AST_Node_Ptr)(&new_nodes, ctx.parent_map.builtin_decls.data[i]);
		}
		for (i = 0; i < root->nodes.size; ++i) {
			push_array(AST_Node_Ptr)(&new_nodes, root->nodes.data[i]);
		}
		destroy_array(AST_Node_Ptr)(&root->nodes);
		root->nodes = new_nodes;
	}

	if (failure) {
		Token *tok = ctx.error_tok;
		const char *msg = ctx.error_msg.data;
		if (tok && msg) {
			printf("Error at line %i near token '%.*s':\n   %s\n",
					tok->line, BUF_STR_ARGS(tok->text), msg);
		} else {
			printf("Internal parser error (excuse)\n");
		}
		printf("Compilation failed\n");
		destroy_ast(root);
		root = NULL;
	}

	destroy_array(char)(&ctx.error_msg);
	destroy_parent_map(&ctx.parent_map);
	destroy_array(Parse_Stack_Frame)(&ctx.parse_stack);

	return root;
}

