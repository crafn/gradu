#ifndef PARSE_H
#define PARSE_H

#include "ast.h"
#include "tokenize.h"

int biop_prec(Token_Type type);
int biop_assoc(Token_Type type);

/* Created AST will have pointers to tokens */
AST_Scope *parse_tokens(Token *toks);

#endif
