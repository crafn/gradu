#ifndef PARSE_H
#define PARSE_H

#include "ast.h"
#include "tokenize.h"

/* Created AST will have pointers to tokens */
AST_Scope *parse_tokens(Token *toks);

#endif
