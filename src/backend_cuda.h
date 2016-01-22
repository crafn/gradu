#ifndef BACKEND_CUDA_H
#define BACKEND_CUDA_H

#include "core.h"
#include "parse.h"

QC_Array(char) gen_cuda_code(AST_Scope *root);

#endif
