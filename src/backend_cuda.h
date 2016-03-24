#ifndef QC_BACKEND_CUDA_H
#define QC_BACKEND_CUDA_H

#include "config.h"
#include "core.h"
#include "parse.h"

QC_API QC_Array(char) qc_gen_cuda_code(QC_AST_Scope *root);

#endif
