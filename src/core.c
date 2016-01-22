#include "core.h"

DEFINE_ARRAY(char)
DEFINE_ARRAY(int)

bool buf_str_equals(QC_Buf_Str a, QC_Buf_Str b)
{
	if (a.len != b.len)
		return false;
	return !strncmp(a.buf, b.buf, a.len);
}

QC_Buf_Str c_str_to_buf_str(const char* str)
{
	QC_Buf_Str b = {0};
	b.buf = str;
	b.len = strlen(str);
	return b;
}

void safe_vsprintf(QC_Array(char) *buf, const char *fmt, va_list args)
{
	char tmp[1024*100]; /* :( */
	int len;
	int i;

	/* @todo Find open source non-gpl snprintf */
	len = vsprintf(tmp, fmt, args);
	if (len < 0)
		return;
	if (len > (int)sizeof(tmp)/2) /* Crappy failsafe */
		abort();

	if (buf->size > 0 && buf->data[buf->size - 1] == '\0')
		pop_array(char)(buf);

	for (i = 0; i < len; ++i)
		push_array(char)(buf, tmp[i]);
	push_array(char)(buf, '\0');
}

void append_str(QC_Array(char) *buf, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	safe_vsprintf(buf, fmt, args);
	va_end(args);
}


