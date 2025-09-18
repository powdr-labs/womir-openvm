// Adapted from musl's memmove.c

_Static_assert(sizeof(unsigned int) == 4,   "int must be 4 bytes");
_Static_assert(sizeof(void*) == 4, "pointer must be 4 bytes");

typedef unsigned int uintptr_t;

#ifdef __GNUC__
typedef __attribute__((__may_alias__)) unsigned int WT;
#define WS (sizeof(WT))
#endif

__attribute__((export_name("memory_copy")))
void memory_copy(void *dest, const void *src, unsigned int n)
{
	char *d = dest;
	const char *s = src;

	if (d==s) return;

	if (d<s) {
#ifdef __GNUC__
		if ((uintptr_t)s % WS == (uintptr_t)d % WS) {
			while ((uintptr_t)d % WS) {
				if (!n--) return;
				*d++ = *s++;
			}
			for (; n>=WS; n-=WS, d+=WS, s+=WS) *(WT *)d = *(WT *)s;
		}
#endif
		for (; n; n--) *d++ = *s++;
	} else {
#ifdef __GNUC__
		if ((uintptr_t)s % WS == (uintptr_t)d % WS) {
			while ((uintptr_t)(d+n) % WS) {
				if (!n--) return;
				d[n] = s[n];
			}
			while (n>=WS) n-=WS, *(WT *)(d+n) = *(WT *)(s+n);
		}
#endif
		while (n) n--, d[n] = s[n];
	}
}
