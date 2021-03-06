#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <magica/decimal.h>

#include "mg_assert.h"

static void int64_convert_test(int64_t value, const char *ret)
{
	char strbuf[1000];
	int size;
	mg_decimal value1;
	int64_t value2;

	mg_assert(mg_decimal_value_of_int64(value, &value1) == 0);

	mg_assert(mg_decimal_to_int64(&value1, /*out*/&value2) == 0);

	mg_assert(value == value2);

	mg_assert(mg_decimal_to_string(&value1, strbuf, 1000, &size) == 0);

	mg_assert(strcmp(strbuf, ret) == 0);
}

void decimal_convert_int64_test()
{
	clock_t tm = clock();
	
	int64_convert_test(1000, "1000");
	int64_convert_test(-1000, "-1000");
	int64_convert_test(-9999999999999, "-9999999999999");
	int64_convert_test(-999999999999999999, "-999999999999999999");
	int64_convert_test(-425415311, "-425415311");
	int64_convert_test(9223372036854775807, "9223372036854775807");
	int64_convert_test(-9223372036854775808LL, "-9223372036854775808");
	int64_convert_test(0, "0");

	{
		mg_decimal value;
		mg_assert(mg_decimal_parse_string("1234567891234156498715634865156465151654152165453132416854114", &value) == MG_DECIMAL_ERROR_OVERFLOW);
	}

	printf("TEST mg_decimal convert int64 methods: OK\n");
}
