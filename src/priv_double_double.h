/**
 * Copyright (C) Takuo Hada 2016
 * @author t.hada 2016/08/11
 * 
 * double-double implementation.
 */
#pragma once

#include <assert.h>
#include <stdint.h>

#define __X86_FMA

#if defined(__X86_FMA)
#include <immintrin.h>
#endif

typedef struct mg_dd
{
    double lo;
    double hi;
} mg_dd_t;

static inline uint64_t mg_dd_init(mg_dd_t *op1)
{
    op1->hi = 0.0;
    op1->lo = 0.0;
}

static inline void mg_dd_twosum(double a, double b, mg_dd_t *ret)
{
	double s = a + b;
	double v = s - a;
	double e = (a - (s - v)) + (b - v);
	
	ret->hi = s;
	ret->lo = e;
}

static inline void mg_dd_quick_twosum(double a, double b, mg_dd_t *ret)
{
	double s = a + b;
	double e = b - (s - a);
	
	ret->hi = s;
	ret->lo = e;
}


static inline void mg_dd_twoprod(double a, double b, mg_dd_t *ret)
{
#if defined(__X86_FMA)
	double p = a * b;
	double e;
	_mm_store_sd(&e, _mm_fmsub_sd(
			_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(p)));
#else
	double a_tmp = a * (double)(1 << 27);
	double a_hi = a_tmp - (a_tmp - a);
	double a_lo = a - a_hi;
	
	double b_tmp = b * (double)(1 << 27);
	double b_hi = b_tmp - (b_tmp - b);
	double b_lo = b - b_hi;

	double p = a * b;
	double e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
#endif
	
	ret->hi = p;
	ret->lo = e;
}

static inline uint64_t mg_dd_get_uint64(const mg_dd_t *op1)
{
	if(op1->hi >= 0.0) {
		return (uint64_t)op1->hi + (uint64_t)op1->lo;
	} else {
		return 0;
	}
}

static inline double mg_dd_get_double(const mg_dd_t *op1)
{
    return op1->hi + op1->lo;
}

static inline void mg_dd_value_of_uint64(uint64_t value, mg_dd_t *ret)
{
	double hi = (double)value;
	double lo;
	if((uint64_t)hi <= value) {
		lo = (double)(value - (uint64_t)hi);
	} else {
		lo = -(double)((uint64_t)(hi) - value);
	}

	mg_dd_twosum(hi, lo, /*out*/ret);
}

static inline void mg_dd_value_of_double(double value, mg_dd_t *ret)
{
	ret->hi = value;
	ret->lo = 0.0;
}

static inline void mg_dd_add(const mg_dd_t *op1, const mg_dd_t *op2, mg_dd_t *ret)
{
	mg_dd_t z;
	mg_dd_twosum(op1->hi, op2->hi, &z);
	
	z.lo += op1->lo + op2->lo;
	
	mg_dd_twosum(z.hi, z.lo, /*out*/ret);
}

static inline void mg_dd_sub(const mg_dd_t *op1, const mg_dd_t *op2, mg_dd_t *ret)
{
	mg_dd_t z;
	mg_dd_twosum(op1->hi, -op2->hi, &z);
	
	z.lo += op1->lo - op2->lo;
	
	mg_dd_twosum(z.hi, z.lo, /*out*/ret);
}

static inline void mg_dd_mul(const mg_dd_t *op1, const mg_dd_t *op2, mg_dd_t *ret)
{
	mg_dd_t z;
	mg_dd_twoprod(op1->hi, op2->hi, &z);
	
	z.lo += op1->hi * op2->lo + op1->lo * op2->hi + op1->lo * op2->lo;
	
	mg_dd_twosum(z.hi, z.lo, /*out*/ret);
}

static inline void mg_dd_div(const mg_dd_t *op1, const mg_dd_t *op2, mg_dd_t *ret)
{
	double z_hi = op1->hi / op2->hi;
	mg_dd_t t;
	mg_dd_twoprod(-z_hi, op2->hi, &t);
	
	double z_lo = (((t.hi + op1->hi) - z_hi * op2->lo) + t.lo) / (op2->hi + op2->lo);
	//double z_lo = ((((t.hi + op1->hi) - z_hi * op2->lo) + op1->lo) + t.lo) / op2->hi;

	mg_dd_twosum(z_hi, z_lo, /*out*/ret);
}

static inline void mg_dd_mul_double(const mg_dd_t *op1, double op2, mg_dd_t *ret)
{
	mg_dd_t z;
	mg_dd_twoprod(op1->hi, op2, &z);
	
	z.lo += op1->lo * op2;
	
	mg_dd_twosum(z.hi, z.lo, /*out*/ret);
}

static inline int mg_dd_compare(const mg_dd_t *op1, double value)
{
	double d = op1->hi + op1->lo;

	if(d < value)
		return -1;
	else if(d > value)
		return 1;
	else
		return 0;
}

static inline void __mg_dd_get_decimal_part(mg_dd_t *op1)
{
	uint64_t hi = (uint64_t)op1->hi;
	int64_t lo = (int64_t)op1->lo;

	op1->hi -= (double)hi;
	op1->lo -= (double)lo;
}
