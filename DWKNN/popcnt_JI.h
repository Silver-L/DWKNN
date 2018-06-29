#pragma once
#ifndef __POPCNT_JI_H
#define __POPCNT_JI_H

/**
* @popcnt JI
*
* @Author Zhihui Lu
* @Sponsor Atsushi Saito
* @date 2018/06/02
**/

/**
* @Functions
*
* @Element Count
* @Pack Data
* @Calculate JI
**/

#include <intrin.h>
#include <iostream>

namespace popcnt {

	const unsigned int uint32_table[32] =
	{
		0x00000001,	0x00000002,	0x00000004,	0x00000008,
		0x00000010,	0x00000020,	0x00000040,	0x00000080,
		0x00000100,	0x00000200,	0x00000400,	0x00000800,
		0x00001000,	0x00002000,	0x00004000,	0x00008000,
		0x00010000,	0x00020000,	0x00040000,	0x00080000,
		0x00100000,	0x00200000,	0x00400000,	0x00800000,
		0x01000000,	0x02000000,	0x04000000,	0x08000000,
		0x10000000,	0x20000000,	0x40000000,	0x80000000
	};

	/**
	* @Get element count of package
	* @Usage: [image_size]
	**/
	int element_count(int size) {
		int ele_cnt = 0;
		if (size == 0) {
			ele_cnt = size / 32;
		}
		else ele_cnt = int(size / 32) + 1;
		return ele_cnt;
	}

	/**
	* @Pack data
	* @Usage: [Label_image] [pointer] [image_size]
	**/
	template<class T>
	void pack(T &raw, unsigned long long *pbw, int m) {
		int i, k, q, r;
		q = m / 32;
		r = m - q * 32;
		for (i = 0; i<q; i++, pbw++) {
			*pbw = 0;
			for (k = 0; k<32; k++)
				*pbw += (*raw++) ? uint32_table[k] : 0;
		}
		*pbw = 0;
		for (k = 0; k<r; k++)
			*pbw += (*raw++) ? uint32_table[k] : 0;
	}

	/**
	* @Calculate JI
	* @Usage: [pointer_A] [pointer_B] [element_count]
	**/
	double JI(unsigned long long *P, unsigned long long *Q, int ele_cnt) {
		int cap = 0, cup = 0;

		for (int i = 0; i<ele_cnt; i++, P++, Q++) {
			cap += (int)__popcnt64(*P & *Q);
			cup += (int)__popcnt64(*P | *Q);
		}
		return (double)cap / (double)cup;
	}

};

#endif __POPCNT_JI_H