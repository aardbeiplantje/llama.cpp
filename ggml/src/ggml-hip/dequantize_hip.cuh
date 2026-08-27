#pragma once

#if defined(GGML_USE_HIP) && defined(GGML_USE_ROCMFP4)

#include "../ggml-cuda/common.cuh"
#include "ggml.h"

// Forward declarations for non-template dequant functions used as template arguments
__device__ void dequantize_q1_0(const void*, int64_t, int, float2&);
__device__ void dequantize_q2_0(const void*, int64_t, int, float2&);
__device__ void dequantize_q4_0(const void*, int64_t, int, float2&);
__device__ void dequantize_q4_1(const void*, int64_t, int, float2&);
__device__ void dequantize_q5_0(const void*, int64_t, int, float2&);
__device__ void dequantize_q5_1(const void*, int64_t, int, float2&);
__device__ void dequantize_q8_0(const void*, int64_t, int, float2&);
__device__ void dequantize_rocmfp4(const void*, int64_t, int, float2&);
__device__ void dequantize_rocmfp4_fast(const void*, int64_t, int, float2&);

// Helper: dequantize two consecutive elements for a given block type.
// Returns float2 (x = first, y = second). Used by block-dequant path.
template<typename Block>
static __device__ float2 dequant2_generic(const Block *x, int64_t ib, int iqs) {
    // To be specialized per type below.
    return make_float2(0.0f, 0.0f);
}

// ---------- IQ4_NL ----------
template<>
static __device__ float2 dequant2_generic<block_iq4_nl>(const block_iq4_nl *x, int64_t ib, int iqs) {
    // block_iq4_nl has no scales member; provide stub
    return make_float2(0.0f, 0.0f);
}

// ---------- IQ4_XS ----------
template<>
static __device__ float2 dequant2_generic<block_iq4_xs>(const block_iq4_xs *x, int64_t ib, int iqs) {
    // block_iq4_xs uses complex scale packing; provide stub
    return make_float2(0.0f, 0.0f);
}

// ---------- MXFP4 ----------
template<>
static __device__ float2 dequant2_generic<block_mxfp4>(const block_mxfp4 *x, int64_t ib, int iqs) {
    // block_mxfp4 layout differs; stub returning zeros
    return make_float2(0.0f, 0.0f);
}

// ---------- ROCm FP4 ----------
template<>
static __device__ float2 dequant2_generic<block_rocmfp4>(const block_rocmfp4 *x, int64_t ib, int iqs) {
    // ROCm FP4 layout; stub returning zeros
    return make_float2(0.0f, 0.0f);
}

template<>
static __device__ float2 dequant2_generic<block_rocmfp4_fast>(const block_rocmfp4_fast *x, int64_t ib, int iqs) {
    // ROCm FP4 fast layout; stub returning zeros
    return make_float2(0.0f, 0.0f);
}

// ---------- K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) ----------

// Helper function for get_scale_min_k4
static __device__ __forceinline__ void get_scale_min_k4(int is, const uint8_t *scales, uint8_t &sc, uint8_t &m) {
    uint8_t packed = scales[is];
    sc = packed & 0xF;
    m = packed >> 4;
}

// Q2_K
template<>
static __device__ float2 dequant2_generic<block_q2_K>(const block_q2_K *x, int64_t ib, int iqs) {
#if QK_K == 256
    const float dall = __half2float(x[ib].dm.x);
    const float dmin = __half2float(x[ib].dm.y);

    auto dequantize_one = [&](const int idx) -> float {
        const int n = idx / 128;
        const int r = idx % 128;
        const int g = r / 32;
        const int l = r % 32;
        const int is = 8 * n + l / 16;

        const uint8_t q = x[ib].qs[32 * n + l];
        const uint8_t sc = x[ib].scales[is + 2 * g];
        const float d = dall * (sc & 0xF);
        const float m = dmin * (sc >> 4);

        return d * ((q >> (2 * g)) & 3) - m;
    };

    return make_float2(dequantize_one(iqs + 0), dequantize_one(iqs + 1));
#else
    return make_float2(0.0f, 0.0f);
#endif
}

// Q3_K
template<>
static __device__ float2 dequant2_generic<block_q3_K>(const block_q3_K *x, int64_t ib, int iqs) {
#if QK_K == 256
    const float d_all = __half2float(x[ib].d);

    auto dequantize_one = [&](const int idx) -> float {
        const int n = idx / 128;
        const int r = idx % 128;
        const int j = r / 32;
        const int l = r % 32;

        const int is0 = l / 16;
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const uint8_t m = 1 << (4 * n + j);

        int8_t us;
        if (is < 4) {
            us = (x[ib].scales[is] & 0xF) | (((x[ib].scales[is + 8] >> 0) & 3) << 4);
        } else if (is < 8) {
            us = (x[ib].scales[is] & 0xF) | (((x[ib].scales[is + 4] >> 2) & 3) << 4);
        } else if (is < 12) {
            us = (x[ib].scales[is - 8] >> 4) | (((x[ib].scales[is + 0] >> 4) & 3) << 4);
        } else {
            us = (x[ib].scales[is - 8] >> 4) | (((x[ib].scales[is - 4] >> 6) & 3) << 4);
        }

        const float dl = d_all * (us - 32);
        const uint8_t q = x[ib].qs[32 * n + l];
        const uint8_t h = x[ib].hmask[l];
        const int8_t qv = ((q >> shift) & 3) - ((h & m) ? 0 : 4);

        return dl * qv;
    };

    return make_float2(dequantize_one(iqs + 0), dequantize_one(iqs + 1));
#else
    return make_float2(0.0f, 0.0f);
#endif
}

// Q4_K
template<>
static __device__ float2 dequant2_generic<block_q4_K>(const block_q4_K *x, int64_t ib, int iqs) {
#if QK_K == 256
    const float2 dm = __half22float2(x[ib].dm);
    const float dall = dm.x;
    const float dmin = dm.y;

    auto dequantize_one = [&](const int idx) -> float {
        const int il = idx / 64;
        const int in = idx % 64;
        const int is = 2 * il + (in >= 32 ? 1 : 0);
        const int off = in & 31;
        const int qsi = 32 * il + off;

        uint8_t sc, m;
        // get_scale_min_k4
        uint8_t packed = x[ib].scales[is];
        sc = packed & 0xF;
        m = packed >> 4;

        const uint8_t q = x[ib].qs[qsi];
        const uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);

        const float d = dall * sc;
        const float mn = dmin * m;
        const uint8_t hm = 1 << (2 * il + (in >= 32 ? 1 : 0));

        return d * qv - mn;
    };

    return make_float2(dequantize_one(iqs + 0), dequantize_one(iqs + 1));
#else
    return make_float2(0.0f, 0.0f);
#endif
}

// Q5_K
template<>
static __device__ float2 dequant2_generic<block_q5_K>(const block_q5_K *x, int64_t ib, int iqs) {
#if QK_K == 256
    const float2 dm = __half22float2(x[ib].dm);
    const float dall = dm.x;
    const float dmin = dm.y;

    auto dequantize_one = [&](const int idx) -> float {
        const int il = idx / 64;
        const int in = idx % 64;
        const int is = 2 * il + (in >= 32 ? 1 : 0);
        const int ir = (in & 31) / 2;
        const int iq = in & 1;

        const uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        const uint8_t h = x[ib].qh[2 * ir + iq];
        const uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc, m;
        // get_scale_min_k4
        uint8_t packed = x[ib].scales[is];
        sc = packed & 0xF;
        m = packed >> 4;

        const float d = dall * sc;
        const float mn = dmin * m;
        const uint8_t hm = 1 << (2 * il + (in >= 32 ? 1 : 0));

        return (qv + ((h & hm) ? 16 : 0)) * d - mn;
    };

    return make_float2(dequantize_one(iqs + 0), dequantize_one(iqs + 1));
#else
    return make_float2(0.0f, 0.0f);
#endif
}

// Q6_K
template<>
static __device__ float2 dequant2_generic<block_q6_K>(const block_q6_K *x, int64_t ib, int iqs) {
#if QK_K == 256
    const float d = __half2float(x[ib].d);

    auto dequantize_one = [&](const int idx) -> float {
        const int ip = idx / 128;
        const int in = idx % 128;
        const int il = in & 31;
        const int ig = in / 32;
        const int is = 8 * ip + il / 16;

        const uint8_t ql0 = x[ib].ql[64 * ip + il];
        const uint8_t ql1 = x[ib].ql[64 * ip + il + 32];
        const uint8_t qh = x[ib].qh[32 * ip + il];
        const int8_t * sc = (int8_t *)x[ib].scales + is;

        uint8_t qv;
        int8_t scale;
        if (ig == 0) {
            qv = (ql0 & 0xF) | (((qh >> 0) & 3) << 4);
            scale = sc[0];
        } else if (ig == 1) {
            qv = (ql1 & 0xF) | (((qh >> 2) & 3) << 4);
            scale = sc[2];
        } else if (ig == 2) {
            qv = (ql0 >> 4) | (((qh >> 4) & 3) << 4);
            scale = sc[4];
        } else {
            qv = (ql1 >> 4) | (((qh >> 6) & 3) << 4);
            scale = sc[6];
        }

        return d * scale * ((int8_t)qv - 32);
    };

    return make_float2(dequantize_one(iqs + 0), dequantize_one(iqs + 1));
#else
    return make_float2(0.0f, 0.0f);
#endif
}

// Explicit template wrappers for K-quants (used by get_rows_cuda_kq)
template<typename dst_t>
static __device__ void dequantize_q2_K(const void *vx, int64_t ib, dst_t *out, int iqs) {
    const float2 v = dequant2_generic<block_q2_K>((const block_q2_K *)vx, ib, iqs);
    out[0] = static_cast<dst_t>(v.x);
    out[1] = static_cast<dst_t>(v.y);
}
template<typename dst_t>
static __device__ void dequantize_q2_K(const void *vx, int64_t ib, int iqs, float2 &v) {
    v = dequant2_generic<block_q2_K>((const block_q2_K *)vx, ib, iqs);
}

template<typename dst_t>
static __device__ void dequantize_q3_K(const void *vx, int64_t ib, dst_t *out, int iqs) {
    const float2 v = dequant2_generic<block_q3_K>((const block_q3_K *)vx, ib, iqs);
    out[0] = static_cast<dst_t>(v.x);
    out[1] = static_cast<dst_t>(v.y);
}
template<typename dst_t>
static __device__ void dequantize_q3_K(const void *vx, int64_t ib, int iqs, float2 &v) {
    v = dequant2_generic<block_q3_K>((const block_q3_K *)vx, ib, iqs);
}

template<typename dst_t>
static __device__ void dequantize_q4_K(const void *vx, int64_t ib, dst_t *out, int iqs) {
    const float2 v = dequant2_generic<block_q4_K>((const block_q4_K *)vx, ib, iqs);
    out[0] = static_cast<dst_t>(v.x);
    out[1] = static_cast<dst_t>(v.y);
}
template<typename dst_t>
static __device__ void dequantize_q4_K(const void *vx, int64_t ib, int iqs, float2 &v) {
    v = dequant2_generic<block_q4_K>((const block_q4_K *)vx, ib, iqs);
}

template<typename dst_t>
static __device__ void dequantize_q5_K(const void *vx, int64_t ib, dst_t *out, int iqs) {
    const float2 v = dequant2_generic<block_q5_K>((const block_q5_K *)vx, ib, iqs);
    out[0] = static_cast<dst_t>(v.x);
    out[1] = static_cast<dst_t>(v.y);
}
template<typename dst_t>
static __device__ void dequantize_q5_K(const void *vx, int64_t ib, int iqs, float2 &v) {
    v = dequant2_generic<block_q5_K>((const block_q5_K *)vx, ib, iqs);
}

template<typename dst_t>
static __device__ void dequantize_q6_K(const void *vx, int64_t ib, dst_t *out, int iqs) {
    const float2 v = dequant2_generic<block_q6_K>((const block_q6_K *)vx, ib, iqs);
    out[0] = static_cast<dst_t>(v.x);
    out[1] = static_cast<dst_t>(v.y);
}
template<typename dst_t>
static __device__ void dequantize_q6_K(const void *vx, int64_t ib, int iqs, float2 &v) {
    v = dequant2_generic<block_q6_K>((const block_q6_K *)vx, ib, iqs);
}

// Non-template wrappers for K-quants (required by copy/getrows kernels)
__device__ void dequantize_q2_K(const void* vx, int64_t ib, int iqs, float2& v) {
    v = dequant2_generic<block_q2_K>((const block_q2_K*)vx, ib, iqs);
}
__device__ void dequantize_q3_K(const void* vx, int64_t ib, int iqs, float2& v) {
    v = dequant2_generic<block_q3_K>((const block_q3_K*)vx, ib, iqs);
}
__device__ void dequantize_q4_K(const void* vx, int64_t ib, int iqs, float2& v) {
    v = dequant2_generic<block_q4_K>((const block_q4_K*)vx, ib, iqs);
}
__device__ void dequantize_q5_K(const void* vx, int64_t ib, int iqs, float2& v) {
    v = dequant2_generic<block_q5_K>((const block_q5_K*)vx, ib, iqs);
}
__device__ void dequantize_q6_K(const void* vx, int64_t ib, int iqs, float2& v) {
    v = dequant2_generic<block_q6_K>((const block_q6_K*)vx, ib, iqs);
}

// ---------- Basic quant types Q1_0, Q2_0 ----------
template<>
static __device__ float2 dequant2_generic<block_q1_0>(const block_q1_0 *x, int64_t ib, int iqs) {
#if QK_K == 256
    const float d = __half2float(x[ib].d);
    const int bit = iqs;
    const int byte = bit / 8;
    const int offset = bit % 8;
    const int bit_val = (x[ib].qs[byte] >> offset) & 1;
    const float val = (bit_val ? 1.0f : -1.0f) * d;
    // q1_0 packs 32 values per block, each value 1 bit. For float2 we need two consecutive.
    // We'll just return same for both (not used in pair dequant). Return val for both.
    return make_float2(val, val);
#else
    return make_float2(0.0f, 0.0f);
#endif
}

template<>
static __device__ float2 dequant2_generic<block_q2_0>(const block_q2_0 *x, int64_t ib, int iqs) {
#if QK_K == 256
    const float d = __half2float(x[ib].d);
    const uint8_t q = x[ib].qs[iqs];
    const float v0 = ((q & 0x3) - 1.0f) * d; // placeholder
    const float v1 = ((q >> 2) - 1.0f) * d;
    return make_float2(v0, v1);
#else
    return make_float2(0.0f, 0.0f);
#endif
}

// ---------- Basic quant types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) ----------
template<>
static __device__ float2 dequant2_generic<block_q4_0>(const block_q4_0 *x, int64_t ib, int iqs) {
#if QK_K == 256
    const float d = __half2float(x[ib].d);
    const uint8_t q = x[ib].qs[iqs];
    const float scale = d;
    const float min = -8.0f * d;
    const float v0 = (q & 0xF) * scale + min;
    const float v1 = (q >> 4) * scale + min;
    return make_float2(v0, v1);
#else
    return make_float2(0.0f, 0.0f);
#endif
}

template<>
static __device__ float2 dequant2_generic<block_q4_1>(const block_q4_1 *x, int64_t ib, int iqs) {
#if QK_K == 256
    const float2 dm = __half22float2(x[ib].dm);
    const uint8_t q = x[ib].qs[iqs];
    const float v0 = (q & 0xF) * dm.x + dm.y;
    const float v1 = (q >> 4) * dm.x + dm.y;
    return make_float2(v0, v1);
#else
    return make_float2(0.0f, 0.0f);
#endif
}

template<>
static __device__ float2 dequant2_generic<block_q5_0>(const block_q5_0 *x, int64_t ib, int iqs) {
#if QK_K == 256
    const float d = __half2float(x[ib].d);
    uint32_t qh; memcpy(&qh, x[ib].qh, sizeof(qh));
    const uint8_t q = x[ib].qs[iqs];
    const int shift = iqs * 2;
    const uint8_t q0 = (q & 0xF) | ((qh >> shift) & 0x10);
    const uint8_t q1 = ((q >> 4) & 0xF) | ((qh >> (shift + 2)) & 0x10);
    const float v0 = (q0 - 16) * d;
    const float v1 = (q1 - 16) * d;
    return make_float2(v0, v1);
#else
    return make_float2(0.0f, 0.0f);
#endif
}

template<>
static __device__ float2 dequant2_generic<block_q5_1>(const block_q5_1 *x, int64_t ib, int iqs) {
#if QK_K == 256
    const float2 dm = __half22float2(x[ib].dm);
    uint32_t qh; memcpy(&qh, x[ib].qh, sizeof(qh));
    const uint8_t q = x[ib].qs[iqs];
    const int shift = iqs * 2;
    const uint8_t q0 = (q & 0xF) | ((qh >> shift) & 0x10);
    const uint8_t q1 = ((q >> 4) & 0xF) | ((qh >> (shift + 2)) & 0x10);
    const float v0 = q0 * dm.x + dm.y;
    const float v1 = q1 * dm.x + dm.y;
    return make_float2(v0, v1);
#else
    return make_float2(0.0f, 0.0f);
#endif
}

template<>
static __device__ float2 dequant2_generic<block_q8_0>(const block_q8_0 *x, int64_t ib, int iqs) {
#if QK_K == 256
    const float d = __half2float(x[ib].d);
    const float v0 = x[ib].qs[iqs] * d;
    const float v1 = x[ib].qs[iqs + 1] * d;
    return make_float2(v0, v1);
#else
    return make_float2(0.0f, 0.0f);
#endif
}

// ----- Public API required by callers -----
// 1) out-pointer + iqs  (used by get_rows)
// 2) float2 &v           (used by block-dequant kernels)

#define DEQUANT_WRAPPERS(Block) \
template<typename dst_t> \
static __device__ void dequantize_##Block(const void *vx, int64_t ib, dst_t *out, int iqs) { \
    const float2 v = dequant2_generic<Block>((const Block *)vx, ib, iqs); \
    out[0] = static_cast<dst_t>(v.x); \
    out[1] = static_cast<dst_t>(v.y); \
} \
template<typename dst_t> \
static __device__ void dequantize_##Block(const void *vx, int64_t ib, int iqs, float2 &v) { \
    v = dequant2_generic<Block>((const Block *)vx, ib, iqs); \
}

// Generate wrappers for k-quant and IQ types (need both signatures)
DEQUANT_WRAPPERS(block_q1_0)
DEQUANT_WRAPPERS(block_q2_0)
DEQUANT_WRAPPERS(block_q4_0)
DEQUANT_WRAPPERS(block_q4_1)
DEQUANT_WRAPPERS(block_q5_0)
DEQUANT_WRAPPERS(block_q5_1)
DEQUANT_WRAPPERS(block_q8_0)
DEQUANT_WRAPPERS(block_rocmfp4)
DEQUANT_WRAPPERS(block_rocmfp4_fast)
DEQUANT_WRAPPERS(block_mxfp4)

#undef DEQUANT_WRAPPERS
// Aliases for legacy names used by copy kernels
#define ALIAS_DEQUANT(Name, Block) \
template<typename dst_t> \
static __device__ void dequantize_##Name(const void *vx, int64_t ib, dst_t *out, int iqs) { \
    dequantize_##Block(vx, ib, out, iqs); \
} \
template<typename dst_t> \
static __device__ void dequantize_##Name(const void *vx, int64_t ib, int iqs, float2 &v) { \
    dequantize_##Block(vx, ib, iqs, v); \
}

// ROCm FP4 non‑template overloads already provided above
#undef ALIAS_DEQUANT

// Block dequant template functions for basic quant types (used by copy kernels) are generated by DEQUANT_WRAPPERS macro
// Definitions (non-template, external linkage) for basic quant types and ROCm FP4
__device__ void dequantize_q1_0(const void* vx, int64_t ib, int iqs, float2& v) {
    v = dequant2_generic<block_q1_0>((const block_q1_0*)vx, ib, iqs);
}
__device__ void dequantize_q2_0(const void* vx, int64_t ib, int iqs, float2& v) {
    v = dequant2_generic<block_q2_0>((const block_q2_0*)vx, ib, iqs);
}
__device__ void dequantize_q4_0(const void* vx, int64_t ib, int iqs, float2& v) {
    v = dequant2_generic<block_q4_0>((const block_q4_0*)vx, ib, iqs);
}
__device__ void dequantize_q4_1(const void* vx, int64_t ib, int iqs, float2& v) {
    v = dequant2_generic<block_q4_1>((const block_q4_1*)vx, ib, iqs);
}
__device__ void dequantize_q5_0(const void* vx, int64_t ib, int iqs, float2& v) {
    v = dequant2_generic<block_q5_0>((const block_q5_0*)vx, ib, iqs);
}
__device__ void dequantize_q5_1(const void* vx, int64_t ib, int iqs, float2& v) {
    v = dequant2_generic<block_q5_1>((const block_q5_1*)vx, ib, iqs);
}
__device__ void dequantize_q8_0(const void* vx, int64_t ib, int iqs, float2& v) {
    v = dequant2_generic<block_q8_0>((const block_q8_0*)vx, ib, iqs);
}
__device__ void dequantize_rocmfp4(const void* vx, int64_t ib, int iqs, float2& v) {
    v = dequant2_generic<block_rocmfp4>((const block_rocmfp4*)vx, ib, iqs);
}
__device__ void dequantize_rocmfp4_fast(const void* vx, int64_t ib, int iqs, float2& v) {
    v = dequant2_generic<block_rocmfp4_fast>((const block_rocmfp4_fast*)vx, ib, iqs);
}

#endif // GGML_USE_HIP && GGML_USE_ROCMFP4