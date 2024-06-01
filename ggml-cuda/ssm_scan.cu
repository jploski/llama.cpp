#include "ssm_scan.cuh"

template <int block_size>
static __global__ void ssm_scan_f32(
    const float * src0, const float * src1, const float * src2, const float * src3,
    const float * src4, const float * src5, const float * src6,
    const int src0_nb1, const int src0_nb2,
    const int src1_nb0, const int src1_nb1, const int src1_nb2,
    const int src2_nb0, const int src2_nb1,
    const int src3_nb1,
    const int src4_nb1,
    const int src5_nb1,
    const int src6_nb1,
    float * dst,
    const int nc, const int nr, const int n_t, const int n_kv) {

//    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    const int ith = tid;
    const int nth = WARP_SIZE;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = min(ir0 + dr, nr);
    const int ir  = ir1 - ir0;

    if (n_kv > 1) {
        // it's hard to know if the source states have already been copied
        // when there are multiple, so copy them already.
        for (int i3 = 0; i3 < n_kv; ++i3) {
            float * s0 = (float *) ((char *) src0 + ir0*src0_nb1 + i3*src0_nb2);
            float * s  = (float *) ((char *)  dst + ir0*src0_nb1 + i3*src0_nb2 + src1_nb2);

            //memcpy(s, s0, nc*ir*sizeof(float));
            for (int i4 = 0; i4 < nc*ir; i4++) {
                s[i4] = s0[i4];
            }
        }
    }

    for (int i2 = 0; i2 < n_t; ++i2) {
        int32_t * sq = (int32_t *) ((char *) src6 +  i2*src6_nb1); // {n_kv, n_tokens}
        float *   y  = (float *)   ((char *)  dst + ir0*src1_nb0 +    i2*src1_nb1); // {d_inner, n_tokens}
        float *   s  = (float *)   ((char *)  dst + ir0*src0_nb1 + sq[0]*src0_nb2 + src1_nb2); // {d_state, d_inner, n_kv}
        float *   s0;
        float *   x  = (float *)   ((char *) src1 + ir0*src1_nb0 + i2*src1_nb1); // {d_inner, n_tokens}
        float *   dt = (float *)   ((char *) src2 + ir0*src2_nb0 + i2*src2_nb1); // {d_inner, n_tokens}
        float *   A  = (float *)   ((char *) src3 + ir0*src3_nb1); // {d_state, d_inner}
        float *   B  = (float *)   ((char *) src4 +  i2*src4_nb1); // {d_state, n_tokens}
        float *   C  = (float *)   ((char *) src5 +  i2*src5_nb1); // {d_state, n_tokens}

        // avoid needing to copy the state for the first token
        if (i2 == 0) {
            s0 = (float *) ((char *) src0 + ir0*(src0_nb1) + sq[0]*src0_nb2); // {d_state, d_inner, n_kv}
        } else {
            // otherwise the source is the same as the destination
            s0 = s;
        }

        // d_inner
        for (int i1 = 0; i1 < ir; ++i1) {
            // ref: https://github.com/state-spaces/mamba/blob/34076d664838588a3c97727b263478ab9f621a07/mamba_ssm/ops/triton/selective_state_update.py#L78
            float dt_soft_plus = dt[i1] <= 20.0f ? log1pf(expf(dt[i1])) : dt[i1];
            float x_dt = x[i1] * dt_soft_plus;
            float sumf = 0.0f;
            // d_state
            for (int i0 = 0; i0 < nc; ++i0) {
                int i = i0 + i1*nc;
                // state = prev_state * dA + dB * x
                float state = (s0[i] * expf(dt_soft_plus * A[i])) + (B[i0] * x_dt);
                // y = rowwise_dotprod(state, C)
                sumf += state * C[i0];
                s[i] = state;
            }
            y[i1] = sumf;
        }

        // handle copies when there are multiple output states
        for (int i3 = 1; i3 < n_kv; ++i3) {
            int32_t seq = sq[i3];
            if (0 <= seq && seq < n_kv) {
                float * s1 = s + (seq - sq[0])*nc*nr;
                //memcpy(s1, s, nc*ir*sizeof(float));
                for (int i4 = 0; i4 < nc*ir; i4++) {
                    s1[i4] = s[i4];
                }
            } else {
                // stop at negative or too big seq_ids
                break;
            }
        }
    }
}

static void ssm_scan_f32_cuda(
    const float * src0, const float * src1, const float * src2, const float * src3,
    const float * src4, const float * src5, const float * src6,
    const int src0_nb1, const int src0_nb2,
    const int src1_nb0, const int src1_nb1, const int src1_nb2,
    const int src2_nb0, const int src2_nb1,
    const int src3_nb1,
    const int src4_nb1,
    const int src5_nb1,
    const int src6_nb1,
    float * dst,
    const int nc, const int nr, const int n_t, const int n_kv, cudaStream_t stream) {

    const dim3 block_dims(WARP_SIZE, 1, 1);
    const int nblocks = 1; // TODO

    ssm_scan_f32<WARP_SIZE><<<nblocks, block_dims, 0, stream>>>(
        src0, src1, src2, src3, src4, src5, src6,
        src0_nb1, src0_nb2,
        src1_nb0, src1_nb1, src1_nb2,
        src2_nb0, src2_nb1,
        src3_nb1,
        src4_nb1,
        src5_nb1,
        src6_nb1,
        dst,
        nc, nr, n_t, n_kv);
}

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // s
    const struct ggml_tensor * src1 = dst->src[1]; // x
    const struct ggml_tensor * src2 = dst->src[2]; // dt
    const struct ggml_tensor * src3 = dst->src[3]; // A
    const struct ggml_tensor * src4 = dst->src[4]; // B
    const struct ggml_tensor * src5 = dst->src[5]; // C
    const struct ggml_tensor * src6 = dst->src[6]; // sq

    const int64_t nc   = src0->ne[0]; // d_state
    const int64_t nr   = src0->ne[1]; // d_inner
    const int64_t n_t  = src1->ne[1]; // number of tokens in the batch
    const int64_t n_kv = src0->ne[2]; // max number of sequences in the batch

    GGML_ASSERT(ggml_nelements(src1) + ggml_nelements(src0) == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(float));
    GGML_ASSERT(src4->nb[0] == sizeof(float));
    GGML_ASSERT(src5->nb[0] == sizeof(float));
    // required for the dot product between s and C, and when copying the states
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*sizeof(float));
    // required for per-sequence offsets for states
    GGML_ASSERT(src0->nb[2] == src0->ne[0]*src0->ne[1]*sizeof(float));
    // required to get correct offset for state destination (i.e. src1->nb[2])
    GGML_ASSERT(src1->nb[2] == src1->ne[0]*src1->ne[1]*sizeof(float));

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    const float * src2_d = (const float *)src2->data;
    const float * src3_d = (const float *)src3->data;
    const float * src4_d = (const float *)src4->data;
    const float * src5_d = (const float *)src5->data;
    const float * src6_d = (const float *)src6->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ssm_scan_f32_cuda(
        src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d,
        src0->nb[1], src0->nb[2],
        src1->nb[0], src1->nb[1], src1->nb[2],
        src2->nb[0], src2->nb[1],
        src3->nb[1],
        src4->nb[1],
        src5->nb[1],
        src6->nb[1],
        dst_d,
        nc, nr, n_t, n_kv, stream);
}
