#pragma once

#include <cstddef>
#include <cstdint>
#include "../include/kimodo/result.hpp"
#include <span>
#include <string>
#include <vector>

namespace kimodo::detail {
class ggml_motion_weights;

// One sequence segment with an already-encoded text condition and caller-
// supplied initial noise. Continuation noise includes its transition prefix.
struct sampled_sequence_segment {
    std::span<const float> embedding;
    std::span<const float> initial_noise;
    std::size_t frames;
};

struct sequence_transition {
    std::vector<float> observed;
    std::vector<float> observed_mask;
    float first_heading = 0.F;
    float origin_x = 0.F;
    float origin_z = 0.F;
};

// F32 TransformerEncoderBlock. Inputs/outputs are row-major [B,T,D], while
// the implementation creates GGML [D,T,B] views over the same byte order.
kimodo::expected<std::vector<float>, std::string> run_motion_transformer(
    const ggml_motion_weights &weights, std::string_view prefix,
    std::span<const float> motion, std::size_t motion_dim,
    std::span<const float> text_embedding, std::span<const float> timesteps,
    std::span<const float> headings, std::size_t batch, std::size_t frames);

// Exact two-stage Kimodo denoiser for concatenated motion/mask inputs
// [B,T,2*motion_dim]. Returned clean prediction is [B,T,motion_dim].
kimodo::expected<std::vector<float>, std::string> run_two_stage_denoiser(
    const ggml_motion_weights &weights, std::span<const float> motion_and_mask,
    std::span<const float> text_embedding, std::span<const float> timesteps,
    std::span<const float> headings, std::span<const float> motion_mask,
    std::size_t batch, std::size_t frames);

// Unconstrained separated CFG wrapper. `motion` is [T,motion_dim], embedding
// is [4096], and the result is one clean prediction.
kimodo::expected<std::vector<float>, std::string> run_separated_cfg_denoiser(
    const ggml_motion_weights &weights, std::span<const float> motion,
    std::span<const float> embedding, float timestep, float text_weight,
    float constraint_weight, std::size_t frames);

// The same for `prompts` prompts at once, the way a server batches its
// sequences: `motion` is [prompts,T,motion_dim], `embedding` [prompts,4096],
// `timesteps` [prompts]. Prompt p's three CFG branches are batch entries
// 3p..3p+2 of ONE Transformer pass; every operation is per batch entry, so
// prompt p's result is the one it gets alone. Returns [prompts,T,motion_dim].
kimodo::expected<std::vector<float>, std::string> run_separated_cfg_denoiser_batch(
    const ggml_motion_weights &weights, std::size_t prompts, std::span<const float> motion,
    std::span<const float> embedding, std::span<const float> timesteps, float text_weight,
    float constraint_weight, std::size_t frames);

// The conditioned form of the batch above: `observed` and `observed_mask` are
// [prompts,T,motion_dim], `headings` [prompts].
kimodo::expected<std::vector<float>, std::string> run_separated_cfg_denoiser_conditioned_batch(
    const ggml_motion_weights &weights, std::size_t prompts, std::span<const float> motion,
    std::span<const float> embedding, std::span<const float> observed,
    std::span<const float> observed_mask, std::span<const float> timesteps,
    std::span<const float> headings, float text_weight, float constraint_weight, std::size_t frames);

// Each Transformer runs as one graph that is built and allocated once for its
// (batch, frames) and kept for the steps and jobs after it. Release frees
// those graphs' device buffers; graph_bytes is what they hold now.
void release_denoiser_graphs();
std::size_t denoiser_graph_bytes();

// Measurement only: host-clock time of the Transformer passes, summed since
// the caller last cleared it. clock_us null (the default) reads no clock;
// after_graph runs after each pass's result is back (its GPU work synced).
struct denoiser_profile {
    int64_t (*clock_us)() = nullptr;
    void (*after_graph)(void *user) = nullptr;
    void *user = nullptr;
    int64_t builds = 0, graphs = 0, submits = 0;
    // build: graph made and allocated; upload / download: the inputs and the
    // result; compute: graph_compute (records and submits); total: all of
    // run_motion_transformer.
    int64_t build_us = 0, upload_us = 0, compute_us = 0, download_us = 0, total_us = 0;
    int64_t upload_bytes = 0, download_bytes = 0;
};
denoiser_profile &denoiser_profile_state();

// How a Transformer pass is cut into submits (graph computes over the one
// allocation, so `state` stays on the device between them). The default,
// clock_us null, is one submit a pass. A host whose driver stops waiting for
// a long submit (Godot's Metal RenderingDevice waits 2 s for a fence and then
// reads on, so a longer submit returns garbage) sets a clock and a budget:
// the first pass then submits one layer at a time, and each later pass cuts
// at layer boundaries so that a submit takes at most `budget_us` at the last
// pass's measured rate. The cut changes no value: the same nodes run in the
// same order on the same bytes.
struct denoiser_submit_policy {
    int64_t (*clock_us)() = nullptr;
    int64_t budget_us = 0;
};
denoiser_submit_policy &denoiser_submit_policy_state();

// Deterministic eta=0 DDIM sampling from caller-supplied F32 initial noise.
kimodo::expected<std::vector<float>, std::string> sample_motion_from_noise(
    const ggml_motion_weights &weights, std::span<const float> initial_noise,
    std::span<const float> embedding, std::size_t frames, unsigned steps,
    float text_weight, float constraint_weight);

// Multi-prompt transition sampler. `observed` and `observed_mask` are [T,motion_dim]
// normalized motion-representation values/masks. This mirrors the upstream
// concat-mask denoiser: text, constraint, and unconditional CFG branches.
kimodo::expected<std::vector<float>, std::string> sample_motion_from_noise_conditioned(
    const ggml_motion_weights &weights, std::span<const float> initial_noise,
    std::span<const float> embedding, std::span<const float> observed,
    std::span<const float> observed_mask, float first_heading, std::size_t frames,
    unsigned steps, float text_weight, float constraint_weight);

// End-to-end upstream `_multiprompt` orchestration.  DDIM operates in
// normalized motion space; the returned joined representation is raw so its
// translated roots and blended tail preserve upstream semantics.
kimodo::expected<std::vector<float>, std::string> sample_motion_sequence_from_noise(
    const ggml_motion_weights &weights, std::span<const sampled_sequence_segment> segments,
    unsigned transition_frames, unsigned steps, float text_weight, float constraint_weight);

// Build the exact condition consumed by the next `_multiprompt` DDIM run.
// Exposed for the raw fixture test as well as the runtime orchestrator.
kimodo::expected<sequence_transition, std::string> prepare_sequence_transition(
    const ggml_motion_weights &weights, std::span<const float> previous,
    std::size_t continuation_frames, unsigned transition_frames);
}
