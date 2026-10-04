// motion.elf (RFD 2277 A2): our own motion for the "does the dress clip in
// motion" fit, generated in the sandbox on ggml-rd (the VRChat SDK's
// proxy_*.anim clips are blocklisted and never read).
//
// Two models, both run as their own ports' graph code on ggml-rd:
//   kimodo_soma      Kimodo-SOMA-RP-v1.1, text to motion on SOMA (the 30
//                    joints the model predicts), vendor/kimodo-ggml's
//                    two-stage denoiser and DDIM sampler
//   motionbricks_g1  MotionBricks-G1, style-driven, vendor/motion-bricks-ggml's
//                    agent, planner, pose and VQ graphs, driven by the loop of
//                    character-fox Tools/rig/motionbricks_motion.pose_tracks
// Weights never enter the guest heap: the host streams each tensor into the
// RenderingDevice buffer (the pump's UPLOAD, rfdetr_seg.elf's pattern).
//
// STAND-IN, printed in every result: Kimodo's text encoder (the LLM2Vec
// Llama-3-8B bundle, 15 GB) is not in the guest. The host embeds each prompt
// natively once (tools/motion_native `embed`, kimodo-ggml's own encoder) and
// passes the 4096 floats in with motion_prompt_embedding(); a Kimodo job
// whose prompt has none fails and says so.
//
// Host side (project/stages/motion_stage.gd, project/gate_motion.gd):
//   motion_open(rd, total_mb, models_dir)   once
//   motion_env("GGML_RD_FAULT=1 KIMODO_STEPS=100 MOTION_GAS=2e7")
//                                           switches for later jobs (MOTION_GAS: the
//                                           instruction budget of a slice)
//   motion_prompt_embedding(prompt, floats) the stand-in's job data
//   motion_generate(model, prompt, seconds, seed) -> "QUEUED <id>"; the job
//                                           queue runs on the pump
//   motion_pump(data)                       one slice a frame until DONE / ERROR (rule 4)
//   motion_progress()                       the queue now (model, prompt, step i of n)
//   motion_status(id), motion_result(id), motion_raw(id)
//   motion_avatar(names, parents, rest_global, rest_local, roles)
//                                           the avatar's bones, exported by the host
//   motion_retarget(id)                     the clip on the avatar + its clip ranges
//                                           (made by the job, eight frames a slice)
//   motion_live_start(seed)                 MotionBricks steered live: the job plans a few
//   motion_live_steer(mx, mz, fx, fz, speed)  frames ahead of the host at a time, toward the
//   motion_live_frames()                    latest steer, and hands back world joint
//   motion_live_skeleton(), motion_live_stop() positions
#include <api.hpp>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "denoiser.hpp"
#include "diffusion.hpp"
#include "ggml-backend.h"
#include "ggml-rd.h"
#include "ggml.h"
#include "ggml_weights.hpp"
#include "kimodo_decode.h"
#include "motion_io.h"
#include "noise.h"
#include "motionbricks/motionbricks.h"
#include "pump/pump.h"
#include "rd_compute.h"
#include "retarget.h"
#include "skeleton.hpp"

namespace {

rdc::Device g_dev;
bool g_attached = false;
std::string g_models;
std::string g_log;

void logf_(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void logf_(const char *fmt, ...) {
	char buf[1024];
	va_list ap;
	va_start(ap, fmt);
	std::vsnprintf(buf, sizeof buf, fmt, ap);
	va_end(ap);
	g_log += buf;
	g_log += '\n';
	std::printf("motion: %s\n", buf);
}

// --- gas and progress -------------------------------------------------------------------
// Every job is gassable and resumable (the operator's rule for godot-sandbox
// work): a vmcall advances it by a bounded amount and then gives the frame
// back, and its state stays in the guest (the job's fiber) for the next
// tick. The natural slices are one ggml graph (each compute yields WAIT_GPU
// and syncs on a later frame), one diffusion step, one MotionBricks plan and
// eight frames of the retarget; MOTION_GAS=<instructions> adds a budget: at
// every slice boundary past it the job yields COOP. libriscv counts a
// vmcall's instructions from its start (Gate 6.P), so rdinstret is the
// budget's meter. Any budget gives the same bytes (gates/10-motion).
uint64_t g_gas = 0;
int64_t g_gas_yields = 0;
uint64_t instret() {
	uint64_t v;
	asm volatile("rdinstret %0" : "=r"(v));
	return v;
}
void gas() {
	if (g_gas != 0 && instret() >= g_gas) {
		++g_gas_yields;
		pump::coop();
	}
}
struct Progress {
	int job = -1;
	int model = -1;
	std::string phase = "idle";
	int64_t i = 0, n = 0;
};
Progress g_prog;

// --- ggml-rd hooks (rfdetr_seg.elf's four) ---------------------------------------------
void hook_wait_gpu(void *) {
	pump::wait_gpu();
}
void hook_coop(void *) {
	pump::coop();
}
bool hook_upload(void *, const std::string &path, uint64_t file_offset, uint64_t bytes, ::RID rid, uint64_t dst_offset) {
	return pump::upload(path, file_offset, bytes, rid, dst_offset);
}
bool hook_read(void *, const std::string &path, uint64_t file_offset, uint64_t bytes, void *dst) {
	const uint64_t chunk = uint64_t(16) << 20;
	for (uint64_t done = 0; done < bytes;) {
		const uint64_t n = std::min(bytes - done, chunk);
		std::vector<uint8_t> v = pump::read(path, file_offset + done, n);
		if (v.size() != n) {
			return false;
		}
		std::memcpy(static_cast<uint8_t *>(dst) + done, v.data(), size_t(n));
		done += n;
	}
	return true;
}
void on_ggml_abort(const char *message) {
	std::fflush(stdout);
	pump::fail(std::string("ggml abort: ") + message);
}

std::vector<std::string> split(const std::string &s, char sep) {
	std::vector<std::string> r;
	size_t i = 0;
	while (i <= s.size()) {
		const size_t j = std::min(s.find(sep, i), s.size());
		if (j > i) {
			r.push_back(s.substr(i, j - i));
		}
		i = j + 1;
	}
	return r;
}

// --- models ------------------------------------------------------------------------------
enum Model : int { KIMODO = 0, MBRICKS = 1 };
const char *kModelNames[] = { "kimodo_soma", "motionbricks_g1" };
constexpr float kFps = 30.f;

std::unique_ptr<kimodo::detail::ggml_motion_weights> g_kimodo;
std::vector<float> g_k_gm, g_k_gs, g_k_bm, g_k_bs; // Kimodo's normalisation stats
mb_model *g_mb = nullptr;
mb_runtime_options *g_mb_opts = nullptr;

std::string kimodo_path() {
	return g_models + "/Kimodo-SOMA-RP-v1.1-gguf/kimodo-soma-rp-v1.1.gguf";
}
std::string mb_bundle() {
	return g_models + "/MotionBricks-G1-GGML/g1-f32";
}
std::string mb_style_path(const std::string &style) {
	return g_models + "/MotionBricks-G1-GGML/styles/" + style + ".mbstyle";
}

// The stand-in's job data: prompt -> 4096 floats.
std::vector<std::string> g_emb_prompts;
std::vector<float> g_emb;

// --- clips: flat records, arrays in one pool (AGENTS.md: keep long-lived guest data flat)
struct Clip {
	int id = 0;
	int model = 0;
	int state = 0; // 0 queued, 1 running, 2 done, 3 error
	float seconds = 0.f;
	int64_t seed = 0;
	int frames = 0;
	int joints = 0;
	int steps = 0;
	size_t root_off = 0;  // frames x 3
	size_t local_off = 0; // frames x joints x 9
	size_t raw_off = 0, raw_n = 0; // Kimodo: the sampler's output, frames x D
	size_t rest_off = 0;  // joints x 3 rest offsets from the parent
	size_t parent_off = 0; // into g_ipool
	size_t names_off = 0;  // into g_names (joints entries)
	int retargeted = 0;    // 1: rt_* hold the clip on the avatar
	size_t rt_off = 0;     // frames x ROLE_COUNT x 12
	size_t rt_deg_off = 0; // frames x ROLE_COUNT, the avatar's angles (degrees)
};
std::vector<Clip> g_clips;
std::vector<std::string> g_clip_prompt, g_clip_error, g_clip_style, g_clip_ranges, g_clip_axes;
std::vector<float> g_pool;
std::vector<int> g_ipool;
std::vector<std::string> g_names;
// The job queue: (kind, clip). GENERATE runs the model and then, when the
// avatar is known, the retarget; RETARGET only the retarget.
enum Kind : int { GENERATE = 0, RETARGET = 1, LIVE = 2 };
std::deque<std::pair<int, int>> g_queue;

size_t push_floats(const float *p, size_t n) {
	const size_t at = g_pool.size();
	g_pool.insert(g_pool.end(), p, p + n);
	return at;
}

const char *state_name(int s) {
	static const char *k[] = { "QUEUED", "RUNNING", "DONE", "ERROR" };
	return s >= 0 && s <= 3 ? k[s] : "?";
}

// --- Kimodo ------------------------------------------------------------------------------
bool kimodo_load(std::string &err) {
	if (g_kimodo) {
		return true;
	}
	auto w = kimodo::detail::ggml_motion_weights::load(kimodo_path());
	if (!w) {
		err = w.error();
		return false;
	}
	g_kimodo = std::move(*w);
	for (auto [name, dst] : { std::pair{ "stats.global_root.mean", &g_k_gm }, std::pair{ "stats.global_root.std", &g_k_gs },
				 std::pair{ "stats.body.mean", &g_k_bm }, std::pair{ "stats.body.std", &g_k_bs } }) {
		auto v = g_kimodo->f32_values(name);
		if (!v) {
			err = v.error();
			return false;
		}
		*dst = std::move(*v);
	}
	logf_("kimodo: %s loaded, skeleton %s, motion_dim %zu", kimodo_path().c_str(),
			std::string(g_kimodo->skeleton_key()).c_str(), g_kimodo->motion_dim());
	return true;
}

int env_int(const char *k, int dflt) {
	const char *v = std::getenv(k);
	return v && *v ? std::atoi(v) : dflt;
}

bool run_kimodo(Clip &c, std::string &err) {
	const std::string &prompt = g_clip_prompt[size_t(c.id)];
	size_t e = 0;
	for (; e < g_emb_prompts.size() && g_emb_prompts[e] != prompt; ++e) {
	}
	if (e == g_emb_prompts.size()) {
		err = "no embedding for prompt '" + prompt +
				"' (stand-in: the Llama-3 text encoder is not in the guest; the host passes each prompt's "
				"native embedding with motion_prompt_embedding)";
		return false;
	}
	g_prog.phase = "streaming Kimodo weights";
	if (!kimodo_load(err)) {
		return false;
	}
	const auto *skel = kimodo::detail::find_skeleton(g_kimodo->skeleton_key());
	const size_t D = g_kimodo->motion_dim(), J = skel->joints();
	const size_t frames = size_t(std::lround(double(c.seconds) * kFps));
	c.steps = env_int("KIMODO_STEPS", 100);
	// The seed's noise, the same on every platform (noise.h says why it is
	// not upstream's std::normal_distribution).
	std::vector<float> noise(frames * D);
	motion::normal_noise(uint64_t(c.seed), noise.data(), noise.size());
	logf_("kimodo job %d: '%s' %zu frames, %d steps, seed %lld, noise[0..2] %.9g %.9g %.9g", c.id, prompt.c_str(),
			frames, c.steps, (long long)c.seed, noise[0], noise[1], noise[2]);
	std::span<const float> emb(g_emb.data() + e * 4096, 4096);
	// kimodo::detail::sample_motion_from_noise, call for call (the same cosine
	// schedule, separated-CFG denoiser and DDIM step in the same order), with
	// a progress line and a slice boundary after every step.
	auto schedule = kimodo::detail::make_cosine_schedule(1000, unsigned(c.steps));
	if (!schedule) {
		err = schedule.error();
		return false;
	}
	std::vector<float> state(noise.begin(), noise.end()), next(state.size());
	g_prog.phase = "diffusion step";
	g_prog.n = c.steps;
	for (unsigned i = unsigned(c.steps); i-- > 0;) {
		g_prog.i = c.steps - int64_t(i);
		auto clean = kimodo::detail::run_separated_cfg_denoiser(*g_kimodo, state, emb, float(schedule->use_timesteps[i]),
				2.f, 2.f, frames);
		if (!clean) {
			err = clean.error();
			return false;
		}
		auto stepped = kimodo::detail::ddim_step(*schedule, i, state.data(), clean->data(), next.data(), state.size());
		if (!stepped) {
			err = stepped.error();
			return false;
		}
		state.swap(next);
		gas();
	}
	const std::vector<float> *sampled = &state;
	std::vector<float> root, local;
	motion::kimodo_decode::decode(sampled->data(), frames, J, skel->parents.data(), g_k_gm.data(), g_k_gs.data(),
			g_k_bm.data(), g_k_bs.data(), root, local);
	c.frames = int(frames);
	c.joints = int(J);
	c.raw_n = sampled->size();
	c.raw_off = push_floats(sampled->data(), sampled->size());
	c.root_off = push_floats(root.data(), root.size());
	c.local_off = push_floats(local.data(), local.size());
	std::vector<float> rest(J * 3);
	for (size_t j = 0; j < J; ++j) {
		for (int k = 0; k < 3; ++k) {
			rest[j * 3 + size_t(k)] = skel->offsets[j][size_t(k)];
		}
	}
	c.rest_off = push_floats(rest.data(), rest.size());
	c.parent_off = g_ipool.size();
	c.names_off = g_names.size();
	for (size_t j = 0; j < J; ++j) {
		g_ipool.push_back(skel->parents[j]);
		g_names.emplace_back(skel->names[j]);
	}
	return true;
}

// --- MotionBricks ------------------------------------------------------------------------
bool mb_ok(mb_status s, const char *what, const char *errbuf, std::string &err) {
	if (s == MB_OK) {
		return true;
	}
	err = std::string(what) + ": " + mb_status_string(s) + " -- " + errbuf;
	return false;
}

// The style for a prompt word: MotionBricks is not text-conditioned; its
// released styles are named clips (styles/manifest.json).
std::string mb_style_for(const std::string &prompt) {
	std::string p = prompt;
	std::transform(p.begin(), p.end(), p.begin(), [](unsigned char ch) { return char(std::tolower(ch)); });
	if (p.find("danc") != std::string::npos) return "walk_happy_dance";
	if (p.find("idle") != std::string::npos || p.find("stand") != std::string::npos) return "idle";
	if (p.find("walk") != std::string::npos) return "walk";
	return "";
}

// The boundary with the vendored library's public output, which is XYZW:
// converted once to the row-major matrix every other line of this ELF uses
// (rule 11).
void xyzw_matrix(const float *q, float *m) {
	const float x = q[0], y = q[1], z = q[2], w = q[3];
	m[0] = 1 - 2 * (y * y + z * z);
	m[1] = 2 * (x * y - z * w);
	m[2] = 2 * (x * z + y * w);
	m[3] = 2 * (x * y + z * w);
	m[4] = 1 - 2 * (x * x + z * z);
	m[5] = 2 * (y * z - x * w);
	m[6] = 2 * (x * z - y * w);
	m[7] = 2 * (y * z + x * w);
	m[8] = 1 - 2 * (x * x + y * y);
}

bool mb_load(std::string &err) {
	char eb[512] = { 0 };
	g_prog.phase = "streaming MotionBricks weights";
	if (g_mb == nullptr) {
		if (!mb_ok(mb_runtime_options_create(&g_mb_opts, eb, sizeof eb), "options", eb, err)) return false;
		if (!mb_ok(mb_model_load(mb_bundle().c_str(), g_mb_opts, &g_mb, eb, sizeof eb), "model_load", eb, err)) return false;
		logf_("motionbricks: %s loaded", mb_bundle().c_str());
	}
	return true;
}

bool run_mbricks(Clip &c, std::string &err) {
	char eb[512] = { 0 };
	const std::string style = mb_style_for(g_clip_prompt[size_t(c.id)]);
	g_clip_style[size_t(c.id)] = style;
	if (style.empty()) {
		err = "MotionBricks has no style for '" + g_clip_prompt[size_t(c.id)] +
				"' (its released styles are walk/idle/dance-like clips, not text)";
		return false;
	}
	if (!mb_load(err)) return false;
	mb_style *st = nullptr;
	mb_agent *agent = nullptr;
	mb_command *cmd = nullptr;
	auto cleanup = [&] {
		if (cmd) mb_command_free(cmd);
		if (agent) mb_agent_free(agent);
		if (st) mb_style_free(st);
	};
	if (!mb_ok(mb_style_load(g_mb, mb_style_path(style).c_str(), &st, eb, sizeof eb), "style_load", eb, err) ||
			!mb_ok(mb_agent_create(g_mb, &agent, eb, sizeof eb), "agent", eb, err) ||
			!mb_ok(mb_agent_reset(agent, st, eb, sizeof eb), "reset", eb, err) ||
			!mb_ok(mb_command_create(&cmd, eb, sizeof eb), "command", eb, err) ||
			!mb_ok(mb_command_set_style(cmd, st, eb, sizeof eb), "set_style", eb, err) ||
			!mb_ok(mb_command_set_target_speed(cmd, 1.2f, eb, sizeof eb), "speed", eb, err) ||
			!mb_ok(mb_command_set_seed(cmd, uint64_t(c.seed), eb, sizeof eb), "seed", eb, err)) {
		cleanup();
		return false;
	}
	uint32_t J = 0;
	mb_model_get_joint_count(g_mb, &J, eb, sizeof eb);
	// motionbricks_motion.pose_tracks(style, seconds, speed=1.2, seed, turn=True), step for step.
	const size_t want = size_t(double(c.seconds) * 30.0);
	std::vector<float> roots, local;
	double heading = 0.0;
	int plans = 0;
	g_prog.phase = "planning frames";
	g_prog.n = int64_t(want);
	while (roots.size() / 3 < want) {
		heading += 0.6 * std::sin(double(roots.size() / 3) / 30.0 * 0.5);
		const float dx = float(std::sin(heading)), dz = float(std::cos(heading));
		mb_motion *m = nullptr;
		if (!mb_ok(mb_command_set_movement_direction(cmd, dx, 0.f, dz, eb, sizeof eb), "dir", eb, err) ||
				!mb_ok(mb_command_set_facing_direction(cmd, dx, 0.f, dz, eb, sizeof eb), "face", eb, err) ||
				!mb_ok(mb_agent_plan(agent, cmd, &m, eb, sizeof eb), "plan", eb, err)) {
			cleanup();
			return false;
		}
		uint64_t fr = 0, jn = 0, tn = 0, rn = 0;
		const float *tp = nullptr, *rp = nullptr;
		mb_motion_get_frame_count(m, &fr, eb, sizeof eb);
		mb_motion_get_joint_count(m, &jn, eb, sizeof eb);
		mb_motion_get_root_translations(m, &tp, &tn, eb, sizeof eb);
		mb_motion_get_local_rotations_xyzw(m, &rp, &rn, eb, sizeof eb);
		const size_t take = std::min<size_t>(tn / 3, want - roots.size() / 3);
		for (size_t i = 0; i < take; ++i) {
			roots.insert(roots.end(), tp + i * 3, tp + i * 3 + 3);
			for (size_t j = 0; j < jn; ++j) {
				float mat[9];
				xyzw_matrix(rp + (i * jn + j) * 4, mat);
				local.insert(local.end(), mat, mat + 9);
			}
		}
		mb_motion_free(m);
		++plans;
		g_prog.i = int64_t(roots.size() / 3);
		if (take == 0 || !mb_ok(mb_agent_advance(agent, uint32_t(take), eb, sizeof eb), "advance", eb, err)) {
			if (take == 0) err = "planner returned no frames";
			cleanup();
			return false;
		}
		gas();
	}
	cleanup();
	logf_("motionbricks job %d: style %s, %zu frames from %d plans, seed %lld", c.id, style.c_str(), roots.size() / 3,
			plans, (long long)c.seed);
	c.frames = int(roots.size() / 3);
	c.joints = int(J);
	c.root_off = push_floats(roots.data(), roots.size());
	c.local_off = push_floats(local.data(), local.size());
	std::vector<float> rest(size_t(J) * 3);
	c.parent_off = g_ipool.size();
	c.names_off = g_names.size();
	for (uint32_t j = 0; j < J; ++j) {
		const char *nm = nullptr;
		int32_t par = -1;
		float x = 0, y = 0, z = 0, px = 0, py = 0, pz = 0;
		mb_model_get_joint_name(g_mb, j, &nm, eb, sizeof eb);
		mb_model_get_joint_parent(g_mb, j, &par, eb, sizeof eb);
		mb_model_get_neutral_joint_position(g_mb, j, &x, &y, &z, eb, sizeof eb);
		if (par >= 0) {
			mb_model_get_neutral_joint_position(g_mb, uint32_t(par), &px, &py, &pz, eb, sizeof eb);
		}
		rest[size_t(j) * 3] = x - px;
		rest[size_t(j) * 3 + 1] = y - py;
		rest[size_t(j) * 3 + 2] = z - pz;
		g_ipool.push_back(par);
		g_names.emplace_back(nm ? nm : "?");
	}
	c.rest_off = push_floats(rest.data(), rest.size());
	return true;
}

// --- the avatar and the source rigs --------------------------------------------------------
motion::Rig g_avatar;
bool g_have_avatar = false;
motion::Rig g_src[2];
bool g_have_src[2] = { false, false };
std::string g_avatar_bones[motion::ROLE_COUNT];

// The source skeletons' own joints for each role and chain end (these tables
// are the models' joint names, the counterpart of motion_poses.py's FLEXION).
const char *kSomaJoints[motion::ROLE_COUNT] = { "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftLeg",
	"RightLeg", "LeftShin", "RightShin", "Hips", "Spine1", "LeftFoot", "RightFoot", "Chest", "Neck1" };
const char *kSomaEnds[motion::ROLE_COUNT] = { "LeftHand", "RightHand", "LeftHand", "RightHand", "LeftFoot", "RightFoot",
	"LeftFoot", "RightFoot", "Head", "Head", "LeftToeBase", "RightToeBase", "Head", "Head" };
const char *kG1Joints[motion::ROLE_COUNT] = { "left_shoulder_pitch_skel", "right_shoulder_pitch_skel", "left_elbow_skel",
	"right_elbow_skel", "left_hip_pitch_skel", "right_hip_pitch_skel", "left_knee_skel", "right_knee_skel", "pelvis_skel", "waist_pitch_skel", "left_ankle_pitch_skel",
	"right_ankle_pitch_skel", "", "" };
const char *kG1Ends[motion::ROLE_COUNT] = { "left_hand_roll_skel", "right_hand_roll_skel", "left_hand_roll_skel",
	"right_hand_roll_skel", "left_ankle_roll_skel", "right_ankle_roll_skel", "left_ankle_roll_skel",
	"right_ankle_roll_skel", "left_shoulder_pitch_skel", "left_shoulder_pitch_skel", "left_toe_base", "right_toe_base", "", "" };

// A source rig from a finished clip's skeleton: identity rest frames (both
// models' rotations are relative to rest frames aligned with the world), rest
// positions by forward kinematics of the offsets.
bool source_rig(const Clip &c, std::string &err) {
	motion::Skeleton s;
	const size_t J = size_t(c.joints);
	s.names.assign(g_names.begin() + long(c.names_off), g_names.begin() + long(c.names_off + J));
	s.parent.assign(g_ipool.begin() + long(c.parent_off), g_ipool.begin() + long(c.parent_off + J));
	s.rot.assign(J * 9, 0.f);
	s.pos.assign(J * 3, 0.f);
	for (size_t j = 0; j < J; ++j) {
		s.rot[j * 9] = s.rot[j * 9 + 4] = s.rot[j * 9 + 8] = 1.f;
		const int p = s.parent[j];
		for (int k = 0; k < 3; ++k) {
			s.pos[j * 3 + size_t(k)] = g_pool[c.rest_off + j * 3 + size_t(k)] + (p >= 0 ? s.pos[size_t(p) * 3 + size_t(k)] : 0.f);
		}
	}
	const char **jn = c.model == KIMODO ? kSomaJoints : kG1Joints;
	const char **en = c.model == KIMODO ? kSomaEnds : kG1Ends;
	int joints[motion::ROLE_COUNT], ends[motion::ROLE_COUNT];
	for (int r = 0; r < motion::ROLE_COUNT; ++r) {
		joints[r] = s.find(jn[r]);
		ends[r] = s.find(en[r]);
	}
	g_src[c.model] = motion::measure(s, joints, ends, joints[motion::L_UPPER_LEG], joints[motion::R_UPPER_LEG]);
	if (!g_src[c.model].error.empty()) {
		err = std::string(kModelNames[c.model]) + ": " + g_src[c.model].error;
		return false;
	}
	g_have_src[c.model] = true;
	return true;
}

// The clip on the avatar, eight frames a slice.
bool run_retarget(Clip &c, std::string &err) {
	if (!source_rig(c, err)) {
		return false;
	}
	g_prog.phase = "retargeting frames";
	g_prog.n = c.frames;
	motion::Retargeted rt;
	motion::retarget_begin(rt, c.frames);
	for (int f = 0; f < c.frames; f += 8) {
		const int f1 = std::min(c.frames, f + 8);
		motion::retarget_range(g_src[c.model], &g_pool[c.local_off], f, f1, c.joints, g_avatar, rt,
				motion::rung_mask(env_int("MOTION_RUNG", 3)));
		g_prog.i = f1;
		gas();
	}
	char label[160];
	std::snprintf(label, sizeof label, "%s/%s", kModelNames[c.model], g_clip_prompt[size_t(c.id)].c_str());
	c.rt_off = push_floats(rt.local.data(), rt.local.size());
	c.rt_deg_off = push_floats(rt.target_deg.data(), rt.target_deg.size());
	g_clip_ranges[size_t(c.id)] = motion::ranges_text(label, rt);
	g_clip_axes[size_t(c.id)] = motion::axes_text(kModelNames[c.model], g_src[c.model]);
	c.retargeted = 1;
	return true;
}

// --- live: MotionBricks steered each frame ------------------------------------------------
struct Live {
	bool want = false;
	bool running = false;
	int64_t seed = 0;
	std::string error;
	float move[2] = { 0.f, 1.f };
	float face[2] = { 0.f, 1.f };
	float speed = 0.f;
	int64_t plans = 0;
	int64_t produced = 0;
	uint32_t joints = 0;
	std::vector<int32_t> parent;
	std::vector<float> offset; // joints x 3, from the parent in the neutral pose
	std::vector<std::string> names;
	std::deque<std::vector<float>> frames; // world joint positions, joints x 3 each
};
Live g_live;
// Frames planned ahead of the host, and committed per plan: a steer reaches the body within
// kLiveTake / 30 s of the frames already queued.
constexpr size_t kLiveAhead = 12;
constexpr size_t kLiveTake = 6;

// One frame's world joint positions from the root translation and the local rotations (XYZW).
std::vector<float> live_positions(const float *root, const float *xyzw) {
	const uint32_t J = g_live.joints;
	std::vector<float> world(size_t(J) * 9), at(size_t(J) * 3);
	for (uint32_t j = 0; j < J; ++j) {
		float local[9];
		xyzw_matrix(xyzw + size_t(j) * 4, local);
		const int32_t p = g_live.parent[j];
		float *r = &world[size_t(j) * 9];
		float *x = &at[size_t(j) * 3];
		if (p < 0) {
			std::copy(local, local + 9, r);
			std::copy(root, root + 3, x);
			continue;
		}
		const float *rp = &world[size_t(p) * 9];
		const float *o = &g_live.offset[size_t(j) * 3];
		for (int a = 0; a < 3; ++a) {
			for (int b = 0; b < 3; ++b) {
				r[a * 3 + b] = rp[a * 3] * local[b] + rp[a * 3 + 1] * local[3 + b] + rp[a * 3 + 2] * local[6 + b];
			}
			x[a] = at[size_t(p) * 3 + size_t(a)] + rp[a * 3] * o[0] + rp[a * 3 + 1] * o[1] + rp[a * 3 + 2] * o[2];
		}
	}
	return at;
}

bool run_live(std::string &err) {
	char eb[512] = { 0 };
	if (!mb_load(err)) return false;
	mb_style *idle = nullptr, *walk = nullptr;
	mb_agent *agent = nullptr;
	mb_command *cmd = nullptr;
	auto cleanup = [&] {
		if (cmd) mb_command_free(cmd);
		if (agent) mb_agent_free(agent);
		if (walk) mb_style_free(walk);
		if (idle) mb_style_free(idle);
	};
	if (!mb_ok(mb_style_load(g_mb, mb_style_path("idle").c_str(), &idle, eb, sizeof eb), "style_load idle", eb, err) ||
			!mb_ok(mb_style_load(g_mb, mb_style_path("walk").c_str(), &walk, eb, sizeof eb), "style_load walk", eb, err) ||
			!mb_ok(mb_agent_create(g_mb, &agent, eb, sizeof eb), "agent", eb, err) ||
			!mb_ok(mb_agent_reset(agent, idle, eb, sizeof eb), "reset", eb, err) ||
			!mb_ok(mb_command_create(&cmd, eb, sizeof eb), "command", eb, err) ||
			!mb_ok(mb_command_set_seed(cmd, uint64_t(g_live.seed), eb, sizeof eb), "seed", eb, err)) {
		cleanup();
		return false;
	}
	uint32_t J = 0;
	mb_model_get_joint_count(g_mb, &J, eb, sizeof eb);
	g_live.joints = J;
	g_live.parent.assign(J, -1);
	g_live.offset.assign(size_t(J) * 3, 0.f);
	g_live.names.assign(J, "?");
	for (uint32_t j = 0; j < J; ++j) {
		const char *nm = nullptr;
		float x = 0, y = 0, z = 0, px = 0, py = 0, pz = 0;
		mb_model_get_joint_name(g_mb, j, &nm, eb, sizeof eb);
		mb_model_get_joint_parent(g_mb, j, &g_live.parent[j], eb, sizeof eb);
		mb_model_get_neutral_joint_position(g_mb, j, &x, &y, &z, eb, sizeof eb);
		if (g_live.parent[j] >= int32_t(j)) {
			err = "MotionBricks joint " + std::to_string(j) + " comes before its parent";
			cleanup();
			return false;
		}
		if (g_live.parent[j] >= 0) {
			mb_model_get_neutral_joint_position(g_mb, uint32_t(g_live.parent[j]), &px, &py, &pz, eb, sizeof eb);
		}
		g_live.offset[size_t(j) * 3] = x - px;
		g_live.offset[size_t(j) * 3 + 1] = y - py;
		g_live.offset[size_t(j) * 3 + 2] = z - pz;
		g_live.names[j] = nm ? nm : "?";
	}
	g_live.running = true;
	g_prog.phase = "live";
	while (g_live.want) {
		if (g_live.frames.size() >= kLiveAhead) {
			pump::coop();
			continue;
		}
		const bool moving = g_live.speed > 0.05f;
		mb_motion *m = nullptr;
		if (!mb_ok(mb_command_set_style(cmd, moving ? walk : idle, eb, sizeof eb), "set_style", eb, err) ||
				!mb_ok(mb_command_set_target_speed(cmd, moving ? g_live.speed : 0.f, eb, sizeof eb), "speed", eb, err) ||
				!mb_ok(mb_command_set_movement_direction(cmd, g_live.move[0], 0.f, g_live.move[1], eb, sizeof eb), "dir", eb, err) ||
				!mb_ok(mb_command_set_facing_direction(cmd, g_live.face[0], 0.f, g_live.face[1], eb, sizeof eb), "face", eb, err) ||
				!mb_ok(mb_agent_plan(agent, cmd, &m, eb, sizeof eb), "plan", eb, err)) {
			cleanup();
			g_live.running = false;
			return false;
		}
		uint64_t jn = 0, tn = 0, rn = 0;
		const float *tp = nullptr, *rp = nullptr;
		mb_motion_get_joint_count(m, &jn, eb, sizeof eb);
		mb_motion_get_root_translations(m, &tp, &tn, eb, sizeof eb);
		mb_motion_get_local_rotations_xyzw(m, &rp, &rn, eb, sizeof eb);
		const size_t take = std::min<size_t>(kLiveTake, tn / 3);
		for (size_t i = 0; i < take && jn == J; ++i) {
			g_live.frames.push_back(live_positions(tp + i * 3, rp + i * jn * 4));
		}
		mb_motion_free(m);
		++g_live.plans;
		g_live.produced += int64_t(take);
		if (take == 0 || jn != J || !mb_ok(mb_agent_advance(agent, uint32_t(take), eb, sizeof eb), "advance", eb, err)) {
			if (take == 0) err = "planner returned no frames";
			if (jn != J) err = "planner returned " + std::to_string(jn) + " joints, the model has " + std::to_string(J);
			cleanup();
			g_live.running = false;
			return false;
		}
		gas();
	}
	cleanup();
	g_live.running = false;
	return true;
}

// --- the queue's runner, on the pump ------------------------------------------------------
void runner(void *) {
	while (!g_queue.empty()) {
		const auto [kind, id] = g_queue.front();
		g_queue.pop_front();
		if (kind == LIVE) {
			g_prog = Progress{};
			std::string err;
			if (!run_live(err)) {
				g_live.error = err;
				logf_("live: ERROR %s", err.c_str());
			}
			g_prog.phase = "idle";
			pump::coop();
			continue;
		}
		g_clips[size_t(id)].state = 1;
		g_prog = Progress{};
		g_prog.job = id;
		g_prog.model = g_clips[size_t(id)].model;
		std::string err;
		Clip c = g_clips[size_t(id)]; // the host may queue more while this runs: work on a copy
		bool ok = true;
		if (kind == GENERATE) {
			ok = c.model == KIMODO ? run_kimodo(c, err) : run_mbricks(c, err);
		}
		if (ok && g_have_avatar) {
			ok = run_retarget(c, err);
		}
		c.state = ok ? 2 : 3;
		g_clips[size_t(id)] = c;
		if (!ok) {
			g_clip_error[size_t(id)] = err;
			logf_("job %d (%s '%s'): ERROR %s", id, kModelNames[c.model], g_clip_prompt[size_t(id)].c_str(), err.c_str());
		} else {
			logf_("job %d (%s '%s'): DONE %d frames x %d joints%s", id, kModelNames[c.model],
					g_clip_prompt[size_t(id)].c_str(), c.frames, c.joints, c.retargeted ? ", retargeted" : "");
		}
		g_prog.phase = "idle";
		pump::coop(); // give the frame back between jobs
	}
}

// --- API ----------------------------------------------------------------------------------
Variant text(const std::string &s) {
	return Variant(String(s));
}

void apply_env(const std::string &env) {
	for (const char *k : { "GGML_RD_FAULT", "GGML_RD_BARRIER_ALL", "GGML_RD_TIMESTAMPS", "KIMODO_STEPS", "MOTION_GAS", "MOTION_RUNG" }) {
		unsetenv(k);
	}
	for (const std::string &kv : split(env, ' ')) {
		const size_t eq = kv.find('=');
		if (eq != std::string::npos) {
			setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1);
		}
	}
	const char *g = std::getenv("MOTION_GAS");
	g_gas = g && *g ? uint64_t(std::strtod(g, nullptr)) : 0;
}

} // namespace

static Variant motion_open(Object rd, int64_t total_mb, String models_dir) {
	if (rd.is_valid()) {
		g_dev.adopt(rd);
	}
	g_attached = g_dev.ok();
	static bool registered = false;
	if (!registered) {
		registered = true;
		ggml_backend_register(ggml_backend_rd_reg());
		ggml_backend_rd_attach(g_attached ? &g_dev : nullptr, size_t(total_mb) << 20);
		ggml_rd_hooks h;
		h.wait_gpu = hook_wait_gpu;
		h.coop = hook_coop;
		h.upload = hook_upload;
		h.read = hook_read;
		ggml_backend_rd_set_hooks(h);
		ggml_set_abort_callback(on_ggml_abort);
	}
	g_models = models_dir.utf8();
	return text(std::string(g_attached ? "OPEN device=" + g_dev.device_name() : std::string("OPEN no RD device")) +
			" models=" + g_models);
}

static Variant motion_env(String env) {
	apply_env(env.utf8());
	return text("ENV " + env.utf8());
}

static Variant motion_prompt_embedding(String prompt, PackedArray<float> embedding) {
	const std::vector<float> e = embedding.fetch();
	if (e.size() != 4096) {
		return text("FAIL: an LLM2Vec embedding is 4096 floats, got " + std::to_string(e.size()));
	}
	const std::string p = prompt.utf8();
	size_t i = 0;
	for (; i < g_emb_prompts.size() && g_emb_prompts[i] != p; ++i) {
	}
	if (i == g_emb_prompts.size()) {
		g_emb_prompts.push_back(p);
		g_emb.resize(g_emb.size() + 4096);
	}
	std::copy(e.begin(), e.end(), g_emb.begin() + long(i * 4096));
	return text("EMBEDDING " + std::to_string(i) + " '" + p + "' (stand-in: embedded natively by the host)");
}

static Variant motion_generate(String model, String prompt, double seconds, int64_t seed) {
	const std::string m = model.utf8();
	int mi = -1;
	for (int k = 0; k < 2; ++k) {
		if (m == kModelNames[k]) {
			mi = k;
		}
	}
	if (mi < 0) {
		return text("FAIL: model is kimodo_soma or motionbricks_g1, not '" + m + "'");
	}
	if (!(seconds > 0.0 && seconds <= 20.0)) {
		return text("FAIL: seconds must be in (0, 20]");
	}
	if (g_models.empty()) {
		return text("FAIL: motion_open first");
	}
	Clip c;
	c.id = int(g_clips.size());
	c.model = mi;
	c.seconds = float(seconds);
	c.seed = seed;
	g_clips.push_back(c);
	g_clip_prompt.push_back(prompt.utf8());
	g_clip_error.emplace_back();
	g_clip_style.emplace_back();
	g_clip_ranges.emplace_back();
	g_clip_axes.emplace_back();
	g_queue.push_back({ GENERATE, c.id });
	if (!pump::running()) {
		if (!pump::start(&runner, nullptr, size_t(16) << 20)) {
			return text("FAIL: the job fiber did not start");
		}
	}
	return text("QUEUED " + std::to_string(c.id));
}

static Variant motion_live_start(int64_t seed) {
	if (g_models.empty()) {
		return text("FAIL: motion_open first");
	}
	if (g_live.want) {
		return text("RUNNING");
	}
	g_live.want = true;
	g_live.seed = seed;
	g_live.error.clear();
	g_live.frames.clear();
	g_queue.push_back({ LIVE, -1 });
	if (!pump::running() && !pump::start(&runner, nullptr, size_t(16) << 20)) {
		g_live.want = false;
		return text("FAIL: the job fiber did not start");
	}
	return text("QUEUED live");
}

static Variant motion_live_steer(double mx, double mz, double fx, double fz, double speed) {
	const double ml = std::sqrt(mx * mx + mz * mz), fl = std::sqrt(fx * fx + fz * fz);
	if (ml > 1e-6) {
		g_live.move[0] = float(mx / ml);
		g_live.move[1] = float(mz / ml);
	}
	if (fl > 1e-6) {
		g_live.face[0] = float(fx / fl);
		g_live.face[1] = float(fz / fl);
	}
	g_live.speed = float(std::max(0.0, speed));
	return text("STEER");
}

// The frames planned since the last call, oldest first: [joints, frames, then frames x joints x 3].
static Variant motion_live_frames() {
	std::vector<float> out = { float(g_live.joints), float(g_live.frames.size()) };
	for (const std::vector<float> &f : g_live.frames) {
		out.insert(out.end(), f.begin(), f.end());
	}
	g_live.frames.clear();
	return Variant(PackedArray<float>(out));
}

static Variant motion_live_skeleton() {
	std::string s;
	for (size_t j = 0; j < g_live.names.size(); ++j) {
		s += g_live.names[j] + " " + std::to_string(g_live.parent[j]) + "\n";
	}
	return text(s);
}

static Variant motion_live_stop() {
	g_live.want = false;
	return text("STOPPING");
}

static Variant motion_live_status() {
	return text(std::string(g_live.running ? "RUNNING" : g_live.want ? "STARTING" : "STOPPED") +
			" plans=" + std::to_string(g_live.plans) + " frames=" + std::to_string(g_live.produced) +
			" buffered=" + std::to_string(g_live.frames.size()) + " permanent=" + std::to_string(g_dev.permanent_slots()) +
			(g_live.error.empty() ? "" : " error=" + g_live.error));
}

static Variant motion_pump(PackedArray<uint8_t> in) {
	return pump::step(in);
}

static Variant motion_status(int64_t id) {
	if (id < 0 || size_t(id) >= g_clips.size()) {
		return text("FAIL: no clip " + std::to_string(id));
	}
	const Clip &c = g_clips[size_t(id)];
	std::string s = std::string(state_name(c.state)) + " " + kModelNames[c.model] + " '" + g_clip_prompt[size_t(id)] + "'";
	if (c.state == 2) {
		s += " frames=" + std::to_string(c.frames) + " joints=" + std::to_string(c.joints);
	}
	if (c.state == 3) {
		s += " " + g_clip_error[size_t(id)];
	}
	return text(s);
}

static Variant motion_result(int64_t id) {
	if (id < 0 || size_t(id) >= g_clips.size()) {
		return text("FAIL: no clip " + std::to_string(id));
	}
	const Clip &c = g_clips[size_t(id)];
	Dictionary d = Dictionary::Create();
	d["id"] = Variant(int64_t(c.id));
	d["model"] = text(kModelNames[c.model]);
	d["prompt"] = text(g_clip_prompt[size_t(id)]);
	d["state"] = text(state_name(c.state));
	d["error"] = text(g_clip_error[size_t(id)]);
	d["seed"] = Variant(int64_t(c.seed));
	d["seconds"] = Variant(double(c.seconds));
	d["fps"] = Variant(double(kFps));
	d["frames"] = Variant(int64_t(c.frames));
	d["joints"] = Variant(int64_t(c.joints));
	d["steps"] = Variant(int64_t(c.steps));
	d["style"] = text(g_clip_style[size_t(id)]);
	d["stand_in"] = text(c.model == KIMODO ? "prompt embedded natively by the host (the Llama-3 text encoder is not in the guest)"
										   : "MotionBricks is style-driven: the prompt picks a released style");
	if (c.state != 2) {
		return Variant(d);
	}
	const size_t F = size_t(c.frames), J = size_t(c.joints);
	std::string names;
	for (size_t j = 0; j < J; ++j) {
		names += g_names[c.names_off + j] + "\n";
	}
	d["joint_names"] = text(names);
	d["parents"] = Variant(PackedArray<int32_t>(std::vector<int32_t>(g_ipool.begin() + long(c.parent_off),
			g_ipool.begin() + long(c.parent_off + J))));
	d["rest_offsets"] = Variant(PackedArray<float>(g_pool.data() + c.rest_off, J * 3));
	// Per frame, per joint: the parent-local 3x3 (row-major) then the
	// translation (the root's is the root position, the others their rest offset).
	std::vector<float> xf(F * J * 12);
	for (size_t f = 0; f < F; ++f) {
		for (size_t j = 0; j < J; ++j) {
			float *o = &xf[(f * J + j) * 12];
			std::memcpy(o, &g_pool[c.local_off + (f * J + j) * 9], 9 * sizeof(float));
			const float *t = g_ipool[c.parent_off + j] < 0 ? &g_pool[c.root_off + f * 3] : &g_pool[c.rest_off + j * 3];
			o[9] = t[0];
			o[10] = t[1];
			o[11] = t[2];
		}
	}
	d["transforms"] = Variant(PackedArray<float>(xf));
	return Variant(d);
}

// The sampler's raw output (Kimodo; empty for MotionBricks), for the gate.
static Variant motion_raw(int64_t id) {
	if (id < 0 || size_t(id) >= g_clips.size() || g_clips[size_t(id)].state != 2) {
		return Variant(PackedArray<float>(std::vector<float>()));
	}
	const Clip &c = g_clips[size_t(id)];
	return Variant(PackedArray<float>(g_pool.data() + c.raw_off, c.raw_n));
}

// The avatar, exported by the host from its asset: names and humanoid roles
// one per line, parents, and per bone its global rest (rest_global) and its
// parent-local rest (rest_local), 12 floats each: row-major 3x3, then the
// origin / translation.
static Variant motion_avatar(String names, PackedArray<int32_t> parents, PackedArray<float> rest_global,
		PackedArray<float> rest_local, String roles) {
	motion::Skeleton s;
	std::string nm = names.utf8(), rl = roles.utf8();
	// split() drops empty fields, and a bone without a role is an empty line.
	std::vector<std::string> role_of;
	{
		size_t i = 0;
		while (i < rl.size()) {
			const size_t j = std::min(rl.find('\n', i), rl.size());
			role_of.push_back(rl.substr(i, j - i));
			i = j + 1;
		}
	}
	s.names = split(nm, '\n');
	const std::vector<int32_t> par = parents.fetch();
	const std::vector<float> g = rest_global.fetch(), l = rest_local.fetch();
	const size_t J = s.names.size();
	if (par.size() != J || g.size() != J * 12 || l.size() != J * 12 || role_of.size() < J) {
		return text("FAIL: " + std::to_string(J) + " names, " + std::to_string(par.size()) + " parents, " +
				std::to_string(g.size()) + "/" + std::to_string(l.size()) + " rest floats, " +
				std::to_string(role_of.size()) + " roles");
	}
	s.parent.assign(par.begin(), par.end());
	s.rot.resize(J * 9);
	s.pos.resize(J * 3);
	s.local = l;
	for (size_t j = 0; j < J; ++j) {
		std::copy(&g[j * 12], &g[j * 12 + 9], &s.rot[j * 9]);
		std::copy(&g[j * 12 + 9], &g[j * 12 + 12], &s.pos[j * 3]);
	}
	auto role_bone = [&](const char *role) -> int {
		for (size_t j = 0; j < J; ++j) {
			if (role_of[j] == role) return int(j);
		}
		return -1;
	};
	int joints[motion::ROLE_COUNT], ends[motion::ROLE_COUNT];
	const char *end_role[motion::ROLE_COUNT] = { "LeftHand", "RightHand", "LeftHand", "RightHand", "LeftFoot",
		"RightFoot", "LeftFoot", "RightFoot", "Head", "Head", "LeftToes", "RightToes", "Head", "Head" };
	for (int r = 0; r < motion::ROLE_COUNT; ++r) {
		joints[r] = role_bone(motion::role_name(r));
		ends[r] = role_bone(end_role[r]);
		if (joints[r] < 0 || ends[r] < 0) {
			return text(std::string("FAIL: the avatar declares no ") + (joints[r] < 0 ? motion::role_name(r) : end_role[r]));
		}
		g_avatar_bones[r] = s.names[size_t(joints[r])];
	}
	g_avatar = motion::measure(s, joints, ends, joints[motion::L_UPPER_LEG], joints[motion::R_UPPER_LEG]);
	if (!g_avatar.error.empty()) {
		return text("FAIL: " + g_avatar.error);
	}
	g_have_avatar = true;
	return text(motion::axes_text("avatar", g_avatar));
}

// The clip on the avatar. A GENERATE job retargets on its own when the avatar
// is known; a clip made before motion_avatar is queued here for a RETARGET job
// (answers {state: QUEUED}; ask again in a later frame).
static Variant motion_retarget(int64_t id) {
	if (id < 0 || size_t(id) >= g_clips.size() || g_clips[size_t(id)].state != 2) {
		return text("FAIL: clip " + std::to_string(id) + " is not done");
	}
	if (!g_have_avatar) {
		return text("FAIL: motion_avatar first");
	}
	const Clip &c = g_clips[size_t(id)];
	Dictionary d = Dictionary::Create();
	d["id"] = Variant(int64_t(c.id));
	if (!c.retargeted) {
		g_clips[size_t(id)].state = 0;
		g_queue.push_back({ RETARGET, c.id });
		if (!pump::running() && !pump::start(&runner, nullptr, size_t(16) << 20)) {
			return text("FAIL: the job fiber did not start");
		}
		d["state"] = text("QUEUED");
		return Variant(d);
	}
	char label[160];
	std::snprintf(label, sizeof label, "%s/%s", kModelNames[c.model], g_clip_prompt[size_t(id)].c_str());
	d["state"] = text("DONE");
	d["clip"] = text(label);
	d["frames"] = Variant(int64_t(c.frames));
	d["fps"] = Variant(double(kFps));
	std::string bones, roles;
	for (int r = 0; r < motion::ROLE_COUNT; ++r) {
		bones += g_avatar_bones[r] + "\n";
		roles += std::string(motion::role_name(r)) + "\n";
	}
	d["bones"] = text(bones);
	d["roles"] = text(roles);
	d["local"] = Variant(PackedArray<float>(g_pool.data() + c.rt_off, size_t(c.frames) * motion::ROLE_COUNT * 12));
	d["target_deg"] = Variant(PackedArray<float>(g_pool.data() + c.rt_deg_off, size_t(c.frames) * motion::ROLE_COUNT));
	d["ranges"] = text(g_clip_ranges[size_t(id)]);
	d["axes"] = text(g_clip_axes[size_t(id)]);
	return Variant(d);
}

// What the job queue is doing now, for the host's progress line.
static Variant motion_progress() {
	Dictionary d = Dictionary::Create();
	d["job"] = Variant(int64_t(g_prog.job));
	d["model"] = text(g_prog.model >= 0 ? kModelNames[g_prog.model] : "");
	d["prompt"] = text(g_prog.job >= 0 && size_t(g_prog.job) < g_clip_prompt.size() ? g_clip_prompt[size_t(g_prog.job)] : "");
	d["phase"] = text(g_prog.phase);
	d["i"] = Variant(g_prog.i);
	d["n"] = Variant(g_prog.n);
	d["queued"] = Variant(int64_t(g_queue.size()));
	d["gas"] = Variant(int64_t(g_gas));
	d["gas_yields"] = Variant(g_gas_yields);
	return Variant(d);
}

static Variant motion_output() {
	return text(g_log);
}

static Variant motion_stats() {
	std::string r = ggml_backend_rd_stats();
	r += " rule4_same_frame_syncs=" + std::to_string(g_dev.same_frame_syncs());
	r += " syncs=" + std::to_string(g_dev.syncs());
	r += " submits=" + std::to_string(g_dev.submits());
	r += " queued=" + std::to_string(g_queue.size());
	r += " gas=" + std::to_string(g_gas) + " gas_yields=" + std::to_string(g_gas_yields);
	return text(r);
}

int main() {
	ADD_API_FUNCTION(motion_open, "String", "Object rd, int total_mb, String models_dir", "Attach the host's RenderingDevice; models_dir holds Kimodo-SOMA-RP-v1.1-gguf/ and MotionBricks-G1-GGML/");
	ADD_API_FUNCTION(motion_env, "String", "String env", "K=V switches for later jobs: GGML_RD_FAULT, GGML_RD_BARRIER_ALL, GGML_RD_TIMESTAMPS, KIMODO_STEPS, MOTION_GAS, MOTION_RUNG");
	ADD_API_FUNCTION(motion_prompt_embedding, "String", "String prompt, PackedFloat32Array embedding", "Stand-in: a prompt's native LLM2Vec embedding (4096 floats)");
	ADD_API_FUNCTION(motion_generate, "String", "String model, String prompt, float seconds, int seed", "Queue a clip: kimodo_soma | motionbricks_g1 -> QUEUED <id>");
	ADD_API_FUNCTION(motion_pump, "Array", "PackedByteArray data", "Resume the job queue once: [header, text, rid]");
	ADD_API_FUNCTION(motion_status, "String", "int id", "QUEUED | RUNNING | DONE | ERROR, with the clip");
	ADD_API_FUNCTION(motion_result, "Dictionary", "int id", "The clip: per-frame parent-local joint transforms (12 floats) and the skeleton");
	ADD_API_FUNCTION(motion_raw, "PackedFloat32Array", "int id", "Kimodo's sampler output (frames x 369), for the gate");
	ADD_API_FUNCTION(motion_avatar, "String", "String names, PackedInt32Array parents, PackedFloat32Array rest_global, PackedFloat32Array rest_local, String roles", "The avatar's bones from its asset; answers the measured flexion axes");
	ADD_API_FUNCTION(motion_retarget, "Dictionary", "int id", "The clip on the avatar (flexion of its role bones) and its clip ranges; queues the retarget if it is not made yet");
	ADD_API_FUNCTION(motion_progress, "Dictionary", "", "The job queue now: job, model, prompt, phase, i of n, gas");
	ADD_API_FUNCTION(motion_output, "String", "", "What the jobs logged");
	ADD_API_FUNCTION(motion_live_start, "String", "int seed", "Start MotionBricks live, steered by motion_live_steer");
	ADD_API_FUNCTION(motion_live_steer, "String", "float move_x, float move_z, float face_x, float face_z, float speed", "Movement and facing directions on the floor, and speed in m/s (0 stands idle)");
	ADD_API_FUNCTION(motion_live_frames, "PackedFloat32Array", "", "The frames planned since the last call: joints, frames, then world joint positions");
	ADD_API_FUNCTION(motion_live_skeleton, "String", "", "Each joint's name and parent, one per line");
	ADD_API_FUNCTION(motion_live_stop, "String", "", "Stop the live agent after its current plan");
	ADD_API_FUNCTION(motion_live_status, "String", "", "RUNNING | STARTING | STOPPED, with plan and frame counts");
	ADD_API_FUNCTION(motion_stats, "String", "", "ggml-rd and rd_compute counters");
	halt();
}
