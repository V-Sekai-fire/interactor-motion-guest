// The pose-envelope retarget of character-fox 13f3c4bf, as matrices (rule 11).
//
// Only the FLEXION angle of eight limb joints (and, by rung, the pelvis, spine and ankles) crosses from a source skeleton
// (Kimodo's SOMA-30 or MotionBricks' G1-34) to the avatar, and it is applied
// about the avatar's own flexion axis. G1 splits a shoulder into three single
// axis joints where the avatar has one bone per segment; composing them is a
// retarget, and a careless one invents contorted poses and so invented
// clipping. Hinges (elbow, knee) take the absolute angle; shoulders and hips
// keep their sign, so a wrong axis convention shows (a hand through the
// floor) instead of passing as plausible.
//
// "Measured, not assumed": a joint's flexion axis is the one of its three
// rest-frame axes whose rotation moves the chain's end (hand or foot) furthest
// along the body's forward direction, with the sign that bends the limb the
// way the joint bends (hand forward for shoulder and elbow, foot forward for
// the hip, foot backward for the knee). The same measurement is made on the
// source skeleton and on the avatar, from rest data only: the avatar's comes
// from the host (its imported FBX, never a table here), the sources' from
// their model files' own skeletons.
//
// Plain C++ (no sandbox API), so a native tool can compile it too.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace motion {

// Unity humanoid names: the roles the avatar's importer declares.
enum Role : int {
	L_UPPER_ARM,
	R_UPPER_ARM,
	L_LOWER_ARM,
	R_LOWER_ARM,
	L_UPPER_LEG,
	R_UPPER_LEG,
	L_LOWER_LEG,
	R_LOWER_LEG,
	// The ladder (MOTION_RUNG): 1 adds the pelvis, 2 the spine (the default), 3 the ankles; a role
	// above the rung stays at rest.
	HIPS,
	SPINE,
	L_FOOT,
	R_FOOT,
	ROLE_COUNT
};
constexpr int kLimbRoles = 8;
// The roles a rung drives, as a bit mask.
uint32_t rung_mask(int rung);
const char *role_name(int role);
bool role_is_hinge(int role);

// A skeleton at rest: per joint its parent (-1 for a root) and its GLOBAL
// rest frame, a row-major 3x3 whose columns are the joint's local axes in
// world space, and its origin. World is Y-up.
struct Skeleton {
	std::vector<std::string> names;
	std::vector<int> parent;
	std::vector<float> rot; // J x 9, row-major, columns orthonormalised by measure()
	std::vector<float> pos; // J x 3
	// Optional: each joint's rest transform relative to its parent as the
	// asset holds it (J x 12: row-major 3x3, then the translation), scale
	// included. The retarget composes onto it; empty means derive it from
	// the global frames.
	std::vector<float> local;
	int find(const std::string &name) const;
};

// A role's measured flexion axis on one skeleton.
struct Axis {
	int joint = -1;   // the joint that bends
	int end = -1;     // the chain end whose travel is measured
	int local = -1;   // 0, 1, 2: the joint's local x, y, z
	float sign = 1.f; // +1 or -1: positive rotation bends the limb
	float travel_mm[3] = { 0, 0, 0 };   // |end displacement| for 30 deg about each local axis
	float forward_mm[3] = { 0, 0, 0 };  // its component along forward
	float up_mm[3] = { 0, 0, 0 };       // its component along up
	bool tie = false; // two axes moved the end forward within 20%: the one that also lifts it won
};

struct Rig {
	Skeleton skel;
	Axis axis[ROLE_COUNT];
	float forward[3] = { 0, 0, 1 };
	std::string error; // non-empty: the rig could not be measured
};

// joints[r] / ends[r]: joint indices for role r and the end its travel is
// measured at; left_hip / right_hip give the forward direction
// (up x (right - left), up = +Y).
Rig measure(const Skeleton &skel, const int joints[ROLE_COUNT], const int ends[ROLE_COUNT], int left_hip,
		int right_hip);

// Signed angle (radians) of a rotation about a unit axis: the angle through
// which it turns a vector perpendicular to the axis, projected onto the plane
// perpendicular to it. For a rotation about the axis itself it is that
// rotation's angle.
float angle_about(const float r[9], const float axis[3]);

// Rotation of `angle` radians about a unit axis, row-major (Rodrigues).
void axis_angle_matrix(const float axis[3], float angle, float out[9]);

// One clip retargeted: frames x ROLE_COUNT local transforms of the avatar's
// role bones (row-major 3x3 rest-local rotation times the flexion, then the
// rest-local translation), the target angles (degrees) and the source angles.
struct Retargeted {
	int frames = 0;
	std::vector<float> local;      // frames x ROLE_COUNT x 12
	std::vector<float> target_deg; // frames x ROLE_COUNT
	std::vector<float> source_deg; // frames x ROLE_COUNT
};

// source_local: frames x source-joints x 9, each joint's rotation relative to
// its parent (the source rests are identity frames, so this is also the
// rotation away from rest). avatar: the measured avatar rig.
Retargeted retarget(const Rig &source, const float *source_local, int frames, int source_joints, const Rig &avatar);
// The same in slices (motion.elf's job retargets a few frames per slice:
// every guest job is gassable and resumable): retarget_begin sizes `out` for
// `frames`, and
// retarget_range fills frames [f0, f1). Any slicing gives the same bytes.
void retarget_begin(Retargeted &out, int frames);
void retarget_range(const Rig &source, const float *source_local, int f0, int f1, int source_joints, const Rig &avatar,
		Retargeted &out, uint32_t roles = ~0u);

// The clip-range report's lines for one clip: per role, the target range and
// the source range in degrees.
std::string ranges_text(const std::string &clip, const Retargeted &r);
// The axes the retarget used, on both sides, with the travel measurements.
std::string axes_text(const std::string &what, const Rig &rig);

} // namespace motion
