# motion.elf in a Sandbox: probe loads it, run makes the weight-free calls; tools/check_bintr.exs reads the "key value" lines.
#   godot --path tests/bintr --script run_motion.gd -- --mode=probe|run --translate=yes|no --elf=E --bintr=D
#         [--models=M --out=O --repeat=N]
extends SceneTree

const READ := 2
const UPLOAD := 3
const COOP := 4
const DONE := 5
const ERROR := 6
const WAIT_GPU := 1


func _initialize() -> void:
	var a := {}
	for arg in OS.get_cmdline_user_args():
		var kv := arg.trim_prefix("--").split("=", true, 1)
		a[kv[0]] = kv[1] if kv.size() > 1 else ""
	quit(_run(a))


func _run(a: Dictionary) -> int:
	if not ClassDB.class_exists("Sandbox"):
		return _fail("the Sandbox class is not registered")
	ProjectSettings.set_setting("sandbox/binary_translation/cache_dir", a.bintr + "/")
	ProjectSettings.set_setting("sandbox/binary_translation/enabled", a.get("translate") == "yes")
	var sb = ClassDB.instantiate("Sandbox")
	sb.memory_max = 1024
	sb.references_max = 4096
	sb.allocations_max = 1000000
	var elf := FileAccess.get_file_as_bytes(a.elf)
	if elf.is_empty():
		return _fail("%s did not read" % a.elf)
	sb.load_buffer(elf)
	if not sb.has_function("motion_avatar"):
		return _fail("%s did not load (no motion_avatar)" % a.elf.get_file())
	print("hash %08X" % (int(sb.get_translation_hash()) & 0xFFFFFFFF))
	print("translated %s" % ("yes" if sb.is_binary_translated() else "no"))
	if a.get("mode") != "run":
		sb.free()
		return 0

	var out := PackedStringArray()
	out.append(str(sb.vmcall("motion_open", 0, 64, a.models)))
	out.append(str(sb.vmcall("motion_env", "KIMODO_STEPS=2 MOTION_RUNG=4")))
	out.append(str(sb.vmcall("motion_generate", "no_such_model", "walk", 1.0, 7)))
	out.append(str(sb.vmcall("motion_generate", "kimodo_soma", "walk", 0.0, 7)))
	out.append(str(sb.vmcall("motion_status", 0)))
	var av := _avatar()
	var repeat := int(a.get("repeat", "1"))
	var t0 := Time.get_ticks_usec()
	var axes := ""
	for i in repeat:
		axes = str(sb.vmcall("motion_avatar", av.names, av.parents, av.rest_global, av.rest_local, av.roles))
	print("vm_ms %d" % ((Time.get_ticks_usec() - t0) / 1000))
	out.append(axes)
	var emb := PackedFloat32Array()
	emb.resize(4096)
	for i in 4096:
		emb[i] = float((i * 37) % 101 - 50) / 64.0
	out.append(str(sb.vmcall("motion_prompt_embedding", "a person walks forward", emb)))
	out.append(str(sb.vmcall("motion_generate", "kimodo_soma", "a person walks forward", 1.0, 7)))
	var pumps := 0
	var reads := 0
	while pumps < 1000:
		var r = sb.vmcall("motion_pump", PackedByteArray())
		pumps += 1
		if typeof(r) != TYPE_ARRAY or r.size() < 3:
			return _fail("motion_pump returned %s" % str(r))
		var hdr: PackedInt64Array = r[0]
		if hdr[0] == READ:
			reads += 1
			out.append("READ %s %d %d" % [str(r[1]).get_file(), hdr[1], hdr[2]])
		elif hdr[0] == DONE or hdr[0] == ERROR:
			out.append("%s %s" % ["DONE" if hdr[0] == DONE else "ERROR", str(r[1])])
			break
		elif hdr[0] != COOP and hdr[0] != WAIT_GPU:
			return _fail("unknown request kind %d" % hdr[0])
	out.append(str(sb.vmcall("motion_status", 0)))
	out.append(str(sb.vmcall("motion_output")))
	print("pumps %d" % pumps)
	print("reads %d" % reads)
	var text := "\n".join(out) + "\n"
	var w := FileAccess.open(a.out, FileAccess.WRITE)
	if w == null:
		return _fail("cannot write %s" % a.out)
	w.store_string(text)
	w.close()
	print("result_bytes %d" % text.to_utf8_buffer().size())
	print("translated_after %s" % ("yes" if sb.is_binary_translated() else "no"))
	sb.free()
	return 0


# A Y-up humanoid in T-pose facing +Z, every rest frame the identity: its flexion axes are known in closed form
# (30 degrees about an upper arm's -Y moves the hand, 540 mm out, 2 * 540 * sin(15 deg) = 279.5 mm).
func _avatar() -> Dictionary:
	var bones := [
		["Hips", -1, Vector3(0, 1.0, 0), "Hips"],
		["Spine", 0, Vector3(0, 1.1, 0), "Spine"],
		["Chest", 1, Vector3(0, 1.3, 0), "Chest"],
		["Neck", 2, Vector3(0, 1.5, 0), "Neck"],
		["Head", 3, Vector3(0, 1.62, 0), "Head"],
		["L_Arm", 2, Vector3(0.18, 1.45, 0), "LeftUpperArm"],
		["L_Elbow", 5, Vector3(0.46, 1.45, 0), "LeftLowerArm"],
		["L_Hand", 6, Vector3(0.72, 1.45, 0), "LeftHand"],
		["R_Arm", 2, Vector3(-0.18, 1.45, 0), "RightUpperArm"],
		["R_Elbow", 8, Vector3(-0.46, 1.45, 0), "RightLowerArm"],
		["R_Hand", 9, Vector3(-0.72, 1.45, 0), "RightHand"],
		["L_Leg", 0, Vector3(0.1, 0.95, 0), "LeftUpperLeg"],
		["L_Knee", 11, Vector3(0.1, 0.52, 0), "LeftLowerLeg"],
		["L_Foot", 12, Vector3(0.1, 0.08, 0), "LeftFoot"],
		["L_Toes", 13, Vector3(0.1, 0.02, 0.13), "LeftToes"],
		["R_Leg", 0, Vector3(-0.1, 0.95, 0), "RightUpperLeg"],
		["R_Knee", 15, Vector3(-0.1, 0.52, 0), "RightLowerLeg"],
		["R_Foot", 16, Vector3(-0.1, 0.08, 0), "RightFoot"],
		["R_Toes", 17, Vector3(-0.1, 0.02, 0.13), "RightToes"],
	]
	var names := ""
	var roles := ""
	var parents := PackedInt32Array()
	var rg := PackedFloat32Array()
	var rl := PackedFloat32Array()
	for b in bones:
		names += b[0] + "\n"
		roles += b[3] + "\n"
		parents.append(b[1])
		var t: Vector3 = b[2] - (bones[b[1]][2] if b[1] >= 0 else Vector3.ZERO)
		rg.append_array([1, 0, 0, 0, 1, 0, 0, 0, 1, b[2].x, b[2].y, b[2].z])
		rl.append_array([1, 0, 0, 0, 1, 0, 0, 0, 1, t.x, t.y, t.z])
	return { names = names, roles = roles, parents = parents, rest_global = rg, rest_local = rl }


func _fail(why: String) -> int:
	print("FAIL %s" % why)
	return 1
