/-!
Every input the engine registers for the headset's controller profile, routed to a motion-guest
motion or to a named use that is not a motion. `route` and `prop` are exhaustive matches, so an input
without a route, or a route without a prop decision, does not build; the theorems are checked by
`decide`.
-/

namespace ControllerRoutes

inductive Hand where
  | left | right
  deriving DecidableEq, Repr

/-- The profile's inputs: the first block on both hands, then the right's own, then the left's. -/
inductive Input where
  | gripPose (h : Hand) | aimPose (h : Hand) | gripSurfacePose (h : Hand)
  | systemTouch (h : Hand) | systemClick (h : Hand)
  | bumperTouch (h : Hand) | bumperClick (h : Hand)
  | trigger (h : Hand) | triggerTouch (h : Hand) | triggerClick (h : Hand)
  | squeeze (h : Hand) | squeezeTouch (h : Hand) | squeezeClick (h : Hand)
  | thumbstick (h : Hand) | thumbstickClick (h : Hand) | thumbstickTouch (h : Hand)
  | thumbstickUp (h : Hand) | thumbstickDown (h : Hand) | thumbstickLeft (h : Hand)
  | thumbstickRight (h : Hand)
  | haptic (h : Hand)
  | menuTouch | menuClick | aTouch | aClick | bTouch | bClick | xTouch | xClick | yTouch | yClick
  | viewTouch | viewClick
  | dpadUpTouch | dpadUpClick | dpadLeftTouch | dpadLeftClick | dpadDownTouch | dpadDownClick
  | dpadRightTouch | dpadRightClick
  deriving DecidableEq, Repr

/-- What motion-guest makes the avatar do. -/
inductive Motion where
  | move | sprint | snapTurn | teleportAim | crouch | jump | sitStand | grab | interact
  | emoteUp | emoteLeft | emoteDown | emoteRight | fingerCurl | handPose
  deriving DecidableEq, Repr

/-- Uses of an input that are not a motion. -/
inductive Use where
  | quickMenu | mainMenu | expressionMenu | emoji | micToggle | camera | pointer
  | runtimeReserved | hapticOutput
  deriving DecidableEq, Repr

inductive Route where
  | motion (m : Motion)
  | use (u : Use)
  deriving DecidableEq, Repr

/-- World grab is a fallback: off until the radial menu turns it on. -/
inductive GrabMode where
  | off | fallback
  deriving DecidableEq, Repr

/-- The scrappy prop built from CSG solids that shows while a route acts, or none. -/
inductive CsgProp where
  | none
  | turnMarker     -- a cone on a short cylinder at the feet, pointing the turn
  | landingRing    -- a torus on the floor at the teleport target, under an arc of small spheres
  | seat           -- a cylinder stool under the avatar
  | pinchHandles   -- a sphere at each gripping hand and a cylinder between them
  | emoteBubble    -- a sphere with a cone tail above the head
  | wristPanel     -- a rounded box on the off-hand wrist
  | floatingPanel  -- a box at arm's length
  | radialRing     -- a torus cut into wedges around the hand
  | micBadge       -- a capsule on a cylinder above the head
  | cameraBody     -- a box with a cylinder lens, held in the hand
  | laser          -- a thin cylinder out of the aim pose
  deriving DecidableEq, Repr

open Hand Input Motion Use

def hands : List Hand := [left, right]

def perHand (h : Hand) : List Input :=
  [gripPose h, aimPose h, gripSurfacePose h, systemTouch h, systemClick h, bumperTouch h,
   bumperClick h, trigger h, triggerTouch h, triggerClick h, squeeze h, squeezeTouch h,
   squeezeClick h, thumbstick h, thumbstickClick h, thumbstickTouch h, thumbstickUp h,
   thumbstickDown h, thumbstickLeft h, thumbstickRight h, haptic h]

def allInputs : List Input :=
  hands.flatMap perHand ++
  [menuTouch, menuClick, aTouch, aClick, bTouch, bClick, xTouch, xClick, yTouch, yClick,
   viewTouch, viewClick, dpadUpTouch, dpadUpClick, dpadLeftTouch, dpadLeftClick, dpadDownTouch,
   dpadDownClick, dpadRightTouch, dpadRightClick]

def allMotions : List Motion :=
  [.move, sprint, snapTurn, teleportAim, crouch, jump, sitStand, grab, interact, emoteUp,
   emoteLeft, emoteDown, emoteRight, fingerCurl, handPose]

def allRoutes : List Route := allMotions.map .motion ++
  [quickMenu, mainMenu, expressionMenu, micToggle, camera, pointer, runtimeReserved,
   hapticOutput].map .use

def side : Hand → String
  | left => "left"
  | right => "right"

def path : Input → String
  | gripPose h => s!"/user/hand/{side h}/input/grip/pose"
  | aimPose h => s!"/user/hand/{side h}/input/aim/pose"
  | gripSurfacePose h => s!"/user/hand/{side h}/input/grip_surface/pose"
  | systemTouch h => s!"/user/hand/{side h}/input/system/touch"
  | systemClick h => s!"/user/hand/{side h}/input/system/click"
  | bumperTouch h => s!"/user/hand/{side h}/input/bumper/touch"
  | bumperClick h => s!"/user/hand/{side h}/input/bumper/click"
  | trigger h => s!"/user/hand/{side h}/input/trigger/value"
  | triggerTouch h => s!"/user/hand/{side h}/input/trigger/touch"
  | triggerClick h => s!"/user/hand/{side h}/input/trigger/click"
  | squeeze h => s!"/user/hand/{side h}/input/squeeze/value"
  | squeezeTouch h => s!"/user/hand/{side h}/input/squeeze/touch"
  | squeezeClick h => s!"/user/hand/{side h}/input/squeeze/click"
  | thumbstick h => s!"/user/hand/{side h}/input/thumbstick"
  | thumbstickClick h => s!"/user/hand/{side h}/input/thumbstick/click"
  | thumbstickTouch h => s!"/user/hand/{side h}/input/thumbstick/touch"
  | thumbstickUp h => s!"/user/hand/{side h}/input/thumbstick/dpad_up"
  | thumbstickDown h => s!"/user/hand/{side h}/input/thumbstick/dpad_down"
  | thumbstickLeft h => s!"/user/hand/{side h}/input/thumbstick/dpad_left"
  | thumbstickRight h => s!"/user/hand/{side h}/input/thumbstick/dpad_right"
  | haptic h => s!"/user/hand/{side h}/output/haptic"
  | menuTouch => "/user/hand/right/input/menu/touch"
  | menuClick => "/user/hand/right/input/menu/click"
  | aTouch => "/user/hand/right/input/a/touch"
  | aClick => "/user/hand/right/input/a/click"
  | bTouch => "/user/hand/right/input/b/touch"
  | bClick => "/user/hand/right/input/b/click"
  | xTouch => "/user/hand/right/input/x/touch"
  | xClick => "/user/hand/right/input/x/click"
  | yTouch => "/user/hand/right/input/y/touch"
  | yClick => "/user/hand/right/input/y/click"
  | viewTouch => "/user/hand/left/input/view/touch"
  | viewClick => "/user/hand/left/input/view/click"
  | dpadUpTouch => "/user/hand/left/input/dpad_up/touch"
  | dpadUpClick => "/user/hand/left/input/dpad_up/click"
  | dpadLeftTouch => "/user/hand/left/input/dpad_left/touch"
  | dpadLeftClick => "/user/hand/left/input/dpad_left/click"
  | dpadDownTouch => "/user/hand/left/input/dpad_down/touch"
  | dpadDownClick => "/user/hand/left/input/dpad_down/click"
  | dpadRightTouch => "/user/hand/left/input/dpad_right/touch"
  | dpadRightClick => "/user/hand/left/input/dpad_right/click"

/-- The left stick moves and the right stick turns, aims a teleport, or crouches; both grips grab the
world and one grip an object; every touch and analog value curls the finger resting on it. -/
def route : Input → Route
  | gripPose _ | gripSurfacePose _ => .motion handPose
  | aimPose _ => .use pointer
  | systemTouch _ | bumperTouch _ | trigger _ | triggerTouch _ | squeeze _ | squeezeTouch _
  | thumbstickTouch _ => .motion fingerCurl
  | systemClick _ => .use runtimeReserved
  | bumperClick left => .use micToggle
  | bumperClick right => .use camera
  | triggerClick _ => .motion interact
  | squeezeClick _ => .motion grab
  | thumbstick left | thumbstickUp left | thumbstickDown left | thumbstickLeft left
  | thumbstickRight left => .motion .move
  | thumbstickClick left => .motion sprint
  | thumbstick right | thumbstickLeft right | thumbstickRight right => .motion snapTurn
  | thumbstickUp right => .motion teleportAim
  | thumbstickDown right | thumbstickClick right => .motion crouch
  | haptic _ => .use hapticOutput
  | menuTouch | aTouch | bTouch | xTouch | yTouch | viewTouch | dpadUpTouch | dpadLeftTouch
  | dpadDownTouch | dpadRightTouch => .motion fingerCurl
  | menuClick => .motion sitStand
  | viewClick => .use mainMenu
  | aClick => .motion jump
  | bClick | yClick => .use quickMenu
  | xClick => .use expressionMenu
  | dpadUpClick => .motion emoteUp
  | dpadLeftClick => .motion emoteLeft
  | dpadDownClick => .motion emoteDown
  | dpadRightClick => .motion emoteRight

def prop (g : GrabMode) : Route → CsgProp
  | .motion snapTurn => .turnMarker
  | .motion teleportAim => .landingRing
  | .motion sitStand => .seat
  | .motion grab => if g = .fallback then .pinchHandles else .none
  | .motion emoteUp | .motion emoteLeft | .motion emoteDown | .motion emoteRight => .emoteBubble
  | .motion .move | .motion sprint | .motion crouch | .motion jump | .motion interact
  | .motion fingerCurl | .motion handPose => .none
  | .use quickMenu => .radialRing
  | .use mainMenu => .floatingPanel
  | .use expressionMenu => .radialRing
  | .use emoji => .emoteBubble
  | .use micToggle => .micBadge
  | .use camera => .cameraBody
  | .use pointer => .laser
  | .use runtimeReserved | .use hapticOutput => .none

def isOutput : Input → Bool
  | haptic _ => true
  | _ => false

theorem every_input_is_listed : ∀ i : Input, i ∈ allInputs := by
  intro i; cases i <;> first | decide | (rename_i h; cases h <;> decide)

theorem the_profile_has_62_inputs : allInputs.length = 62 := by decide

theorem paths_are_distinct : (allInputs.map path).Nodup := by decide

theorem every_motion_has_an_input :
    allMotions.all (fun m => allInputs.any (route · == .motion m)) = true := by decide

theorem every_route_is_used : allRoutes.all (fun r => allInputs.any (route · == r)) = true := by decide

theorem pinch_handles_only_in_fallback :
    allRoutes.all (fun r => prop .off r != .pinchHandles) = true ∧
      prop .fallback (.motion grab) = .pinchHandles := by decide

theorem secondary_buttons_open_the_radial : route bClick = .use quickMenu ∧ route yClick = .use quickMenu := by
  decide

theorem only_outputs_route_to_haptics :
    allInputs.all (fun i => isOutput i == (route i == .use hapticOutput)) = true := by decide

/-- The same input on Meta's Quest 3 Touch Plus profile, as the pen's action and the path it binds,
or none where that controller has no such input (the profile is `/interaction_profiles/meta/touch_plus_controller`). -/
def metaBinding : Input → Option (String × String)
  | gripPose h => some ("grip_pose", s!"/user/hand/{side h}/input/grip/pose")
  | aimPose h => some ("aim_pose", s!"/user/hand/{side h}/input/aim/pose")
  | gripSurfacePose h => some ("palm_pose", s!"/user/hand/{side h}/input/grip_surface/pose")
  | trigger h => some ("trigger", s!"/user/hand/{side h}/input/trigger/value")
  | triggerTouch h => some ("trigger_touch", s!"/user/hand/{side h}/input/trigger/touch")
  | triggerClick h => some ("trigger_click", s!"/user/hand/{side h}/input/trigger/value")
  | squeeze h => some ("grip", s!"/user/hand/{side h}/input/squeeze/value")
  | squeezeClick h => some ("grip_click", s!"/user/hand/{side h}/input/squeeze/value")
  | thumbstick h | thumbstickUp h | thumbstickDown h | thumbstickLeft h | thumbstickRight h =>
    some ("primary", s!"/user/hand/{side h}/input/thumbstick")
  | thumbstickClick h => some ("primary_click", s!"/user/hand/{side h}/input/thumbstick/click")
  | thumbstickTouch h => some ("primary_touch", s!"/user/hand/{side h}/input/thumbstick/touch")
  | haptic h => some ("haptic", s!"/user/hand/{side h}/output/haptic")
  | menuClick => some ("menu_button", "/user/hand/left/input/menu/click")
  | aClick => some ("ax_button", "/user/hand/right/input/a/click")
  | aTouch => some ("ax_touch", "/user/hand/right/input/a/touch")
  | bClick => some ("by_button", "/user/hand/right/input/b/click")
  | bTouch => some ("by_touch", "/user/hand/right/input/b/touch")
  | xClick => some ("ax_button", "/user/hand/left/input/x/click")
  | xTouch => some ("ax_touch", "/user/hand/left/input/x/touch")
  | yClick => some ("by_button", "/user/hand/left/input/y/click")
  | yTouch => some ("by_touch", "/user/hand/left/input/y/touch")
  | _ => none

/-- What the station walk needs from a controller: every one is bound on Touch Plus. -/
def walkRoutes : List Route :=
  [.motion .move, .motion sprint, .motion snapTurn, .motion teleportAim, .motion jump, .motion grab,
   .motion interact, .use quickMenu]

theorem touch_plus_binds_every_walk_route :
    walkRoutes.all (fun r => allInputs.any (fun i => route i == r && (metaBinding i).isSome)) = true := by
  decide

/-! Control: with B and Y unbound the radial has no Touch Plus input, and the check sees it. -/

example : (walkRoutes.all fun r => allInputs.any fun i =>
    route i == r && (if i == bClick || i == yClick then false else (metaBinding i).isSome)) = false := by
  decide

/-! Control: dropping one input's route leaves its motion with none, and the coverage check sees it. -/

example : (allMotions.all fun m => (allInputs.filter (· != aClick)).any (route · == .motion m)) = false := by
  decide

end ControllerRoutes
