import ControllerRoutes

open ControllerRoutes

def main (args : List String) : IO UInt32 := do
  match args with
  | ["paths"] =>
    for i in allInputs do IO.println (path i)
    return 0
  | ["table"] =>
    for i in allInputs do
      let r := route i
      IO.println s!"{path i}\t{repr r}\t{repr (prop .off r)}\t{repr (prop .fallback r)}"
    return 0
  | ["meta-bindings"] =>
    for i in allInputs do
      match metaBinding i with
      | some (action, p) => IO.println s!"{action}\t{p}"
      | none => pure ()
    return 0
  | _ => IO.eprintln "usage: controller_routes paths | table | meta-bindings"; return 2
