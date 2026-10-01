import Lake
open Lake DSL

package routes

@[default_target] lean_lib ControllerRoutes

@[default_target] lean_exe controller_routes where
  root := `Main
