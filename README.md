# interactor-motion-guest

Text-to-motion and style-driven motion models run on the sandbox's ggml runtime as a guest program, for fitting a dress in motion.

## What it is for

The guest generates body motion inside the sandbox from a prompt or a style, so a garment can be checked for clipping while the avatar moves. A Lean route table maps every input of the headset's controller profile to a motion or a named use, and it does not build while any input lacks a route. [RFD 2277](https://github.com/V-Sekai-fire/manuals-weftspun/tree/main/rfd/2277-curvenet-cage-refit-and-unified-expressions-in-modular-avatar) owns the in-motion fit.

## Build and run

The guest program builds from this checkout, its vendored model ports and the goal manifest's sibling checkouts of `contract-guest-runtime`, `contract-ggml-rd`, `ggml` and `repository-riscv64-sysroot`. It then runs under the sandbox's emulator, natively translated and interpreted, and the two runs must agree:

```sh
elixir tools/build.exs
elixir tools/check_bintr.exs --elf=build/rv64/motion.elf
```

The route table builds on its own:

```sh
cd routes
lake build
```

## Licence

There is no licence file, and the guest and route sources state none. The vendored motion model ports are Apache-2.0.
