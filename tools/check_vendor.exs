# motion.elf builds from vendor/: nothing under guest/, cmake/ or tools/ (or the top CMakeLists.txt) may name a
# 3-interactor/*-ggml checkout, a *_GGML_ROOT, or the archived interactor-dress-on.
#   elixir tools/check_vendor.exs              # this checkout
#   elixir tools/check_vendor.exs --self-test  # a planted line of each kind must fail, a clean tree must pass
defmodule CheckVendor do
  @root Path.expand("..", __DIR__)
  @scanned ~w(guest cmake tools CMakeLists.txt)
  @patterns [
    {"a 3-interactor/*-ggml checkout", ~r{3-interactor/[A-Za-z0-9._-]*-ggml}},
    {"a *_GGML_ROOT", ~r{\b[A-Z0-9]+(?:_[A-Z0-9]+)*_GGML_ROOT\b}},
    {"the archived interactor-dress-on", ~r{interactor-dress-on|DRESS_ON_ROOT}}
  ]

  def main(["--self-test"]), do: self_test()

  def main([]) do
    {files, hits} = scan(@root)
    for {file, line, n, what} <- hits, do: IO.puts("#{file}:#{n}: #{what}: #{String.trim(line)}")
    IO.puts("== #{length(files)} files read (this script, which spells the patterns, is not), #{length(hits)} lines name a sibling checkout")
    if files == [] or hits != [], do: System.halt(1)
  end

  def main(_), do: (IO.puts(:stderr, "usage: elixir tools/check_vendor.exs [--self-test]"); System.halt(2))

  defp scan(root) do
    files =
      for top <- @scanned, path = Path.join(root, top), File.exists?(path),
          file <- (if File.dir?(path), do: Path.wildcard(Path.join(path, "**/*"), match_dot: true), else: [path]),
          File.regular?(file), Path.expand(file) != Path.expand(__ENV__.file), do: file

    hits =
      for file <- files, {:ok, text} <- [File.read(file)], String.valid?(text),
          {line, n} <- Enum.with_index(String.split(text, "\n"), 1),
          {what, re} <- @patterns, Regex.match?(re, line),
          do: {Path.relative_to(file, root), line, n, what}

    {files, hits}
  end

  defp self_test do
    tmp = Path.join(System.tmp_dir!(), "check_vendor_#{System.unique_integer([:positive])}")
    File.mkdir_p!(Path.join(tmp, "cmake"))
    File.mkdir_p!(Path.join(tmp, "tools"))
    clean = "set(_k ${MOTION_GUEST_ROOT}/vendor/kimodo-ggml/src)\nset(GGML_ROOT ${WEFT_ROOT}/2-contract/ggml)\n" <>
              "set(GGML_RD_ROOT ${WEFT_ROOT}/2-contract/ggml-rd)\n"
    planted = [
      {"cmake/motion.cmake", "set(KIMODO_GGML_ROOT ${WEFT_ROOT}/3-interactor/kimodo-ggml)\n"},
      {"cmake/motion.cmake", "add_stage_elf(motion ${WEFT_ROOT}/3-interactor/rf-detr-ggml/src/ops.cpp)\n"},
      {"tools/x.exs", "root = System.get_env(\"MB_GGML_ROOT\")\n"},
      {"CMakeLists.txt", "add_subdirectory(${DRESS_ON_ROOT}/vendor/ggml ggml)\n"}
    ]
    results =
      [{"a clean tree passes", fn -> write(tmp, "cmake/motion.cmake", clean) end, 0},
       {"control: a tree with nothing to read fails", fn -> :ok end, 1}] ++
        for {file, line} <- planted do
          {"control: #{file} with #{String.trim(line)} fails", fn -> write(tmp, file, clean <> line) end, 1}
        end
    failed =
      Enum.count(results, fn {name, plant, want} ->
        File.rm_rf!(tmp)
        File.mkdir_p!(tmp)
        plant.()
        {files, hits} = scan(tmp)
        got = if files == [] or hits != [], do: 1, else: 0
        ok = got == want
        IO.puts("== #{if ok, do: "PASS", else: "FAIL"} #{name} (#{length(hits)} hits in #{length(files)} files)")
        not ok
      end)
    File.rm_rf!(tmp)
    IO.puts("== #{length(results) - failed} of #{length(results)} controls pass")
    if failed > 0, do: System.halt(1)
  end

  defp write(root, rel, text) do
    path = Path.join(root, rel)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, text)
  end
end

CheckVendor.main(System.argv())
