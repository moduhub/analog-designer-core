# analog-designer-core

A Tkinter GUI + CLI tooling for driving an xschem/ngspice (or Xyce) simulation
pipeline across parameter variations of an analog IC block: Monte Carlo and
manual variation generation, a design-profile classification engine, and a
release-export step for sharing a chosen variation as a presentable snapshot.

This is the public "core" distribution -- simulation orchestration (docker/
ngspice/Xyce) plus results listing, with its own test suite. It has no
perturb/combine-style variation generation, no trained-model/prediction
features, no design-space scatter-plot viewer, and no IC layout generation;
those live in the private `analog-designer-pro` tool, which pulls this repo
in as a git submodule and subclasses its GUI (`analog_designer.gui.
app.App`) to add its own tabs and actions on top.

Successor to the old hand-synced `mh-analog-designer-lite` fork -- this repo
is now the one source of truth for the public feature set, with its own CI
instead of manually ported commits.

## Usage

Point the tool at a project folder (a directory with a `config.json` +
`sch/`/`tb/`/`sim/`, e.g. a circuit design repo):

```sh
python -m analog_designer.gui.app [PROJECT_ROOT]
```

Omit `PROJECT_ROOT` to reopen the last-opened folder, or use the toolbar's
"Open Folder..." to switch projects without restarting. Individual CLI
scripts (`analog_designer.sim.run_sim`, `analog_designer.sim.gen_variations`,
`analog_designer.sim.manual_variation`, `analog_designer.sim.update_variations`,
`analog_designer.sim.check_params`, `analog_designer.sim.diagnose_tb`) all
take their own `--project-root` (and `--block`/`--topology`) for the same
purpose when run standalone.

## Standalone testbench runner

`analog_designer/standalone/run_tb.py` runs a project's testbenches with no
docker orchestration and without this tool installed: one file, Python
standard library only, calling `xschem`/`ngspice`/`Xyce` directly on the
machine it runs on -- typically from a shell *inside* an EDA container that
already has the tools and the PDK. Vendor it into a project repo with:

```sh
python -m analog_designer.standalone.export PROJECT_ROOT          # writes tools/run_tb.py
python -m analog_designer.standalone.export PROJECT_ROOT --check  # exit 1 if the copy is stale
```

then, from the project root:

```sh
python3 tools/run_tb.py --doctor                     # what it found: PDK, xschem, ngspice, Xyce, OSDI models
python3 tools/run_tb.py --list                       # blocks, topologies, tests, condition counts
python3 tools/run_tb.py --block cmos_vref --dry      # temp dir, deleted at the end: no trace in the repo
python3 tools/run_tb.py --block cmos_vref            # same sim/ layout + jsonl files the GUI reads
python3 tools/run_tb.py --block cmos_vref --test startup --where corner=tt -v
python3 tools/run_tb.py --block top --param X1_variation=<registered cmos_vref variation>
```

It mirrors `run_sim.py`'s pipeline: parameter resolution (width groups,
formulas, sub_blocks, `import_params`/`import_metrics`, generators),
condition grid, testbench placeholders, PDK corner names, the Xyce netlist
fixups, and the parser `extract()`/`evaluate()` contract. In the default
mode it writes core-compatible `sim/variations.jsonl`, `sim/results.jsonl`
and `sim/<variation>/runs.jsonl` with the same definition hash, so the GUI
treats those results as fresh. Materialized schematics always go to
`sim/<variation>/_src/`, never into the tracked `sch/`. `--dry` reads the
project and writes nothing to it (not even `__pycache__`); `--json` saves a
summary. Not covered: openEMS tests, the SOA check, and plots (parsers get
`plot_base=None`).

The PDK is found from `$PDK_ROOT`/`$PDK` (or `--pdk-root`/`--pdk`; the PDK
name falls back to the tag of `config.json`'s `container.image`). Binaries
come from `PATH` or `$XSCHEM`/`$NGSPICE`/`$XYCE`. IHP PDKs need ngspice >= 44
(OSDI 0.4) with `psp103`, `psp103_nqs`, `r3_cmc`, `mosvar` compiled into
`libs.tech/ngspice/osdi` (or `--osdi-dir`). xschem only netlists (`-x`), so
it needs no X server; `--xvfb` wraps it in `xvfb-run` for an xschem that
does.

`tests/test_standalone_run_tb.py` keeps the copy honest: parity checks
against `run_sim.py` on the same inputs, plus an end-to-end run against fake
xschem/ngspice executables.

## Requirements

`pip install -r requirements.txt`, plus a running EDA docker container
(image name configurable via a project's `config.json` `container.image`,
defaults to `eda-env-designer:ihp-sg13g2`) for anything that actually
simulates.

## Tests

```sh
python -m pytest tests/ -q
```

Runs standalone -- no private "pro" code is required or imported anywhere in
this repo.

## License

MIT, see [LICENSE](LICENSE).
