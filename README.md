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

## Where simulations run: docker or host

Every simulation entry point (the GUI, `run_sim`, `gen_variations`,
`manual_variation`, `update_variations`, ...) runs the tools in one of two
places, set in the GUI under **File > Simulation Settings** (stored in
`~/.mh-analog-designer/settings.json`):

- **Docker** (default): each job starts a fresh container of the project's
  image (`config.json` `container.image`) and runs xschem/ngspice/Xyce in it,
  with the project folder mounted. The GUI runs on your machine; only docker
  is needed there.
- **Host**: xschem/ngspice/Xyce are called directly on the machine the tool
  runs on, with no docker involved -- e.g. the GUI running inside the EDA
  image itself, or any machine with the tools installed. The PDK comes from
  the settings' PDK root/PDK, else `$PDK_ROOT`/`$PDK`, and the PDK name
  defaults to the tag of `container.image`; binaries come from `PATH`
  (`$XSCHEM`/`$NGSPICE`/`$XYCE` override). Needs bash (Linux, macOS, WSL).
  IHP PDKs need ngspice >= 44 and the `psp103`, `psp103_nqs`, `r3_cmc`,
  `mosvar` OSDI models in `libs.tech/ngspice/osdi`.

`--execution docker|host` on `run_sim`, or `ANALOG_DESIGNER_EXECUTION`, overrides
the setting for one run. The two modes share all of the pipeline code:
`analog_designer/core/executor.py` only decides where each command runs, and
`workspace.exec_path()` is the one place paths are mapped into a container.

## Standalone runner for project repos

**Data > Export Standalone Runner** (or
`python -m analog_designer.standalone.export PROJECT_ROOT`) writes
`tools/run_tb.py` into the project: one file that runs the project's
testbenches without this tool installed, e.g. from a shell inside the EDA
image. It is not a separate implementation -- it bundles this tool's own
pipeline (`run_sim.py` and the modules it imports, found by following the
imports from `analog_designer/standalone/cli.py`) and unpacks it to a
temporary directory when run. When the GUI opens a project that has one,
it regenerates it if the pipeline changed since, so the project's runner
always matches the tool version that last opened it (commit the update with
the project). `--check` reports a stale copy, e.g. for a project's CI.

```sh
python3 tools/run_tb.py --doctor                    # what it found: PDK, tools, OSDI models
python3 tools/run_tb.py --list                      # blocks, topologies, tests, condition counts
python3 tools/run_tb.py --block cmos_vref --dry     # temp output dir, deleted at the end: no trace in the repo
python3 tools/run_tb.py --block cmos_vref           # sim/ exactly as the GUI writes it; fresh tests skipped
python3 tools/run_tb.py --block cmos_vref --test startup --where corner=tt   # subset, never recorded
python3 tools/run_tb.py --block top --param X1_variation=<registered cmos_vref variation>
```

It runs on the host by default; `--execution docker` makes it start a
container of the project's image instead, like the GUI's docker mode.
Materialized schematics always go to `sim/<variation>/_src/`, never into the
tracked `sch/`.

## Requirements

`pip install -r requirements.txt`, plus, for anything that actually
simulates, either docker with the EDA image (docker mode; image name from a
project's `config.json` `container.image`, default
`eda-env-designer:ihp-sg13g2`) or the EDA tools and PDK on the machine
itself (host mode) -- see above.

## Tests

```sh
python -m pytest tests/ -q
```

Runs standalone -- no private "pro" code is required or imported anywhere in
this repo.

## License

MIT, see [LICENSE](LICENSE).
