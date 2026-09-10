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
