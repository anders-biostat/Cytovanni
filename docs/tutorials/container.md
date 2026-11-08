# Apptainer

## LLMs
Claude code should be able to prepare the container and set up the kernel, if given the relevant information in `https://github.com/anders-biostat/Cytovanni/blob/main/llms.txt`.

## Build Container

The GitHub contains an apptainer definition file `Cytovanni/container/container.def`, from which the `.sif` file can be built using
```bash
apptainer build -F cytovanni-container.sif container.def
```
It is then possible to run scripts within this container, which contains all necessary dependencies for the package, as well as CUDA support as long as it is running on a machine with a CUDA-enabled graphics card.

Note however that the container is not writable, so if you need additional python packages you need to either modify container.def manually, or modify it through the shell with
```bash
sed '/    # custom packages/a\    uv pip install --system package1 package2' -i container.def
```
where package1 package2 should be replaced by whatever additional packages you need. Then rebuild the container to include the new packages.


## Run the Tests
The container also holds the test tools (pytest, pytest-cov, pytest-xdist). The tests are not part of the installed
package: run them from a clone of the repository, where pytest uses the code of the clone (`src/`) rather than the
installed version.
```bash
git clone https://github.com/anders-biostat/Cytovanni.git && cd Cytovanni
apptainer exec --nv /path/to/cytovanni-container.sif python -m pytest -n 4
```
The default run uses small synthetic data and takes about 5 minutes with 6 workers (`-n 6`); `-m "not slow"` leaves out the larger
fits, and `--cov` adds a coverage report. Tests that need a GPU are skipped without one (`--nv` passes the GPU to the
container).

The end-to-end tests run the tutorial pipelines on a part of the example data and are not part of the default run:
```bash
CYTOVANNI_TEST_DATA=/path/to/data CYTOVANNI_TEST_CACHE=/path/to/cache \
    apptainer exec --nv /path/to/cytovanni-container.sif python -m pytest -m data tests/e2e
```
`CYTOVANNI_TEST_DATA` is the folder with the example data sets (`example_fixed-spectra`, `example_variable-spectra`).
Every stage of the pipelines writes its results to `CYTOVANNI_TEST_CACHE`, so a later run reuses them;
`-m "data and not slow"` runs only the stages that need no model training. The other options are described in
`tests/e2e/conftest.py`. Apptainer passes environment variables of the host into the container.

The tests marked `keops` need a working KeOps compilation (`-m keops`); `tests/snapshots/README.md` describes the
stored results that `tests/test_snapshots.py` compares with.

## Use Container as Jupyter Kernel
Our preferred way is using this container as a Jupyter kernel.
For this, you need to create a script `init_kernel.sh` containing
```bash
#!/bin/bash
apptainer exec --nv /path/to/cytovanni-container.sif python -m ipykernel "$@"
```
`--nv` enables the container to access the graphics card, and `/path/to` should be replaced by the correct path structure.
Apptainer automatically mounts the user directory; you can additionally mount other directories using `--bind`.
You then need to create a custom Jupyter kernel file, first by creating the folder
```bash
mkdir -p ~/.local/share/jupyter/kernels/cytovanni-container
cd ~/.local/share/jupyter/kernels/cytovanni-container
```
and then add a file `kernel.json` with
```json
{
  "argv": [
    "/path/to/init_kernel.sh",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (Cytovanni)",
  "language": "python"
}
```
Copying `logo-64x64.png` into `~/.local/share/jupyter/kernels/cytovanni-container` will additionally add a nice icon in the Jupyter launcher.

This custom kernel can then be used just like any other Jupyter kernel.

## R
We also provide `Rcontainer.def` to run CytoNorm. Compiling works the same way as above,
```bash
apptainer build -F cytovanni-R-container.sif Rcontainer.def
```
along with `kernel.json`
```json
{
  "argv": [
    "/path/to/init_R_kernel.sh",
    "-f",
    "{connection_file}"
  ],
  "display_name": "R (Cytovanni)",
  "language": "R"
}
```

However, due to changes in the syntax, the `init_R_kernel.sh` needs to be a bit more complex:
```bash
#!/bin/bash
for i in "$@"; do
    if [[ "$prev" == "-f" ]]; then
        CONN_FILE="$i"
    fi
    prev="$i"
done
apptainer exec /path/to/cytovanni-R-container.sif R --slave -e "IRkernel::main('${CONN_FILE}')"
```
