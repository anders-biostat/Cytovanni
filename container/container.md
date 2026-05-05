# Build Container

The GitHub contains an apptainer definition file `container.def`, from which the .sif file can be built using
```bash
apptainer build -F cytovanni-docker.sif container.def
```
It is then possible to run scripts within this container, which contains all necessary dependencies for the package, as well as CUDA support as long as it is running on a machine with a CUDA-enabled graphics card.

Note however that the container is not writable, so if you need additional python packages you need to either modify container.def manually, or modify it through the shell with
```bash
sed '/    # custom packages/a\    uv pip install --system package1 package2' -i container.def
```
where package1 package2 should be replaced by whatever additional packages you need. Then rebuild the container to include the new packages.


# Use Container as Kernel
However, our preferred way is using this container as a Jupyter kernel.
For this, you need to create a script `init_kernel.sh` containing
```bash
#!/bin/bash
apptainer exec --nv /path/to/cytovanni-docker.sif python -m ipykernel "$@"
```
`--nv` enables the container to access the graphics card, and `/path/to` should be replaced by the correct path structure.
Apptainer automatically mounts the user directory; you can additionally mount other directories using `--bind`.
You then need to create a custom Jupyter kernel file, first by creating the folder
```bash
mkdir -p ~/.local/share/jupyter/kernels/cytovanni-docker
cd ~/.local/share/jupyter/kernels/cytovanni-docker
```
and then add a file `kernel.json` with
```json
{
  "argv": [
    "/path/to/init_kernel.sh",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (Cytovanni Docker)",
  "language": "python"
}
```
Copying `logo-64x64.png` into `~/.local/share/jupyter/kernels/cytovanni-docker` will additionally add a nice icon in the Jupyter launcher.

This custom kernel can then be used just like any other Jupyter kernel.

# R
We also provide `Rcontainer.def` to run CytoNorm. Compiling works the same way as above,
```bash
apptainer build -F cytovanni-R-docker.sif Rcontainer.def
```
along with `kernel.json`
```json
{
  "argv": [
    "/path/to/init_R_kernel.sh",
    "-f",
    "{connection_file}"
  ],
  "display_name": "R (Cytovanni Docker)",
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
apptainer exec /path/to/cytovanni-R-docker.sif R --slave -e "IRkernel::main('${CONN_FILE}')"
```
