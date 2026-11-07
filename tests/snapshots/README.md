# Snapshots

Stored results of the package on fixed synthetic inputs (small `.npz` / `.csv` files), compared by the tests marked
`snapshot` with a tolerance. They catch result changes that are not intended. When a change is intended (a fix or an
approved improvement), regenerate the affected files with the generator script in this folder and record the change
in FIXLOG.md.
