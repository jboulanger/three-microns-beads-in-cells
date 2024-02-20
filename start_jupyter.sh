#!/bin/bash
srun -p cpu -c 112 apptainer run --writable-tmpfs --bind /cephfs2:/cephfs2,/cephfs:/cephfs /public/singularity/containers/lightmicroscopy/bioimaging-container/bioimaging.sif