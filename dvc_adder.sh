#!bash

find data -maxdepth 2 -type f \
    ! -name "*.dvc" \
    ! -name ".gitignore" \
    -exec dvc add {} +

find results/plots -maxdepth 1 -type f \
    ! -name "*.dvc" \
    ! -name ".gitignore" \
    -exec dvc add {} +
