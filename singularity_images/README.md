# Singularity Images

This directory should contain the NeMo Framework containers.

However, since `.sif` files are quite large, we instead provide the steps required to download them:

## 1. Access the NGC website

Sign in to the NGC website, creating a new account if necessary: https://ngc.nvidia.com/signin

On the top-right corner, press your username and go to "Setup". Then generate a new API key or copy an already existing one. Save it for later in an environment variable:
```
NGC_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
Then, on the left-hand side of the website, go to `PRIVATE REGISTRY > CONTAINERS > NeMo Framework Training`.

Select a container (in our case it was version 24.03), copy the tag from the top-right corner, and store it in another environment variable:
```
CONTAINER_TAG=nvcr.io/ea-bignlp/ga-participants/nemofw-training:24.03
```

## 2. Download the image
This can be easily done with singularity's pull command:
```
module load singularity
singularity remote login -u \$oauthtoken -p $NGC_API_KEY docker://nvcr.io
singularity pull docker://$CONTAINER_TAG
```

**Important**: If you run out of memory while building the image (*disk quota exceeded* error), try redirecting the cache to a different directory.
By default, it points to your home directory (`$HOME/.singularity/cache`), which tends to be rather small in HPC environments.
Simply set the environment variables `SINGULARITY_TMPDIR` and `SINGULARITY_CACHEDIR` as follows in order to overcome this issue:
```
SINGULARITY_TMPDIR=<XXX> SINGULARITY_CACHEDIR=<XXX> singularity pull docker://$CONTAINER_TAG
```

## 3. Extract the image
Convert the SIF container to the sandbox format (writable directory) with the following command:
```
singularity build --sandbox nemo_2403 nemo_2403.sif
```
In this case, it will create a directory named *nemo_2403* with all the content from the SIF image, making error debugging much easier.
In addition, using the sandbox saves a lot of time every time a new job is launched.

Note that, from now on, the `$SINGULARITY_PATH` variable from the launcher script should point to the newly created directory rather than the `.sif` file!
