#!/bin/bash

module load singularity

singularity run --nv -B $PATH_TOKENIZER:/tokenizer,$PATH_DATA:/data,$PATH_RESULTS:/results,$PATH_LOGS:/logs,$PATH_CACHE:/cache $PATH_SINGULARITY bash -c "
  cd /opt/NeMo;
  git rev-parse HEAD;
  export PYTHONPATH=/opt/NeMo:\${PYTHONPATH};
  export LOCAL_RANK=$SLURM_LOCALID;
  python3 -u /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=$CONFIG_PATH --config-name=$CONFIG_NAME"
