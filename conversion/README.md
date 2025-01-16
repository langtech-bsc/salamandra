# Conversion script NeMo -> HuggingFace

Code for converting a NeMo checkpoint into the widely adopted HuggingFace format.

## Prerequisites   

Make sure that the following libraries are installed in your environment:
```
   - torch
   - transformers
   - protobuf
   - sentencepiece
```

## Files

- **`convert_salamandra_nemo2hf.py`**: Python script that performs the actual format conversion.
- **`convert_salamandra_nemo2hf.sh`**: Bash script to automate the model conversion.

## Input Arguments

The scripts take the following arguments:
- **`--model_path`**: Path to the NeMo model to be converted. This should point to the directory containing the model's `.nemo` file or (even better) a decompressed nemo directory.
- **`--output_path`**: Directory where the converted HuggingFace model will be saved.

## Usage Example

```bash
bash convert_salamandra_nemo2hf.sh --model_path /path/to/nemo_model --output_path /path/to/output_hf_model
```
