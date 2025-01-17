# Conversion script NeMo -> HuggingFace

Code for converting a NeMo checkpoint into the widely adopted HuggingFace format.

## Prerequisites   

Ensure the following libraries are installed in your environment:
```
   - torch
   - transformers
   - protobuf
   - sentencepiece
```
System requirements:
```
   - Minimum RAM: 80 GB of RAM is required for the conversion process of the 7b model.  
   - Disk Space: Sufficient storage to accommodate both the original and converted models is essential. If converting from a `.nemo` file, ensure at least 3x the original model size due to the decompression step.

## Files

- **`convert_salamandra_nemo2hf.py`**: Python script that performs the actual format conversion.
- **`convert_salamandra_nemo2hf.sh`**: Bash script to automate the model conversion.

## Input Arguments

The scripts take the following arguments:
- **`--model_path`**: Path to a `.nemo` file or (even better) a decompressed nemo directory.
- **`--output_path`**: Directory where the converted HuggingFace model will be saved.

## Usage Example

```bash
bash convert_salamandra_nemo2hf.sh --model_path /path/to/nemo_model --output_path /path/to/output_hf_model
```
