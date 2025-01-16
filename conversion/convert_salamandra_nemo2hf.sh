#!/bin/bash
#SBATCH --job-name=nemo2hf
#SBATCH --output=logs/conversion_%j.out
#SBATCH --error=logs/conversion_%j.err
#SBATCH --qos=XXX
#SBATCH --account=XXX
#SBATCH --constraint=highmem
#SBATCH --exclusive
#SBATCH --nodes 1

# Function to display help
show_help() {
    echo "Usage: $0 --model_path <MODEL_PATH> --output_path <OUTPUT_PATH>"
    echo
    echo "Arguments:"
    echo "  --model_path      Path to the NeMo model file to convert."
    echo "  --output_path     Directory where the Hugging Face model will be saved."
    echo
    echo "Example:"
    echo "  $0 --model_path /gpfs/projects/bsc88/nemo-models/bsc_2b_4thepoch --output_path /gpfs/projects/bsc88/hf-models/copy_bsc_2b_4thepoch_hf"
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --output_path) OUTPUT_PATH="$2"; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Check if required arguments are provided
if [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Error: --model_path and --output_path arguments are required."
    show_help
    exit 1
fi

# Activate python environment
source venv/bin/activate

# Run conversion script
python convert_salamandra_nemo2hf.py --model_path "$MODEL_PATH" --output_path "$OUTPUT_PATH"
