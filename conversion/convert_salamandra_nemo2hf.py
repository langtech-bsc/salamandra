import os
import re
import yaml
import glob
import torch
import tarfile
import logging
import argparse
import warnings
import tempfile
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import tensorstore as ts
from collections import OrderedDict
from difflib import get_close_matches
from transformers import (
    LlamaConfig, 
    LlamaForCausalLM, 
    LlamaTokenizer, 
    LlamaTokenizerFast, 
    convert_slow_tokenizer
)

warnings.simplefilter(action='ignore', category=FutureWarning)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.CRITICAL)


class MultipleTokenizersError(Exception):
    pass


def _unpack_nemo_file(path2file: str, out_folder: str, extract_config_only: bool = False) -> str:

    if not os.path.exists(path2file):
        raise FileNotFoundError(f"{path2file} does not exist")

    tar_header = "r:"
    try:
        tar_test = tarfile.open(path2file, tar_header)
        tar_test.close()
    except IsADirectoryError:
        return path2file
    except tarfile.ReadError:
        tar_header = "r:gz"
    tar = tarfile.open(path2file, tar_header)
    if not extract_config_only:
        tar.extractall(path=out_folder)
    else:
        members = [x for x in tar.getmembers() if ".yaml" in x.name]
        tar.extractall(path=out_folder, members=members)
    tar.close()
    return out_folder

def _open_ts_array(arr_path):
    """
    Opens a Zarr file array with Tensorstore with basic settings.

    Arguments:
        arr_path (Path): path to a Zarr (Tensorstore) array
    """
    spec = {'driver': 'zarr', 'metadata_key': '.zarray', 'kvstore': {}}
    spec['kvstore'] = {
        'driver': 'file',
        'path': str(arr_path),
    }
    try:
        arr = ts.open(ts.Spec(spec), open=True).result()
    except Exception as e:
        raise CheckpointingException(f'Array {arr_path} could not be loaded. Error: {e}') from e
    return arr

def _get_num_layer(element_name: str) -> str:
    """
    Returns the layer number of the element based on the name
        If name is of type shard_x_y.pt -> return x
        If name is of type x.y -> return x
        If name is of type x.y.z -> return x
        Else -> NotimplementedError
       
    Arguments:
        element (str): Name of the element to extract the layer number
    """
    numbers_pattern = r'\d+'
    allowed_patterns = [
            re.compile(fr'shard_{numbers_pattern}_{numbers_pattern}.pt'),
            re.compile(fr'{numbers_pattern}.{numbers_pattern}'),
            re.compile(fr'{numbers_pattern}.{numbers_pattern}.{numbers_pattern}')
        ]
    if any(pattern.match(element_name) for pattern in allowed_patterns):
        all_numbers = re.findall(numbers_pattern, element_name)
        return all_numbers[0]
    else:
        raise NotImplementedError(f"The name of the file {element_name} does not match any of the allowed patterns")

def postprocess_numpy_array(loaded_array):
    """
    When loading Zarr arrays in bfloat16 format, they cannot be converted to bf16 torch tensors right away. This function does exactly that.
    """

    x = loaded_array
    if x.dtype == np.dtype('bfloat16'):
        try:
            x = x.astype(np.dtype('float32'))
            x = torch.from_numpy(x)
            x = x.bfloat16()
        except Exception as e:
            raise e
    else:
        x = torch.from_numpy(x)
    return x


def _load_element(element, ten = None, layer_num = None):
    """
    Loads the element (of a NeMo model) in the Path which can be:
        - *pt -> use torch.load
        - weights in the zarr format -> use the open_ts_array function from the tensorstore library

    Arguments:
        element (Path): Path to a .pt tensor or a Zarr (Tensorstore) array
    """

    content = None
    if element.endswith(".pt"):
        content = torch.load(element)
    else:
        if ten is None:
            ten = deepcopy(_open_ts_array(os.path.dirname(element)).read().result())
        try: 
            folder_with_tensor = os.path.basename(os.path.dirname(element))
            if folder_with_tensor.endswith("output_layer.weight") or \
                    folder_with_tensor.endswith(".bias") or \
                    "final_layernorm" in folder_with_tensor or \
                    "word_embeddings.weight" in folder_with_tensor:
                content = ten
            elif "layer_norm" in folder_with_tensor or \
                    folder_with_tensor.endswith(".weight"):
                content = ten[layer_num]
            else:
                logging.info("_load_element, line 141")
                raise NotImplementedError
        except Exception as e:
            raise e
        assert content is not None, "Content is None when loading element from NeMo model. You should implement a better _load_element function in the modeling class."
        content = postprocess_numpy_array(content)
    return content, ten

    def _search_most_similar(list_of_keys, key):
        """
        Returns the key of the list_of_keys that is most similar to key.
        """

        closest_matches = get_close_matches(key, list_of_keys)
        return closest_matches


def convert_from_nemo_file(nemo_file, output_hf_path):
    """
    Converts NeMo weights to HuggingFace weights.
    """

    logging.captureWarnings(True)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp = _unpack_nemo_file(nemo_file, tmpdir)

        nemo_cfg = None
        with open(Path(temp) / "model_config.yaml") as handle:
            nemo_cfg = yaml.safe_load(handle)

        tokenizer_path = glob.glob(str(Path(temp) / "*.model"))
        if len(tokenizer_path) != 1:
            new_line_and_tab = "\n\t"
            raise MultipleTokenizersError(f"Zero or more than one *.model files found in {nemo_file}:{new_line_and_tab.join([t for t in tokenizer_path])}")
        tokenizer_path = tokenizer_path[0]

        tokenizer = LlamaTokenizer(vocab_file=tokenizer_path, local_files_only=True, legacy=False)
        tmp_tokenizer = convert_slow_tokenizer.convert_slow_tokenizer(tokenizer)
        fast_tokenizer = LlamaTokenizerFast(tokenizer_object=tmp_tokenizer)
        tokenizer_length = len(fast_tokenizer)

        dtype = torch.bfloat16 if "bf16" in nemo_cfg['precision'] else torch.float32
        logging.info(f"Using precision {dtype}")

        common_state_dict = torch.load(Path(temp) / "model_weights" / "common.pt", map_location='cpu')

        for directory in glob.glob(str(Path(temp) / "model_weights" / "*/")):
            if os.path.isdir(directory):
                logging.info(f"Loading {directory}")
                ten = None
                for element in glob.glob(str(Path(directory) / '*')):
                    # This if filters all extra states from the common_state_dict
                    if not element.endswith(".pt"):
                        logging.info("element:", element)
                        layer_num = None
                        if "layers" in directory:
                            element_name = os.path.basename(element)
                            layer_num = int(_get_num_layer(element_name))
                            key_name = os.path.basename(directory).replace("layers.", f"layers.{layer_num}.") 
                        else:
                            key_name = os.path.basename(os.path.dirname(element))
                        content, ten = _load_element(element, ten, layer_num)
                        common_state_dict[key_name] = content

        logging.info("Finished loading the elements to the common_state_dict")

    param_to_weights = lambda param: param.to(dtype)
    checkpoint = OrderedDict()

    hidden_size = nemo_cfg["hidden_size"]
    head_num = nemo_cfg["num_attention_heads"]
    num_layers = nemo_cfg["num_layers"]
    ffn_hidden_size = nemo_cfg["ffn_hidden_size"]
    num_query_groups = nemo_cfg.get("num_query_groups", head_num)
    if num_query_groups is None:
        num_query_groups = head_num

    head_size = hidden_size // head_num
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    # embedding
    embed_weight = common_state_dict[f'model.embedding.word_embeddings.weight']
    embed_weights_base_name = f'model.embed_tokens.weight'
    checkpoint[embed_weights_base_name] = param_to_weights(embed_weight)

    for l in tqdm(range(int(num_layers))):
        logging.info(f"Converting layer {l}...")

        qkv_weights = common_state_dict[f'model.decoder.layers.{l}.self_attention.linear_qkv.weight']
        qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

        q_slice = torch.cat(
            [
                torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
        v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

        q_weights_base_name = f'model.layers.{l}.self_attn.q_proj.weight'
        k_weights_base_name = f'model.layers.{l}.self_attn.k_proj.weight'
        v_weights_base_name = f'model.layers.{l}.self_attn.v_proj.weight'

        checkpoint[q_weights_base_name] = param_to_weights(qkv_weights[q_slice].reshape(-1, hidden_size))
        checkpoint[k_weights_base_name] = param_to_weights(qkv_weights[k_slice].reshape(-1, hidden_size))
        checkpoint[v_weights_base_name] = param_to_weights(qkv_weights[v_slice].reshape(-1, hidden_size))

        # attention dense
        o_weight = common_state_dict[f'model.decoder.layers.{l}.self_attention.linear_proj.weight']
        o_weight_base_name = f'model.layers.{l}.self_attn.o_proj.weight'
        checkpoint[o_weight_base_name] = param_to_weights(o_weight)

        # mlp
        mlp_weights = common_state_dict[f'model.decoder.layers.{l}.mlp.linear_fc1.weight']
        mlp_down_proj_weight = mlp_weights[:ffn_hidden_size, :]
        mlp_gate_proj_weight = mlp_weights[ffn_hidden_size:, :]

        mlp_down_proj_base_name = f'model.layers.{l}.mlp.gate_proj.weight'
        mlp_gate_proj_base_name = f'model.layers.{l}.mlp.up_proj.weight'

        checkpoint[mlp_down_proj_base_name] = param_to_weights(mlp_down_proj_weight)
        checkpoint[mlp_gate_proj_base_name] = param_to_weights(mlp_gate_proj_weight)

        mlp_up_proj_weight = common_state_dict[f'model.decoder.layers.{l}.mlp.linear_fc2.weight']
        mlp_up_proj_base_name = f'model.layers.{l}.mlp.down_proj.weight'
        checkpoint[mlp_up_proj_base_name] = param_to_weights(mlp_up_proj_weight)

        # layernorm
        input_ln_weight = common_state_dict[f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight']
        input_ln_base_name = f'model.layers.{l}.input_layernorm.weight'
        checkpoint[input_ln_base_name] = param_to_weights(input_ln_weight)

        post_attn_ln_weight = common_state_dict[f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight']
        post_attn_ln_base_name = f'model.layers.{l}.post_attention_layernorm.weight'
        checkpoint[post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)

        logging.info(f"Layer {l} done!")

    final_ln_weight = common_state_dict[f'model.decoder.final_layernorm.weight']
    final_ln_base_name = f'model.norm.weight'
    checkpoint[final_ln_base_name] = param_to_weights(final_ln_weight)

    output_layer_weight = common_state_dict[f'model.output_layer.weight']
    output_layer_base_name = f'lm_head.weight'
    checkpoint[output_layer_base_name] = param_to_weights(output_layer_weight)

    logging.info("Finihed converting the weights")

    # Convert checkpoint to LlamaModel
    nemo_exported = checkpoint

    # Set the Llama configuration
    logging.info("Loading Llama config")
    config = LlamaConfig()
    config.vocab_size = tokenizer_length
    config.num_attention_heads = nemo_cfg["num_attention_heads"]
    config.num_key_value_heads = num_query_groups
    config.max_position_embeddings = nemo_cfg["max_position_embeddings"]
    config.num_hidden_layers = nemo_cfg["num_layers"]
    config.hidden_size = nemo_cfg["hidden_size"]
    config.intermediate_size = nemo_cfg["ffn_hidden_size"]
    config.head_dim = int(config.hidden_size / config.num_attention_heads)
    config.rms_norm_eps = nemo_cfg["layernorm_epsilon"] # Makes no sense

    # Create empty Llama model
    model = LlamaForCausalLM(config)

    # Resize embedding according to vocabulary size
    model.resize_token_embeddings(tokenizer_length)

    # Load checkpoint into Llama model
    model.load_state_dict(nemo_exported)
    model.to(dtype)

    # Save model
    logging.info("Saving the model and tokenizer...")
    model.save_pretrained(output_hf_path)
    logging.info(f"HuggingFace weights saved to {output_hf_path}")
    fast_tokenizer.save_pretrained(output_hf_path)
    tokenizer.save_pretrained(output_hf_path)
    logging.info(f"Tokenizer saved to {output_hf_path}")
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a NeMo model to HuggingFace format.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the NeMo model.')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for the HuggingFace model.')

    args = parser.parse_args()
    model_path = args.model_path
    output_path = args.output_path

    print(f"Converting model: {model_path}")
    model, tokenizer = convert_from_nemo_file(model_path, output_path)
    print(f"Finished converting the model. Output directory: {output_path}")


if __name__ == "__main__":
    main()
