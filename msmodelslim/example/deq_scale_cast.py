import os
import sys
import shutil
import argparse
import torch
import torch_npu
from safetensors import safe_open
from safetensors.torch import save_file
from msmodelslim import logger

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, ".."))
sys.path.append(parent_directory)

from example.common.security.path import get_valid_read_path, get_write_directory, get_valid_write_path

SUPPORTED_EXTENSIONS = {'.json', '.py'}
MAX_FILE_NUM = 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Creating quant weights ")
    parser.add_argument("--model_path", type=str, help="Quantied safetensors file path")
    parser.add_argument("--save_directory", type=str, help="The path to save processed quant weights")
    return parser.parse_args()


def cast_deq_scale_to_int64(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a quantized tensor's scale parameter to int64 format.

    Args:
        tensor (torch.Tensor): Input quantized tensor containing scale parameters. 
                               Should be compatible with NPU operations.

    Returns:
        torch.Tensor: Processed tensor in int64 format containing quantization scale information,
                      transferred back to CPU memory.
    """
    processed_tensor = torch_npu.npu_trans_quant_param(tensor.npu()).cpu()
    return processed_tensor


def process_safetensors_file(file_path: str, save_path: str):
    """
    Processes a safetensors file by converting specific dequantization scale tensors from float32 to int64 format.
    
    This function reads a safetensors file, identifies tensors that contain dequantization scale parameters,
    converts them to a more efficient int64 representation, and saves the modified tensors to a new file while
    preserving all original metadata and non-scale tensors.

    Args:
        file_path (str): Path to the input safetensors file to be processed
        save_path (str): Path where the processed safetensors file will be saved
    """
    tensors = {}
    metadata = {}
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            if hasattr(f, 'metadata'):
                metadata = f.metadata()
            for key in keys:
                tensor = f.get_tensor(key)
                if "deq_scale" in key and tensor.dtype == torch.float32:
                    processed_tensor = cast_deq_scale_to_int64(tensor)
                    tensors[key] = processed_tensor
                else:
                    tensors[key] = tensor
        save_file(tensors, save_path, metadata=metadata)
    except Exception as e:
        raise RuntimeError(f"Error processing {file_path}: {e}") from e


def copy_config_files(model_path: str, save_directory: str):
    """
    Copies configuration files from a source model directory to a destination directory.
    
    This function selectively copies configuration files with specific extensions from the source
    model directory to the target save directory. It includes safety checks for file count limits
    and sets secure file permissions on the copied files.

    Args:
        model_path (str): Source directory path containing the configuration files to be copied
        save_directory (str): Destination directory path where files will be copied to
    """
    filenames = os.listdir(model_path)
    if len(filenames) > MAX_FILE_NUM:
        raise ValueError(
            f"The file num in dir is {len(filenames)}, which exceeds the limit {MAX_FILE_NUM}."
        )
    for filename in filenames:
        filename = os.path.basename(filename)
        _, ext = os.path.splitext(filename)
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        src_filepath = get_valid_read_path(os.path.join(model_path, filename))
        dest_filepath = get_valid_write_path(os.path.join(save_directory, filename))
        shutil.copyfile(src_filepath, dest_filepath)
        os.chmod(dest_filepath, 0o600)


def process_safetensors(model_path: str, save_directory: str):
    """
    Processes all safetensors files in a model directory and copying configuration files.
    
    This function serves as the main entry point for processing model files. It discovers all safetensors files
    in the specified model directory, processes each one to convert dequantization scale parameters, and then
    copies relevant configuration files to the destination directory.

    Args:
        model_path (str): Source directory containing the model files (.safetensors) to be processed
        save_directory (str): Target directory where processed files and configurations will be saved
    """
    file_extension = ".safetensors"
    safetensors_files = []
    save_files = []
    for file in os.listdir(model_path):
        file = os.path.basename(file)
        if file.endswith(file_extension):
            safetensors_files.append(os.path.join(model_path, file))
            save_files.append(os.path.join(save_directory, file))
    if not safetensors_files:
        raise RuntimeError(f"No safetensors files found in: {model_path}")
    logger.info(f"Found {len(safetensors_files)} safetensors files to process")
    for i, file_path in enumerate(safetensors_files):
        logger.info(f"Processing: {file_path}")
        process_safetensors_file(file_path, save_files[i])
    copy_config_files(model_path, save_directory)


if __name__ == "__main__":
    args = parse_args()
    args.model_path = get_valid_read_path(args.model_path, is_dir=True, check_user_stat=True)
    args.save_directory = get_write_directory(args.save_directory, write_mode=0o750)
    try:
        process_safetensors(args.model_path, args.save_directory)
        logger.info("Processed weights saved successfully.")
    except Exception as e:
        logger.error(f"Process weights failed. Error detail: {e}")
