import sys
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, AutoModelForCausalLM
from datautils import get_loaders
from parsers import parse_args
from torch.ao.quantization import QuantStub, DeQuantStub
import os
from tqdm import tqdm
import model_loader 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import binary_quadratic_network

def find_sets(obj, path='model'):
    if isinstance(obj, set):
        print(f"Found set at {path}: {obj}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            find_sets(v, f'{path}[{i}]')
    elif isinstance(obj, dict):
        for k, v in obj.items():
            find_sets(v, f'{path}[{k}]')
    elif hasattr(obj, '__dict__'):
        for k, v in vars(obj).items():
            find_sets(v, f'{path}.{k}')




if __name__ == '__main__':
    # args = parse_args()
    gs = 128
    step = 50000
    device = "cuda:2" # module.weight.deviceが読み込まれる

    if True:
        for bit_width in [2, 3, 4]:
            # print('bit width:', bit_width)
            # model = model_loader.get_llama(args.model)
            model_path = "Qwen/Qwen2.5-1.5B"
            # model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

            model_name = os.path.basename(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model = binary_quadratic_network.replace_linear_with_bqq(model, weights_dir=f'/work2/k-kuroki/BQQLLM/bqq_compressed_data/{model_name}-{gs}gs-{step}step', bit_width=bit_width, device=device)

            # hadamard transformation version
            # model = binary_quadratic_network.replace_linear_with_hbqq(model, weights_dir=f'/work2/k-kuroki/BQQLLM/bqq_compressed_data/{model_path.split("/")[-1]}-{gs}gs-{step}step', bit_width=bit_width)

            for name, param in model.named_parameters():
                print(f"{name:40s} | shape: {tuple(param.shape)} | learnable: {param.requires_grad}")
            os.makedirs(os.path.dirname(__file__)+f'/quantized_model_data/{model_name}', exist_ok=True)
            torch.save(model, os.path.dirname(__file__)+f'/quantized_model_data/{model_name}/{model_name}-{bit_width}bit-{gs}gs-{step}step.pth')
        
    if False:
        for bit_width in [2, 3]:
            print('bit width:', bit_width)
            model = model_loader.get_llama(args.model)
            model = binary_quadratic_network.replace_weight(model, weights_dir='/work2/k-kuroki/quadratic_quantization/nn_compression/llm/qq_compressed_data/Llama-2-7b-hf-256gs-100000step', bit_width=bit_width)
            model.save_pretrained(os.path.dirname(__file__)+f'/quantized_model_data/{bit_width}bit-bqq-llama2-7b-shape-preserved')
