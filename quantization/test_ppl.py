import torch
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import numpy as np
import argparse
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_FILE = os.path.join(BASE_DIR, "test-00000-of-00001.parquet")
# ================== 1. 配置部分 ==================
parser = argparse.ArgumentParser(description="计算模型困惑度")
parser.add_argument("--model_path", type=str, required=True, help="模型路径 (例如: /root/models/Qwen2.5-7B-Instruct)")
args = parser.parse_args()

MODEL_PATH = args.model_path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_FILE = os.path.join(BASE_DIR, "test-00000-of-00001.parquet")
MAX_SAMPLES = 200 

# ================== 2. 加载模型 ==================
print(f"正在通过 vLLM 加载模型: {MODEL_PATH}")
try:
    llm = LLM(
        model=MODEL_PATH, 
        trust_remote_code=True, 
        tensor_parallel_size=1, 
        dtype="bfloat16"
    )
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

# ================== 3. 加载数据 (Parquet) ==================
print(f"正在加载数据: {PARQUET_FILE}")
texts = []
try:
    df = pd.read_parquet(PARQUET_FILE)
    target_col = next((col for col in ['text', 'content', 'page'] if col in df.columns), df.columns[0])
    
    raw_data = df[target_col].tolist()
    for line in raw_data:
        if isinstance(line, str):
            line = line.strip()
            # 过滤掉极短的文本，确保至少有两个 token（因为第一个 token 不参与计算）
            if len(line) > 5 and not line.startswith('='):
                texts.append(line)
    
    if MAX_SAMPLES > 0:
        texts = texts[:MAX_SAMPLES]
    print(f"数据加载完毕！共提取 {len(texts)} 条有效文本。")
except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

# ================== 4. 计算困惑度 (完全对齐论文逻辑) ==================
print("开始计算困惑度 (正在对齐 Block-wise 全局聚合逻辑)...")

sampling_params = SamplingParams(
    max_tokens=1, 
    prompt_logprobs=1, 
    temperature=0,
    top_p=1.0
)

#nlls 累加
total_nll_sum = 0.0  
#(nsamples * seqlen)
total_valid_tokens = 0 
processed_count = 0
batch_size = 10 

try:
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        outputs = llm.generate(batch_texts, sampling_params)
        
        for output in outputs:
            if output.prompt_logprobs:
                # 1. 提取所有有效的 logprobs（忽略第一个 None，因为第一个 token 无法计算 PPL）
                logprobs_list = []
                for token_data in output.prompt_logprobs:
                    if token_data is None:
                        continue
                    
                    if isinstance(token_data, dict):
                        # 获取实际被选中的 token 的 logprob
                        logprob_val = next(iter(token_data.values())).logprob
                    else:
                        logprob_val = token_data.logprob
                    logprobs_list.append(logprob_val)
                
                if logprobs_list:
                    # 2. 累加负对数似然 (NLL = -log_prob)
                    total_nll_sum += -sum(logprobs_list)
                    
                    # 3. 统计总 token 数
                    total_valid_tokens += len(logprobs_list)
                    processed_count += 1

    if total_valid_tokens > 0:
        # 4. 计算最终 PPL: exp( 总负对数似然 / 总 Token 数 )
        avg_nll = total_nll_sum / total_valid_tokens
        perplexity = np.exp(avg_nll)
        
        print("-" * 30)
        print(f"有效样本句数: {processed_count}")
        print(f"参与计算的 Token 总数: {total_valid_tokens}")
        print(f"平均负对数似然 (NLL): {avg_nll:.4f}")
        print(f"困惑度 (Perplexity): {perplexity:.4f}")
        print("-" * 30)
    else:
        print("错误：未检测到有效数据进行计算。")

except Exception as e:
    print(f"计算过程中出错: {e}")