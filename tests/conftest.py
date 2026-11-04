"""测试默认离线；微型随机模型不读取任何远程权重。"""

import os
import socket

import pytest

# 在测试模块导入第三方库前关闭隐式资源下载。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires explicitly prepared local model or GPU assets")
    # 临时产物留在工作区缓存，兼容系统临时目录不可写的受限 Windows 环境。
    if config.option.basetemp is None:
        cache_root = config.rootpath / ".pytest_cache"
        # 首次克隆尚无缓存目录，须在pytest创建临时子目录前建立其父目录。
        cache_root.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(cache_root / f"tmp-{os.getpid()}")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """把意外联网变成明确失败，避免缓存存在时掩盖测试依赖。"""
    def reject_connection(*args, **kwargs):
        raise AssertionError("Network access is forbidden in the test suite")

    monkeypatch.setattr(socket.socket, "connect", reject_connection)
    monkeypatch.setattr(socket, "create_connection", reject_connection)


@pytest.fixture
def tiny_causal_model():
    """使用与正式骨干同族的随机小模型验证真实因果和 KV 行为。"""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(17)
    torch.set_num_threads(1)
    config = Qwen2Config(
        vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
