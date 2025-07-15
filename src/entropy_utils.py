import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

class EntropyMasker:
    """
    一个用于生成和管理熵 mask 的单例类。
    在整个项目中，应该只使用本文件末尾创建的 `entropy_masker` 实例。
    """
    def __init__(self):
        """
        初始化。不做任何重量级操作，仅设置占位符。
        应在之后调用 .configure() 方法来设置参数和加载模型。
        """
        self.model = None
        self.tokenizer = None
        self.device = None
        self.fixed_threshold = None
        self.percentile_threshold = None
        self.current_mask: Optional[torch.Tensor] = None

    def configure(
        self,
        model_name: str, 
        device: Optional[str] = None,
        fixed_threshold: Optional[float] = None,
        percentile_threshold: Optional[float] = None
    ):
        """
        配置 masker，加载模型、Tokenizer 并设置一个或多个熵阈值。

        Args:
            model_name (str): 要加载的 Hugging Face 模型名称或路径。
            device (str, optional): 指定运行模型的设备 ('cuda', 'cpu', etc.)。
                                    如果为 None，则自动检测。
            fixed_threshold (Optional[float]): 固定的熵值阈值。熵值大于此值的 token 会被考虑。
                                               如果为 None，则不使用此标准。
            percentile_threshold (Optional[float]): 百分位阈值。熵值排在前 (1-value)*100% 的 token 会被考虑。
                                                     例如, 0.8 意味着考虑熵最高的20%的token。
                                                     如果为 None，则不使用此标准。
        """
        print(f"正在配置 EntropyMasker 并加载模型: {model_name}...")
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.model.eval() # 确保模型处于评估模式
            print(f"模型已成功加载到设备: {self.device}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("请确保您已安装 'transformers' 库并且可以访问 Hugging Face Hub。")
            raise

        # 配置阈值
        if fixed_threshold is None and percentile_threshold is None:
            raise ValueError("必须提供至少一个阈值 ('fixed_threshold' 或 'percentile_threshold')。")
        if percentile_threshold is not None and not (0.0 <= percentile_threshold <= 1.0):
            raise ValueError("percentile_threshold 必须在 0.0 和 1.0 之间。")
            
        self.fixed_threshold = fixed_threshold
        self.percentile_threshold = percentile_threshold
        print(f"熵阈值已配置: fixed={self.fixed_threshold}, percentile={self.percentile_threshold}")

    @torch.no_grad()
    def generate_and_set_mask(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算、设置并返回熵 mask。
        如果同时配置了两种阈值，则返回它们的交集。

        Args:
            token_ids (torch.Tensor): 输入的 token ID 序列，形状应为 [1, seq_len] 或 [seq_len]。

        Returns:
            torch.Tensor: 一个布尔类型的 mask 张量，形状与输入的 token_ids 完全相同。
                          True 表示高熵位置，False 表示低熵位置。
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("必须先调用 .configure() 方法才能生成 mask。")

        # 1. 获取模型输出的 logits
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        token_ids = token_ids.to(self.device)
        
        outputs = self.model(token_ids)
        # Logits 的形状为 [batch_size, sequence_length, vocab_size]
        # 每个位置的 logit 对应于对下一个 token 的预测，我们用它来评估当前位置 token 的不确定性
        logits = outputs.logits

        # 2. 计算熵
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        
        # 3. 根据类实例的阈值配置生成 mask
        final_mask = torch.ones_like(entropy, dtype=torch.bool)
        
        if self.fixed_threshold is not None:
            final_mask &= (entropy > self.fixed_threshold)
            
        if self.percentile_threshold is not None:
            quantile_value = torch.quantile(entropy.to(torch.float32), self.percentile_threshold)
            final_mask &= (entropy >= quantile_value)
        
        # 4. 设置 (set) 内部 mask 并返回
        self.current_mask = final_mask
        return self.current_mask

    def get_mask(self) -> Optional[torch.Tensor]:
        """
        获取最近一次生成的 mask。

        Returns:
            Optional[torch.Tensor]: 返回存储的 mask，如果还未生成则为 None。
        """
        return self.current_mask

# ==============================================================================
# 全局单例
# ==============================================================================
# 在整个项目中，都应该导入并使用这个实例
entropy_masker = EntropyMasker()


# ==============================================================================
# 示例用法
# ==============================================================================
if __name__ == '__main__':
    model_name = "gpt2"
    
    # 准备一个输入序列
    # 注意：tokenizer 需要在 configure 之后才能使用
    temp_tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "The quick brown fox jumps over the lazy dog. This sentence contains all letters of the alphabet."
    token_ids = temp_tokenizer.encode(text, return_tensors="pt")
    seq_len = token_ids.shape[1]
    
    # --- 示例 1: 仅使用百分比阈值 ---
    print(f"\n--- 示例 1: 配置并使用 'percentile' 阈值 ---")
    entropy_masker.configure(
        model_name=model_name,
        percentile_threshold=0.8
    )
    
    mask1 = entropy_masker.generate_and_set_mask(token_ids)
    print(f"Token 序列长度: {seq_len}")
    print(f"生成的 Mask 形状: {mask1.shape}")
    print(f"被标记为高熵的 Token 数量: {mask1.sum().item()}")
    print(f"高熵 Token 的比例: {mask1.sum().item() / seq_len:.2%}")

    # 使用 get_mask() 验证
    retrieved_mask1 = entropy_masker.get_mask()
    print(f"通过 get_mask() 获取的数量是否一致: {retrieved_mask1.sum().item() == mask1.sum().item()}")


    # --- 示例 2: 重新配置并仅使用固定阈值 ---
    print(f"\n--- 示例 2: 重新配置并使用 'fixed' 阈值 ---")
    entropy_masker.configure(
        model_name=model_name,
        fixed_threshold=2.5
    )
    mask2 = entropy_masker.generate_and_set_mask(token_ids)
    print(f"生成的 Mask 形状: {mask2.shape}")
    print(f"被标记为高熵的 Token 数量: {mask2.sum().item()}")


    # --- 示例 3: 重新配置并同时使用两种阈值 ---
    print(f"\n--- 示例 3: 重新配置并同时使用 'percentile' (top 50%) 和 'fixed' (>2.0) 阈值 ---")
    entropy_masker.configure(
        model_name=model_name,
        percentile_threshold=0.5, # 保留熵最高的 50%
        fixed_threshold=2.0        # 并且熵必须大于 2.0
    )
    mask3 = entropy_masker.generate_and_set_mask(token_ids)
    print(f"生成的 Mask 形状: {mask3.shape}")
    print(f"被标记为高熵的 Token 数量: {mask3.sum().item()}")
