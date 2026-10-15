"""
配置管理模块
从 .env 文件加载 API 配置
"""
import os
from pathlib import Path


def _find_env_file(start_dir=None):
    """从指定目录开始向上查找 .env 文件，优先使用更靠近代码目录的配置。"""
    current_dir = Path(start_dir or Path(__file__).resolve().parent).resolve()

    for directory in (current_dir, *current_dir.parents):
        env_path = directory / ".env"
        if env_path.is_file():
            return env_path

        if (directory / ".git").exists():
            break

    return None


# 尝试加载 python-dotenv
try:
    from dotenv import load_dotenv
    # 加载 .env 文件（优先 code/.env，找不到时向上查找父目录）
    env_path = _find_env_file()
    if env_path is not None:
        load_dotenv(env_path)
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("提示: 未安装 python-dotenv，将直接从环境变量读取配置")
    print("建议运行: pip install python-dotenv")


# API 配置
class Config:
    """统一配置管理"""

    # SiliconFlow 配置
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "YOUR_API_KEY")
    SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

    # 阿里云 DashScope 配置
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "YOUR_API_KEY")
    DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # OpenAI 配置（可选）
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # D2 语义分块：默认本地向量模型，API 模式使用独立凭据，不复用聊天接口。
    D2_EMBEDDING_PROVIDER = os.getenv("D2_EMBEDDING_PROVIDER", "local")
    D2_EMBEDDING_API_KEY = os.getenv("D2_EMBEDDING_API_KEY", "")
    D2_EMBEDDING_BASE_URL = os.getenv("D2_EMBEDDING_BASE_URL", "")
    D2_EMBEDDING_MODEL = os.getenv("D2_EMBEDDING_MODEL", "")

    # D3 RAGAS 评测配置；未单独设置时回退到 SiliconFlow。
    RAGAS_API_KEY = os.getenv("RAGAS_API_KEY", "")
    RAGAS_BASE_URL = os.getenv("RAGAS_BASE_URL", "")
    RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "deepseek-ai/DeepSeek-V3")
    RAGAS_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "")

    @classmethod
    def _is_configured_value(cls, value: str) -> bool:
        if not isinstance(value, str):
            return False
        stripped = value.strip()
        if not stripped:
            return False
        lowered = stripped.lower()
        return stripped != "YOUR_API_KEY" and not lowered.startswith("your_")

    @classmethod
    def check_api_key(cls, key_name="SILICONFLOW_API_KEY"):
        """检查 API Key 是否已配置"""
        key = getattr(cls, key_name)
        if isinstance(key, str):
            key = key.strip()
        if not cls._is_configured_value(key):
            print(f"⚠️  警告: {key_name} 未配置！")
            print(f"请在 code/.env 或仓库根目录 .env 文件中设置 {key_name}")
            return False
        return True

    @classmethod
    def require_api_key(cls, key_name="SILICONFLOW_API_KEY"):
        """在创建模型或数据库前校验 Key，失败时给出中文提示。"""
        if not cls.check_api_key(key_name):
            raise RuntimeError(f"{key_name} 未配置，演示已停止")
        return getattr(cls, key_name).strip()

    @classmethod
    def get_siliconflow_config(cls):
        """获取 SiliconFlow 配置"""
        return {
            "base_url": cls.SILICONFLOW_BASE_URL,
            "api_key": cls.SILICONFLOW_API_KEY,
        }

    @classmethod
    def get_dashscope_config(cls):
        """获取 DashScope 配置"""
        return {
            "base_url": cls.DASHSCOPE_BASE_URL,
            "api_key": cls.DASHSCOPE_API_KEY,
        }

    @classmethod
    def get_ragas_evaluator_config(cls):
        """获取 RAGAS 评测所需的 OpenAI 兼容配置，不输出密钥。"""
        api_key = (
            cls.RAGAS_API_KEY
            if cls._is_configured_value(cls.RAGAS_API_KEY)
            else cls.SILICONFLOW_API_KEY
        )
        base_url = cls.RAGAS_BASE_URL or cls.SILICONFLOW_BASE_URL
        missing = []
        if not cls._is_configured_value(api_key):
            missing.append("RAGAS_API_KEY 或 SILICONFLOW_API_KEY")
        if not cls.RAGAS_EMBEDDING_MODEL:
            missing.append("RAGAS_EMBEDDING_MODEL")
        return {
            "api_key": api_key,
            "base_url": base_url,
            "llm_model": cls.RAGAS_LLM_MODEL,
            "embedding_model": cls.RAGAS_EMBEDDING_MODEL,
            "missing": missing,
        }


if __name__ == "__main__":
    # 测试配置
    print("当前配置状态:")
    print(f"SILICONFLOW_API_KEY: {'已配置' if Config.SILICONFLOW_API_KEY != 'YOUR_API_KEY' else '未配置'}")
    print(f"DASHSCOPE_API_KEY: {'已配置' if Config.DASHSCOPE_API_KEY != 'YOUR_API_KEY' else '未配置'}")
