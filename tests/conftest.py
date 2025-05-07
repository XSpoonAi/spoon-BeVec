import os
import shutil
import pytest
from typing import Generator

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """设置测试环境变量"""
    # 设置测试用的环境变量
    os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "test-api-key")
    
    yield
    
    # 清理测试环境
    if os.path.exists("./test_chroma_db"):
        try:
            shutil.rmtree("./test_chroma_db")
        except PermissionError:
            # 如果文件被占用，忽略错误
            pass

@pytest.fixture
def test_vectors():
    """提供测试用的向量数据"""
    return [
        {"id": "1", "values": [0.1, 0.2, 0.3], "metadata": {"text": "hello"}},
        {"id": "2", "values": [0.4, 0.5, 0.6], "metadata": {"text": "world"}},
        {"id": "3", "values": [0.7, 0.8, 0.9], "metadata": {"text": "test"}},
    ]

@pytest.fixture
def query_vector():
    """提供测试用的查询向量"""
    return [0.1, 0.2, 0.3] 