# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# 将 Code 目录加入模块搜索路径，确保直接导入 src.*
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
