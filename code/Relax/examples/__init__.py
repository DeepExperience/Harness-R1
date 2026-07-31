# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# 使用 pkgutil 扩展 examples 命名空间，合并多个路径下的 examples 子包
from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)
