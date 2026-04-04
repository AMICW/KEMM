"""应用层 ship 主线入口。

这个文件故意保持很薄，只做两件事：

1. 重新导出 ship 主线最常用的几个函数，兼顾旧导入路径和新架构入口
2. 提供 `python -m apps.ship_runner` 这样的统一运行方式

真实业务逻辑仍然分别位于：

- `ship_simulation.main_demo`
- `ship_simulation.run_report`
"""

from ship_simulation.main_demo import random_search, run_demo, select_demo_solution
from ship_simulation.run_report import generate_report

__all__ = [
    "generate_report",
    "random_search",
    "run_demo",
    "select_demo_solution",
]


def main() -> None:
    """运行默认 ship demo。

    这里选择 `show_animation=False`，是为了让命令行入口在无图形环境或自动化测试里
    也能稳定执行。需要动画时，可以直接调用 `run_demo(..., show_animation=True)`。
    """

    run_demo(show_animation=False)


if __name__ == "__main__":
    main()
