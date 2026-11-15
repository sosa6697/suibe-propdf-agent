"""把聚合报表注入单文件看板模板，产出可独立打开的 HTML。

两种用法：
- `GET /dashboard`：服务端按当前 analytics.db 实时注入；
- `campus-rules analytics --html var/dashboard_snapshot.html`：导出带数据的静态快照，
  断网、没有服务也能直接双击打开。

页面本身零 CDN、零依赖；注入点是一个固定注释占位符，没有被注入时页面会显示
「未连接数据源」而不是伪造数据。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT

PLACEHOLDER = "/*__ANALYTICS_DATA__*/"
_TEMPLATE = PROJECT_ROOT / "web" / "dashboard.html"


def render_dashboard_html(report: dict[str, Any], *, template: Path | None = None) -> str:
    path = Path(template or _TEMPLATE)
    html = path.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise ValueError(f"看板模板缺少注入占位符 {PLACEHOLDER}：{path}")
    # 防止报表字符串里出现 </script> 提前闭合脚本标签
    payload = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    return html.replace(PLACEHOLDER, f"const DATA = {payload};")
