"""直前情報ページの読み取りのテスト（python -m pytest tests）"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from live import complete, parse_beforeinfo  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def test_beforeinfo_kiryu_9r():
    info = parse_beforeinfo((FIX / "beforeinfo_20261104_01_09.html").read_text(encoding="utf-8"))
    b = info["boats"]
    assert [b[str(w)]["course"] for w in range(1, 7)] == [1, 2, 3, 4, 5, 6]
    assert [b[str(w)]["st"] for w in range(1, 7)] == [0.01, 0.06, 0.02, 0.02, 0.07, 0.06]
    assert [b[str(w)]["time"] for w in range(1, 7)] == [6.73, 6.75, 6.64, 6.72, 6.83, 6.81]
    assert b["4"]["parts"] == ["リング2"]
    assert b["1"]["tilt"] == -0.5 and b["6"]["tilt"] == 0.0
    assert info["windDir"] == 4 and info["wind"] == 2.0 and info["weather"] == 3 and info["wave"] == 1.0
    assert complete(info)
