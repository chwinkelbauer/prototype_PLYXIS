"""Model lint: what compiles but will not do what the engineer thinks."""
import os

from plyxis.lang import compile_model
from plyxis.lang.lint import lint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL = """
define lintpkg as pkg {
  define oil as res [bbl] { rate_unit = "bbl/d"; };
  define well as ent {
    properties:
      q_max = 100 [bbl/d] calibrate in [10, 500];
      forgotten = 3;
    decision:
      speed in [0.2, 1.0] default 0.5;
      typo in [0, 1] default 0.5;
    nodes: out prod <- oil;
    telemetry: rate <- "FLOW";
    behavior:
      rate = q_max * speed;
      prod.oil = rate;
  };
  define sink as ent {
    properties: fee = 1 [USD/bbl];
    state: stock = 10 [bbl] in [0, 1000];
    nodes: in rec <- oil; in spare <- oil;
    behavior:
      stock <- stock + rec.oil * dt;
      cost fees = fee * rec.oil;
      final revenue held = 50 [USD/bbl] * stock;
  };
};
use lintpkg;
define site1 as env {
  contains:
    init W1 as well { tag = "W1"; };
    init W2 as well { };
    init S as sink { };
    connect(W1.prod, S.rec);
    connect(W2.prod, S.rec);
};
define s as sim { env: site1; horizon = 6 [h]; step = 1 [h]; objective: maximize sum(revenue) - sum(cost); };
"""


def _codes(found):
    return {(f.code, f.asset) for f in found}


def test_lint_finds_what_the_compiler_cannot(tmp_path):
    p = tmp_path / "m.plx"
    p.write_text(MODEL)
    m = compile_model([str(p)], env="site1", sim="s")
    found = lint(m)
    codes = _codes(found)
    assert any(f.code == "no-effect" and "typo" in f.message and f.severity == "warning" for f in found)
    assert not any(f.code == "no-effect" and "speed" in f.message for f in found)
    assert any(f.code == "unused-property" and "forgotten" in f.message for f in found)
    assert ("unconnected-input", "S") in codes
    assert ("no-tag", "W2") in codes and ("no-tag", "W1") not in codes
    assert any(f.code == "terminal-value" and "held" in f.message for f in found)


def test_lint_is_quiet_on_the_case_studies():
    for sc in ("kestrel", "corvus"):
        m = compile_model([os.path.join(ROOT, "scenarios", sc, "field.plx")])
        warnings = [f for f in lint(m) if f.severity == "warning"]
        assert not warnings, [f.message for f in warnings]
