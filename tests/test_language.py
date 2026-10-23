"""PLX language: lexer, parser, units, elaboration, diagnostics."""
import os

import pytest

from plyxis.lang import check_source, compile_model
from plyxis.lang import ast as A
from plyxis.lang import units as U
from plyxis.lang.errors import Diagnostics, PlxError
from plyxis.lang.lexer import tokenize
from plyxis.lang.parser import parse

from conftest import KESTREL, MODELS, ROOT


def test_lexer_numbers_ranges_ops():
    toks, _ = tokenize("for i in 1..10 { x <- 1.5e-3 [bbl/d]; a <= b; }")
    kinds = [(t.kind, t.value) for t in toks[:6]]
    assert kinds[0] == ("ident", "for")
    assert ("number", "1") in [(t.kind, t.value) for t in toks]
    assert any(t.is_op("..") for t in toks)
    assert any(t.is_op("<-") for t in toks)
    assert any(t.is_op("<=") for t in toks)


def test_parser_templates_units_and_sections():
    src = """
    define p as pkg {
      define w as ent {
        properties: q = 900 [bbl/d] calibrate in [100, 2000];
        decision: s in [0.3, 1] default 0.8 hold 2; on : bool;
        nodes: out prod <- oil;
        behavior: prod.oil = q * s * on; cost c = 1.5 [USD/bbl] * prod.oil;
      };
    };
    define e as env { contains: for i in 1..3 { init w{i} as w { q = 100 * i; }; } };
    """
    d = Diagnostics()
    prog = parse(src, "t.plx", d)
    assert not d.has_errors(), [x.message for x in d.items]
    w = prog.items[0].items[0]
    assert w.params[0].unit == "bbl/d" and w.params[0].calibrate is not None
    assert w.decisions[1].domain == "bool"
    term = w.behavior[1]
    assert isinstance(term, A.TermStmt) and term.kind == "cost"
    loop = prog.items[1].contains[0]
    assert isinstance(loop, A.ForStmt)
    assert isinstance(loop.body[0].name, A.Template)


def test_parser_error_recovery_reports_all_errors():
    src = "define e as ent { behavior: x = ; y = 2; z = (; };"
    d = Diagnostics()
    parse(src, "bad.plx", d)
    errs = [x for x in d.items if x.severity == "error"]
    assert len(errs) >= 2
    assert all(x.loc.line == 1 for x in errs)


def test_legacy_draft_gives_located_diagnostics():
    legacy = os.path.join(os.path.dirname(ROOT), "proto1_ag", "str3")
    if not os.path.exists(legacy):
        pytest.skip("legacy draft not available")
    d = Diagnostics()
    parse(open(legacy, encoding="utf-8").read(), "str3", d)
    assert any("legacy" in x.message or "unexpected" in x.message for x in d.items)


def test_units_conversion_and_dims():
    assert U.to_canonical(24, "bbl/d") == pytest.approx(0.158987294928, rel=1e-9)
    assert U.from_canonical(U.to_canonical(1500, "Mscf/d"), "Mscf/d") == pytest.approx(1500)
    f, d = U.parse_unit("kW/(bbl/d)")
    assert d == U.dims_div(U.dims_of("kW"), U.dims_of("bbl/d"))
    assert U.dims_equal(U.dims_of("bbl/d"), U.dims_of("m3/h"))
    assert not U.dims_equal(U.dims_of("bbl"), U.dims_of("kWh"))


def test_unit_mismatch_warning():
    src = """
    define e as ent {
      properties: a = 5 [bbl/d]; b = 3 [kW];
      nodes: out o <- *;
      behavior: x = a + b;
    };
    define s as env { contains: init q as e; };
    """
    diags = check_source(["u.plx"], sources={"u.plx": src})
    assert any(x["code"] == "W-UNIT" for x in diags)


def test_algebraic_loop_is_rejected():
    src = """
    define r as res;
    define a as ent { nodes: in i <- r; out o <- r; behavior: o.r = 0.5 * i.r + 1; };
    define s as env { contains: init x as a; init y as a; connect(x.o, y.i); connect(y.o, x.i); };
    """
    diags = check_source(["loop.plx"], sources={"loop.plx": src})
    assert any(x["code"] == "E-LOOP" for x in diags if x["severity"] == "error")


def test_elaboration_tables_composites_via_and_wildcards(kestrel_model):
    flat = kestrel_model.flat
    wells = [i for i in flat.instances if i.type.short == "esp_well"]
    assert len(wells) == 96
    links = [i for i in flat.instances if i.is_link]
    assert len(links) >= 30 and all(l.line for l in links if l.type.short == "flowline")
    lane = flat.instance("LANE_S_OIL")
    ports = {p: flat.ports[pid].resources for p, pid in lane.ports.items()}
    assert [r.split(".")[-1] for r in ports["cargo"]] == ["oil"]          # wildcard specialised to oil
    assert [r.split(".")[-1] for r in ports["drop"]] == ["oil"]           # truck_hours not leaked into cargo
    w = flat.instance("P01_W01")
    assert w.params["q_max"] == pytest.approx(U.to_canonical(453.9, "bbl/d"), rel=1e-6)


def test_environment_inheritance_and_set_overlay():
    m = compile_model([os.path.join(KESTREL, "field.plx"), os.path.join(KESTREL, "truth.plx")],
                      env="kestrel_truth")
    flat = m.flat
    assert flat.instance("P07_W11") is not None                           # added by the child environment
    assert flat.instance("P01_W01").params["q_max"] == pytest.approx(U.to_canonical(527.1, "bbl/d"), rel=1e-6)


def test_composite_entities_flatten_to_junctions():
    m = compile_model([os.path.join(MODELS, "stress.plx")], include_stdlib=False)
    flat = m.flat
    assert any(i.is_junction for i in flat.instances)
    assert flat.instance("pp.p1") is not None and flat.instance("pp.p2") is not None


def test_unknown_names_have_suggestions():
    src = """
    define r as res;
    define a as ent { properties: q_max = 1; nodes: out o <- r; behavior: o.r = q_mx; };
    define s as env { contains: init x as a; };
    """
    diags = check_source(["n.plx"], sources={"n.plx": src})
    err = [x for x in diags if x["code"] == "E-NAME"]
    assert err and "q_max" in err[0]["hint"]
