import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Layers, RotateCcw } from "lucide-react";
import type { Asset, Snapshot, Structure } from "../api";
import { fmt } from "../api";
import { useStore } from "../store";

/* 3D twin: every asset at its true position and depth (x east, y up = elevation, z south).
   Openings come from the twin's line geometry (e.g. a 3D GeoJSON export of the mine design). */

const OPENING_STYLE: Record<string, { color: string; width: number; label: string }> = {
  shaft: { color: "#f4f4f5", width: 3, label: "Shafts" },
  raise: { color: "#8b5cf6", width: 2, label: "Exhaust raises" },
  orepass: { color: "#c97a14", width: 2, label: "Ore passes" },
  decline: { color: "#71717a", width: 1.5, label: "Decline" },
  drive: { color: "#4b4b6a", width: 1.5, label: "Ore drives" },
  access: { color: "#34344c", width: 1, label: "Level accesses" },
  haulage: { color: "#a8640f", width: 2, label: "Haulage drives" },
};
const STATUS = { mucking: "#fe0265", idle: "#8b93a7", closed: "#ef4444" };
const TYPE_COLOR: Record<string, string> = {
  main_fan: "#8b5cf6", vent_district: "#8b5cf6", pump_station: "#3987e5", ug_crusher: "#c97a14", orepass: "#c97a14",
  skip_hoist: "#f4f4f5", stockpile: "#d4aa00", concentrator: "#f4f4f5", conc_sales: "#f4f4f5", site_services: "#71717a",
  lhd_fleet: "#ff65a3", haul_fleet: "#e0a24a",
};
const GROUPS = [
  { id: "openings", label: "Openings" }, { id: "stopes", label: "Stopes" }, { id: "ventilation", label: "Ventilation" },
  { id: "dewatering", label: "Dewatering" }, { id: "processing", label: "Shaft, crushers, passes" },
  { id: "equipment", label: "Mobile equipment" }, { id: "surface", label: "Surface" }, { id: "labels", label: "Labels" },
];
const GROUP_OF: Record<string, string> = {
  stope: "stopes", main_fan: "ventilation", vent_district: "ventilation", pump_station: "dewatering",
  ug_crusher: "processing", orepass: "processing", skip_hoist: "processing", lhd_fleet: "equipment",
  haul_fleet: "equipment", mobile_unit: "equipment", stockpile: "surface", concentrator: "surface",
  conc_sales: "surface", site_services: "surface",
};
const LABELLED = new Set(["main_fan", "vent_district", "ug_crusher", "skip_hoist", "orepass", "concentrator", "stockpile", "pump_station"]);

type ColorBy = "activity" | "grade" | "broken";

function tagValue(snap: Snapshot | null, path: string, suffix: string): number | undefined {
  const v = snap?.assets[path];
  if (!v) return undefined;
  const k = Object.keys(v).find((x) => x.endsWith(suffix));
  return k === undefined ? undefined : v[k];
}

function ramp(t: number, lo: string, hi: string): THREE.Color {
  return new THREE.Color(lo).lerp(new THREE.Color(hi), Math.max(0, Math.min(1, t)));
}

export function Mine3DPage() {
  const structure = useStore((s) => s.structure);
  const snap = useStore((s) => s.snap);
  const selected = useStore((s) => s.selected);
  const set = useStore((s) => s.set);
  const host = useRef<HTMLDivElement>(null);
  const labelHost = useRef<HTMLDivElement>(null);
  const ctx = useRef<{ renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.PerspectiveCamera; controls: OrbitControls;
    groups: Record<string, THREE.Group>; stopes: Map<string, THREE.Mesh>; pickables: THREE.Object3D[]; render: () => void;
    labels: { el: HTMLDivElement; pos: THREE.Vector3; prio: number; group: string }[]; selMarker: THREE.Mesh; home: () => void } | null>(null);
  const [vis, setVis] = useState<Record<string, boolean>>(() => Object.fromEntries(GROUPS.map((g) => [g.id, true])));
  const [colorBy, setColorBy] = useState<ColorBy>("activity");

  const geo = useMemo(() => (structure ? buildGeo(structure) : null), [structure]);

  // ---------------------------------------------------------------- scene
  useEffect(() => {
    if (!host.current || !geo || !structure) return;
    const el = host.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    renderer.setClearColor("#09090b");
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog("#09090b", 4000, 9000);
    const camera = new THREE.PerspectiveCamera(42, 1, 5, 20000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(geo.center);
    const home = () => {
      // oblique view from the east-south-east: the ore lenses (north-south strike) spread across the screen
      camera.position.set(geo.center.x + geo.span * 1.05, geo.center.y + geo.span * 0.42, geo.center.z + geo.span * 0.38);
      controls.target.copy(geo.center);
      controls.update();
    };
    home();
    scene.add(new THREE.AmbientLight("#ffffff", 0.75));
    const sun = new THREE.DirectionalLight("#ffffff", 1.1);
    sun.position.set(1, 2, 1.5);
    scene.add(sun);

    const groups: Record<string, THREE.Group> = {};
    for (const g of GROUPS) { groups[g.id] = new THREE.Group(); scene.add(groups[g.id]); }
    // surface: translucent plane with a grid at the collar elevation
    const surfY = geo.surface;
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(geo.span * 2.4, geo.span * 2.4),
      new THREE.MeshBasicMaterial({ color: "#16161f", transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false }));
    plane.rotation.x = -Math.PI / 2;
    plane.position.set(geo.center.x, surfY, geo.center.z);
    groups.surface.add(plane);
    const grid = new THREE.GridHelper(geo.span * 2.4, 24, "#2e2e44", "#1f1f2e");
    grid.position.set(geo.center.x, surfY + 0.5, geo.center.z);
    groups.surface.add(grid);

    // openings
    for (const o of geo.openings) {
      const st = OPENING_STYLE[o.kind] ?? OPENING_STYLE.access;
      const g = new THREE.BufferGeometry().setFromPoints(o.pts);
      groups.openings.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: st.color, transparent: true, opacity: 0.9 })));
    }
    // assets
    const stopes = new Map<string, THREE.Mesh>();
    const pickables: THREE.Object3D[] = [];
    const sphere = new THREE.SphereGeometry(1, 14, 10);
    const box = new THREE.BoxGeometry(1, 1, 1);
    for (const a of geo.points) {
      const group = GROUP_OF[a.type] ?? (a.pos.y > surfY - 5 ? "surface" : "processing");
      let mesh: THREE.Mesh;
      if (a.type === "stope") {
        mesh = new THREE.Mesh(box, new THREE.MeshLambertMaterial({ color: STATUS.idle }));
        mesh.scale.set(30, 26, 14);
        stopes.set(a.path, mesh);
      } else if (a.type === "mobile_unit") {
        const c = a.path.startsWith("LHD") ? "#ff65a3" : a.path.startsWith("TRK") ? "#e0a24a" : "#a1a1aa";
        mesh = new THREE.Mesh(sphere, new THREE.MeshLambertMaterial({ color: c }));
        mesh.scale.setScalar(7);
      } else {
        const big = ["skip_hoist", "concentrator", "ug_crusher", "main_fan"].includes(a.type);
        mesh = new THREE.Mesh(a.type === "vent_district" ? sphere : box,
          new THREE.MeshLambertMaterial({ color: TYPE_COLOR[a.type] ?? "#a1a1aa", transparent: a.type === "vent_district", opacity: 0.55 }));
        mesh.scale.setScalar(big ? 38 : a.type === "vent_district" ? 12 : 22);
      }
      mesh.position.copy(a.pos);
      mesh.userData = { path: a.path };
      groups[group].add(mesh);
      pickables.push(mesh);
    }
    const selMarker = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14),
      new THREE.MeshBasicMaterial({ color: "#fe0265", wireframe: true, transparent: true, opacity: 0.9 }));
    selMarker.scale.setScalar(40);
    selMarker.visible = false;
    scene.add(selMarker);

    // labels (HTML overlay, decluttered in screen space)
    const labels: { el: HTMLDivElement; pos: THREE.Vector3; prio: number; group: string }[] = [];
    for (const a of geo.points) {
      if (!LABELLED.has(a.type)) continue;
      const d = document.createElement("div");
      d.className = "absolute whitespace-nowrap pointer-events-none text-[10.5px] font-semibold";
      d.style.cssText += "color:#f4f4f5;text-shadow:0 0 4px #000,0 0 2px #000;transform:translate(-50%,-140%)";
      d.textContent = a.label;
      labelHost.current?.appendChild(d);
      labels.push({ el: d, pos: a.pos.clone(), prio: a.type === "vent_district" ? 1 : 3, group: GROUP_OF[a.type] ?? "processing" });
    }
    labels.sort((x, y) => y.prio - x.prio);

    let raf = 0;
    const render = () => {
      raf = 0;
      renderer.render(scene, camera);
      const w = el.clientWidth, h = el.clientHeight;
      const boxes: [number, number, number, number][] = [];
      const showLabels = labelHost.current?.dataset.on === "1";
      for (const l of labels) {
        const p = l.pos.clone().project(camera);
        const vis_ = showLabels && groups[l.group].visible && p.z < 1 && Math.abs(p.x) < 1.05 && Math.abs(p.y) < 1.05;
        if (!vis_) { l.el.style.display = "none"; continue; }
        const x = (p.x + 1) / 2 * w, y = (1 - p.y) / 2 * h;
        const bw = l.el.textContent!.length * 6.2;
        const bx: [number, number, number, number] = [x - bw / 2, y - 26, x + bw / 2, y - 10];
        const hit = boxes.some((b) => bx[0] < b[2] && bx[2] > b[0] && bx[1] < b[3] && bx[3] > b[1]);
        l.el.style.display = hit ? "none" : "block";
        if (!hit) { boxes.push(bx); l.el.style.left = `${x}px`; l.el.style.top = `${y}px`; }
      }
    };
    const schedule = () => { if (!raf) raf = requestAnimationFrame(render); };
    controls.addEventListener("change", schedule);
    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / Math.max(1, h);
      camera.updateProjectionMatrix();
      schedule();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();
    // picking
    const ray = new THREE.Raycaster();
    let down = { x: 0, y: 0 };
    const onDown = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY }; };
    const onUp = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;      // a drag, not a click
      const r = renderer.domElement.getBoundingClientRect();
      ray.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1), camera);
      const hit = ray.intersectObjects(pickables.filter((o) => o.parent?.visible), false)[0];
      if (hit) set({ selected: hit.object.userData.path });
    };
    renderer.domElement.addEventListener("pointerdown", onDown);
    renderer.domElement.addEventListener("pointerup", onUp);
    ctx.current = { renderer, scene, camera, controls, groups, stopes, pickables, render: schedule, labels, selMarker, home };
    return () => {
      ro.disconnect();
      controls.dispose();
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("pointerup", onUp);
      labels.forEach((l) => l.el.remove());
      renderer.dispose();
      el.removeChild(renderer.domElement);
      if (raf) cancelAnimationFrame(raf);
      ctx.current = null;
    };
  }, [geo, structure, set]);

  // ---------------------------------------------------------------- live colours
  const live = useMemo(() => {
    let mucking = 0, idle = 0, closed = 0, tph = 0;
    const out = new Map<string, THREE.Color>();
    if (!geo) return { out, mucking, idle, closed, tph };
    for (const a of geo.points) {
      if (a.type !== "stope") continue;
      const open = tagValue(snap, a.path, "STOPE.OPEN");
      const rate = tagValue(snap, a.path, "LHD.TPH") ?? 0;
      const broken = tagValue(snap, a.path, "CMS.BROKEN") ?? 0;
      const isClosed = open !== undefined && open < 0.5;
      if (isClosed) closed++; else if (rate > 5) { mucking++; tph += rate; } else idle++;
      let c: THREE.Color;
      if (colorBy === "grade") c = ramp((a.grade - 0.4) / 2.0, "#3a0a20", "#ff65a3");
      else if (colorBy === "broken") c = ramp(broken / 60000, "#10233f", "#7cb3f2");
      else c = new THREE.Color(isClosed ? STATUS.closed : rate > 5 ? STATUS.mucking : STATUS.idle);
      if (colorBy === "activity" && rate > 5) c = ramp(0.35 + rate / 400, "#5c0a2c", STATUS.mucking);
      out.set(a.path, c);
    }
    return { out, mucking, idle, closed, tph };
  }, [geo, snap, colorBy]);

  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    live.out.forEach((col, path) => {
      const m = c.stopes.get(path);
      if (m) (m.material as THREE.MeshLambertMaterial).color.copy(col);
    });
    c.render();
  }, [live]);

  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    for (const g of GROUPS) if (c.groups[g.id]) c.groups[g.id].visible = vis[g.id];
    if (labelHost.current) labelHost.current.dataset.on = vis.labels ? "1" : "0";
    c.render();
  }, [vis, geo]);

  useEffect(() => {
    const c = ctx.current;
    if (!c || !geo) return;
    const p = geo.points.find((a) => a.path === selected);
    c.selMarker.visible = !!p;
    if (p) c.selMarker.position.copy(p.pos);
    c.render();
  }, [selected, geo]);

  const fanAir = (geo?.points ?? []).filter((a) => a.type === "main_fan").reduce((s, a) => s + (tagValue(snap, a.path, "FAN.Q") ?? 0), 0);
  const hoist = (geo?.points ?? []).filter((a) => a.type === "skip_hoist").reduce((s, a) => s + (tagValue(snap, a.path, "HOIST.TPH") ?? 0), 0);

  if (!geo) return <div className="p-4 text-dim">Loading the twin…</div>;
  return (
    <div className="absolute inset-0">
      <div ref={host} className="absolute inset-0" />
      <div ref={labelHost} className="absolute inset-0 pointer-events-none overflow-hidden" data-on="1" />
      <div className="absolute top-3 left-3 panel w-[214px] z-10" style={{ background: "#12121aee" }}>
        <div className="panel-h"><span className="flex items-center gap-1.5"><Layers size={12} />Layers</span>
          <button className="btn btn-ghost !p-0.5" title="Reset view" onClick={() => { ctx.current?.home(); ctx.current?.render(); }}><RotateCcw size={12} /></button></div>
        <div className="px-3 py-2 space-y-1.5">
          {GROUPS.map((g) => (
            <label key={g.id} className="flex items-center gap-2 text-[12px] cursor-pointer">
              <input type="checkbox" checked={vis[g.id]} onChange={(e) => setVis({ ...vis, [g.id]: e.target.checked })} className="accent-[#fe0265]" />{g.label}
            </label>
          ))}
          <div className="pt-1.5 border-t border-line mt-1.5 text-[11px] text-dim">Colour stopes by</div>
          <div className="flex gap-1">
            {(["activity", "grade", "broken"] as ColorBy[]).map((c) => (
              <button key={c} className={`btn !py-0.5 !px-1.5 !text-[11px] flex-1 justify-center ${colorBy === c ? "btn-primary" : ""}`} onClick={() => setColorBy(c)}>{c}</button>
            ))}
          </div>
        </div>
      </div>
      <div className="absolute top-3 right-3 panel z-10 px-3 py-2 text-[11.5px] space-y-1 w-[200px]" style={{ background: "#12121aee" }}>
        <div className="text-[10.5px] uppercase tracking-wider text-muted font-semibold">Underground now</div>
        <div className="flex justify-between"><span className="text-muted">stopes mucking</span><span className="num">{live.mucking}</span></div>
        <div className="flex justify-between"><span className="text-muted">open, idle</span><span className="num">{live.idle}</span></div>
        <div className="flex justify-between"><span className="text-muted">closed</span><span className="num">{live.closed}</span></div>
        <div className="flex justify-between"><span className="text-muted">ore mucked</span><span className="num">{fmt.n(live.tph)} t/h</span></div>
        <div className="flex justify-between"><span className="text-muted">main fan air</span><span className="num">{fmt.n(fanAir)} m³/s</span></div>
        <div className="flex justify-between"><span className="text-muted">hoisting</span><span className="num">{fmt.n(hoist)} t/h</span></div>
      </div>
      <div className="absolute bottom-3 left-3 panel z-10 px-3 py-2 text-[11px] space-y-1" style={{ background: "#12121aee" }}>
        <div className="flex gap-3 flex-wrap">
          {colorBy === "activity" ? <>
            <span><span style={{ color: STATUS.mucking }}>■</span> mucking</span><span><span style={{ color: STATUS.idle }}>■</span> open, idle</span>
            <span><span style={{ color: STATUS.closed }}>■</span> closed</span></>
            : <span className="text-muted">{colorBy === "grade" ? "grade: dark = low, bright = high (Cu-eq %)" : "broken stock: dark = empty, bright = full"}</span>}
        </div>
        <div className="flex gap-3 flex-wrap text-dim">
          {Object.entries(OPENING_STYLE).filter(([k]) => geo.kinds.has(k)).map(([k, s]) => (
            <span key={k}><span style={{ color: s.color }}>━</span> {s.label.toLowerCase()}</span>
          ))}
        </div>
        <div className="text-dim">drag to rotate · right-drag to pan · scroll to zoom · click an asset for details</div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- geometry: lat/lon/elevation -> local metres
function buildGeo(st: Structure) {
  const pts3 = st.assets.filter((a) => a.at);
  const lat0 = st.origin?.[0] ?? (pts3[0]?.at?.[0] ?? 0), lon0 = st.origin?.[1] ?? (pts3[0]?.at?.[1] ?? 0);
  const kx = 111320 * Math.cos((lat0 * Math.PI) / 180), ky = 111320;
  const v = (lat: number, lon: number, z: number) => new THREE.Vector3((lon - lon0) * kx, z, -(lat - lat0) * ky);
  const points: { path: string; label: string; type: string; pos: THREE.Vector3; grade: number }[] = [];
  const openings: { kind: string; pts: THREE.Vector3[] }[] = [];
  const kinds = new Set<string>();
  let surface = -Infinity;
  for (const a of st.assets as Asset[]) {
    if (a.line && a.line.length >= 2 && !a.is_link) {
      const kind = a.layer in OPENING_STYLE ? a.layer : "access";
      kinds.add(kind);
      openings.push({ kind, pts: a.line.map((p) => v(p[0], p[1], p[2] ?? 0)) });
      for (const p of a.line) surface = Math.max(surface, p[2] ?? 0);
      continue;
    }
    if (!a.at || a.is_link) continue;
    const z = (a.at as number[])[2] ?? 0;
    surface = Math.max(surface, z);
    points.push({ path: a.path, label: a.label, type: a.type, pos: v(a.at[0], a.at[1], z), grade: a.params?.grade?.value ?? 1 });
  }
  const all = [...points.map((p) => p.pos), ...openings.flatMap((o) => o.pts)];
  const bb = new THREE.Box3().setFromPoints(all.length ? all : [new THREE.Vector3()]);
  const center = bb.getCenter(new THREE.Vector3());
  const size = bb.getSize(new THREE.Vector3());
  return { points, openings, kinds, center, span: Math.max(size.x, size.y, size.z, 500), surface: Number.isFinite(surface) ? surface : 0 };
}
