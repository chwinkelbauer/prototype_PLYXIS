import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { Layers } from "lucide-react";
import type { Asset, Snapshot, Structure } from "../api";
import { fmt } from "../api";
import { useStore } from "../store";

const BASEMAPS: Record<string, { label: string; tiles?: string[]; overlay?: string[]; attribution?: string }> = {
  dark: { label: "Dark (Esri)", tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
    overlay: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}"],
    attribution: "Esri, HERE, Garmin, © OpenStreetMap contributors" },
  satellite: { label: "Satellite (Esri)", tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
    attribution: "Imagery © Esri, Maxar, Earthstar Geographics" },
  none: { label: "None (offline)" },
};

const TYPE_COLOR: Record<string, string> = {
  battery: "#fe0265", compressor_station: "#8b5cf6", flare: "#d68519", swd_well: "#3987e5", oil_sales: "#f4f4f5",
  gas_sales: "#f4f4f5", truck_fleet: "#ff65a3", workover_rig: "#a78bfa", field_crew: "#7cb3f2", pad: "#71717a",
};
// subsystems: the validated categorical slots first, then lighter/darker steps of the same hues
const SUB_COLORS = ["#fe0265", "#3987e5", "#c97a14", "#8b5cf6", "#199e70", "#ff65a3", "#7cb3f2", "#e0a24a", "#a78bfa",
  "#3fc495", "#c9024f", "#2f6fbf", "#a8640f", "#6d43d6", "#137a57", "#ffa3c6"];

const GROUPS = [
  { id: "wells", label: "Wells" }, { id: "pads", label: "Pads" }, { id: "facilities", label: "Facilities" },
  { id: "pipelines", label: "Pipelines" }, { id: "lanes", label: "Truck lanes" }, { id: "trucks", label: "Trucks" },
  { id: "crews", label: "Rigs & crews" }, { id: "labels", label: "Labels" },
];

function wellStatus(a: Asset, snap: Snapshot | null): { status: string; rate: number; vib: number } {
  const v = snap?.assets[a.path] ?? {};
  const key = (s: string) => Object.keys(v).find((k) => k.endsWith(s));
  const run = key("ESP.RUN") ? v[key("ESP.RUN")!] : undefined;
  const rate = key("FLOW.LIQ") ? v[key("FLOW.LIQ")!] : 0;
  const vib = key("ESP.VIB") ? v[key("ESP.VIB")!] : 0;
  const status = run === undefined ? "unknown" : run < 0.5 ? "down" : vib >= 5.5 ? "degraded" : "running";
  return { status, rate, vib };
}

export function MapPage() {
  const structure = useStore((s) => s.structure);
  const snap = useStore((s) => s.snap);
  const set = useStore((s) => s.set);
  const selected = useStore((s) => s.selected);
  const ref = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const markers = useRef<maplibregl.Marker[]>([]);
  const [basemap, setBasemap] = useState<string>(() => (localStorage.getItem("plx.basemap2") ?? "dark"));
  const [vis, setVis] = useState<Record<string, boolean>>({ wells: true, pads: true, facilities: true, pipelines: true, lanes: true, trucks: true, crews: true, labels: true });
  const [colorBy, setColorBy] = useState<"status" | "subsystem">("status");
  const [ready, setReady] = useState(false);

  // ---------------------------------------------------------------- init map
  useEffect(() => {
    if (!ref.current || !structure) return;
    const bm = BASEMAPS[basemap] ?? BASEMAPS.none;
    const style: maplibregl.StyleSpecification = {
      version: 8,
      sources: {
        ...(bm.tiles ? { base: { type: "raster", tiles: bm.tiles, tileSize: 256, attribution: bm.attribution, maxzoom: 16 } } : {}),
        ...(bm.overlay ? { ref: { type: "raster", tiles: bm.overlay, tileSize: 256, maxzoom: 16 } } : {}),
      } as any,
      layers: [{ id: "bg", type: "background", paint: { "background-color": "#09090b" } },
        ...(bm.tiles ? [{ id: "base", type: "raster", source: "base", paint: { "raster-opacity": basemap === "satellite" ? 0.85 : 0.9 } } as maplibregl.LayerSpecification] : []),
        ...(bm.overlay ? [{ id: "ref", type: "raster", source: "ref", paint: { "raster-opacity": 0.75 } } as maplibregl.LayerSpecification] : [])],
    };
    const pts = structure.assets.filter((a) => a.at).map((a) => a.at!) as [number, number][];
    const lats = pts.map((p) => p[0]), lons = pts.map((p) => p[1]);
    const m = new maplibregl.Map({
      container: ref.current, style, attributionControl: { compact: true },
      bounds: [[Math.min(...lons) - 0.02, Math.min(...lats) - 0.02], [Math.max(...lons) + 0.02, Math.max(...lats) + 0.02]],
    });
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    m.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
    map.current = m;
    m.on("load", () => {
      m.addSource("links", { type: "geojson", data: linksGeo(structure) });
      m.addSource("edges", { type: "geojson", data: edgesGeo(structure) });
      m.addSource("points", { type: "geojson", data: pointsGeo(structure, null, colorBy) });
      m.addSource("trucks", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      m.addSource("jobs", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      m.addLayer({ id: "edges", type: "line", source: "edges", paint: { "line-color": "#33334a", "line-width": 1, "line-opacity": 0.8 } });
      m.addLayer({ id: "pipelines", type: "line", source: "links", filter: ["==", ["get", "kind"], "pipe"],
        paint: { "line-color": ["get", "color"], "line-width": ["interpolate", ["linear"], ["zoom"], 10, 1.2, 14, 3], "line-opacity": 0.85 } });
      m.addLayer({ id: "lanes", type: "line", source: "links", filter: ["==", ["get", "kind"], "lane"],
        paint: { "line-color": "#ff65a3", "line-width": 1.6, "line-dasharray": [2, 2], "line-opacity": 0.8 } });
      m.addLayer({ id: "pads", type: "circle", source: "points", filter: ["==", ["get", "type"], "pad"],
        paint: { "circle-radius": 9, "circle-color": "#12121a", "circle-stroke-color": "#71717a", "circle-stroke-width": 1.4 } });
      m.addLayer({ id: "wells", type: "circle", source: "points", filter: ["==", ["get", "group"], "wells"],
        paint: { "circle-radius": ["interpolate", ["linear"], ["get", "rate"], 0, 2.5, 600, 4, 2500, 7],
          "circle-color": ["get", "color"], "circle-stroke-color": "#09090b", "circle-stroke-width": 0.8 } });
      m.addLayer({ id: "facilities", type: "circle", source: "points", filter: ["==", ["get", "group"], "facilities"],
        paint: { "circle-radius": 8, "circle-color": ["get", "color"], "circle-stroke-color": "#09090b", "circle-stroke-width": 2, "circle-opacity": 0.95 } });
      m.addLayer({ id: "crews", type: "circle", source: "points", filter: ["==", ["get", "group"], "crews"],
        paint: { "circle-radius": 5, "circle-color": "#7cb3f2", "circle-stroke-color": "#09090b", "circle-stroke-width": 1.5 } });
      m.addLayer({ id: "jobs", type: "circle", source: "jobs",
        paint: { "circle-radius": 13, "circle-color": "transparent", "circle-stroke-color": ["get", "color"], "circle-stroke-width": 2 } });
      m.addLayer({ id: "trucks", type: "circle", source: "trucks",
        paint: { "circle-radius": 4.5, "circle-color": ["match", ["get", "state"], "loaded", "#ff65a3", "loading", "#ffd1e3", "returning", "#a1a1aa", "#52525b"],
          "circle-stroke-color": "#09090b", "circle-stroke-width": 1.2 } });
      m.addLayer({ id: "selected", type: "circle", source: "points", filter: ["==", ["get", "path"], ""],
        paint: { "circle-radius": 14, "circle-color": "transparent", "circle-stroke-color": "#fe0265", "circle-stroke-width": 2.5 } });
      const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 10 });
      for (const layer of ["wells", "facilities", "pads", "crews", "pipelines", "lanes", "trucks"]) {
        m.on("mouseenter", layer, (e) => {
          m.getCanvas().style.cursor = "pointer";
          const f = e.features?.[0];
          if (!f) return;
          const p = f.properties as any;
          popup.setLngLat(e.lngLat).setHTML(`<div style="font-weight:600">${p.label ?? p.truck}</div><div style="color:#a1a1aa">${p.info ?? ""}</div>`).addTo(m);
        });
        m.on("mouseleave", layer, () => { m.getCanvas().style.cursor = ""; popup.remove(); });
        m.on("click", layer, (e) => {
          const p = e.features?.[0]?.properties as any;
          if (p?.path) set({ selected: p.path });
        });
      }
      setReady(true);
    });
    return () => { markers.current.forEach((mk) => mk.remove()); markers.current = []; m.remove(); map.current = null; setReady(false); };
  }, [structure, basemap]);

  // ---------------------------------------------------------------- labels (HTML markers: no glyph server needed)
  useEffect(() => {
    const m = map.current;
    if (!m || !ready || !structure) return;
    markers.current.forEach((mk) => mk.remove());
    markers.current = [];
    if (!vis.labels) return;
    for (const a of structure.assets) {
      if (!a.at || a.is_link || ["esp_well", "rod_pump_well", "field_crew"].includes(a.type)) continue;
      const el = document.createElement("div");
      el.style.cssText = "font:600 10.5px Inter;color:#f4f4f5;text-shadow:0 0 4px #000,0 0 2px #000;pointer-events:none;white-space:nowrap;transform:translate(0,-15px)";
      el.textContent = a.label;
      markers.current.push(new maplibregl.Marker({ element: el, anchor: "bottom" }).setLngLat([a.at[1], a.at[0]]).addTo(m));
    }
  }, [ready, structure, vis.labels]);

  // ---------------------------------------------------------------- live data
  useEffect(() => {
    const m = map.current;
    if (!m || !ready || !structure) return;
    (m.getSource("points") as maplibregl.GeoJSONSource)?.setData(pointsGeo(structure, snap, colorBy));
    (m.getSource("trucks") as maplibregl.GeoJSONSource)?.setData({
      type: "FeatureCollection",
      features: (snap?.trucks ?? []).map((t) => ({ type: "Feature", geometry: { type: "Point", coordinates: [t.lon, t.lat] },
        properties: { truck: t.truck, label: `Truck ${t.truck}`, state: t.state, info: t.lane ? `${t.state} · ${t.lane} · ${t.cargo}` : "idle at yard" } })),
    });
    (m.getSource("jobs") as maplibregl.GeoJSONSource)?.setData({
      type: "FeatureCollection",
      features: (snap?.jobs ?? []).filter((j) => j.at && j.status !== "done").map((j) => ({ type: "Feature",
        geometry: { type: "Point", coordinates: [j.at![1], j.at![0]] },
        properties: { color: j.status === "active" ? "#8b5cf6" : "#71717a" } })),
    });
  }, [snap, ready, structure, colorBy]);

  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return;
    m.setFilter("selected", ["==", ["get", "path"], selected ?? ""]);
  }, [selected, ready]);

  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return;
    const lay: Record<string, string[]> = { wells: ["wells", "edges"], pads: ["pads"], facilities: ["facilities"],
      pipelines: ["pipelines"], lanes: ["lanes"], trucks: ["trucks"], crews: ["crews", "jobs"] };
    for (const [g, ids] of Object.entries(lay)) for (const id of ids) if (m.getLayer(id)) m.setLayoutProperty(id, "visibility", vis[g] ? "visible" : "none");
  }, [vis, ready]);

  const counts = useMemo(() => {
    const c = { running: 0, degraded: 0, down: 0 };
    for (const a of structure?.assets ?? []) {
      if (a.type !== "esp_well") continue;
      const s = wellStatus(a, snap).status as keyof typeof c;
      if (s in c) c[s]++;
    }
    return c;
  }, [structure, snap]);

  return (
    <div className="h-full relative">
      <div className="absolute inset-0"><div ref={ref} style={{ width: "100%", height: "100%" }} /></div>
      <div className="absolute top-3 left-3 panel w-[210px] z-10" style={{ background: "#12121aee" }}>
        <div className="panel-h"><span className="flex items-center gap-1.5"><Layers size={12} />Layers</span></div>
        <div className="p-2 space-y-1">
          {GROUPS.map((g) => (
            <label key={g.id} className="flex items-center gap-2 text-[12px] cursor-pointer">
              <input type="checkbox" checked={vis[g.id]} onChange={(e) => setVis({ ...vis, [g.id]: e.target.checked })} />{g.label}
            </label>
          ))}
          <div className="pt-1.5 border-t border-line mt-1.5 text-[11px] text-dim">Colour assets by</div>
          <div className="flex gap-1">
            {(["status", "subsystem"] as const).map((c) => (
              <button key={c} className={`btn !py-0.5 !text-[11px] flex-1 justify-center ${colorBy === c ? "btn-primary" : ""}`} onClick={() => setColorBy(c)}>{c}</button>
            ))}
          </div>
          <div className="pt-1.5 text-[11px] text-dim">Basemap</div>
          <select className="inp w-full" value={basemap} onChange={(e) => { setBasemap(e.target.value); localStorage.setItem("plx.basemap2", e.target.value); }}>
            {Object.entries(BASEMAPS).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
          </select>
        </div>
      </div>
      <div className="absolute bottom-3 right-14 panel z-10 px-3 py-2 text-[11px] space-y-1" style={{ background: "#12121aee" }}>
        <div className="flex gap-3">
          <span><span className="text-good">●</span> running <span className="num">{counts.running}</span></span>
          <span><span className="text-warn">●</span> degraded <span className="num">{counts.degraded}</span></span>
          <span><span className="text-crit">●</span> down <span className="num">{counts.down}</span></span>
        </div>
        <div className="flex gap-3 text-dim">
          <span><span style={{ color: "#c97a14" }}>■</span> multiphase / oil</span><span><span style={{ color: "#8b5cf6" }}>■</span> gas</span>
          <span><span style={{ color: "#3987e5" }}>■</span> water</span><span><span style={{ color: "#ff65a3" }}>┅</span> truck lane</span>
        </div>
      </div>
    </div>
  );
}

function linkColor(res: string[]): string {
  if (res.includes("gas") && !res.includes("oil")) return "#8b5cf6";
  if (res.length === 1 && res[0] === "water") return "#3987e5";
  return "#c97a14";
}

function linksGeo(st: Structure): GeoJSON.FeatureCollection {
  return {
    type: "FeatureCollection",
    features: st.assets.filter((a) => a.line).map((a) => ({
      type: "Feature", geometry: { type: "LineString", coordinates: a.line!.map((p) => [p[1], p[0]]) },
      properties: { path: a.path, label: a.label, kind: a.type.includes("lane") || a.type.includes("haul") ? "lane" : "pipe",
        color: linkColor((a as any).resources ?? []), info: `${a.type.replace(/_/g, " ")} · ${((a as any).resources ?? []).join(", ")}` },
    })),
  };
}

function edgesGeo(st: Structure): GeoJSON.FeatureCollection {
  return { type: "FeatureCollection", features: st.edges.map((e) => ({ type: "Feature",
    geometry: { type: "LineString", coordinates: e.line.map((p) => [p[1], p[0]]) }, properties: {} })) };
}

function pointsGeo(st: Structure, snap: Snapshot | null, colorBy: "status" | "subsystem"): GeoJSON.FeatureCollection {
  const features: GeoJSON.Feature[] = [];
  for (const a of st.assets) {
    if (!a.at || a.is_link) continue;
    let group = "facilities", color = TYPE_COLOR[a.type] ?? "#a1a1aa", rate = 0, info = a.type.replace(/_/g, " ");
    if (a.type.endsWith("_well") && a.type !== "swd_well") {
      group = "wells";
      const w = wellStatus(a, snap);
      rate = w.rate;
      color = w.status === "down" ? "#ef4444" : w.status === "degraded" ? "#f59e0b" : w.status === "unknown" ? "#52525b" : "#34d399";
      info = `${w.status} · ${fmt.n(w.rate)} bbl/d liquid · vib ${fmt.n(w.vib, 1)} mm/s`;
    } else if (a.type === "pad") group = "pads";
    else if (a.type === "workover_rig" || a.type === "field_crew") group = "crews";
    else {
      const v = snap?.assets[a.path] ?? {};
      const parts = Object.entries(v).slice(0, 4).map(([k, x]) => `${k.split(".").slice(-2).join(".")} ${fmt.n(x, x < 10 ? 2 : 0)}`);
      if (parts.length) info = parts.join(" · ");
    }
    if (colorBy === "subsystem" && a.subsystem !== null && group !== "crews") color = SUB_COLORS[a.subsystem % SUB_COLORS.length];
    features.push({ type: "Feature", geometry: { type: "Point", coordinates: [a.at[1], a.at[0]] },
      properties: { path: a.path, label: a.label, type: a.type, group, color, rate, info } });
  }
  return { type: "FeatureCollection", features };
}
