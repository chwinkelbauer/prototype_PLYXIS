import { useEffect, useRef } from "react";
import * as echarts from "echarts";

/* Chart roles. Categorical slots are validated for the dark surface (dataviz validator: lightness band,
   chroma, adjacent CVD separation >= 20 dE, contrast): pink, blue, orange, violet, aqua - in this order. */
export const SERIES = ["#fe0265", "#3987e5", "#c97a14", "#8b5cf6", "#199e70"];
export const C = {
  plx: "#fe0265",            // the PLYXIS world (brand)
  leg: "#8b93a7",            // manual operations (neutral context)
  teal: "#fe0265", opt: "#8b5cf6", info: "#3987e5",
  oil: "#c97a14", gas: "#8b5cf6", water: "#3987e5",
  good: "#34d399", crit: "#ef4444", warn: "#f59e0b",
  grid: "#1f1f2e", axis: "#71717a", text: "#a1a1aa", now: "#f59e0b",
};

export function base(extra: echarts.EChartsOption = {}): echarts.EChartsOption {
  return {
    animation: false,
    backgroundColor: "transparent",
    textStyle: { fontFamily: "Inter", color: C.text, fontSize: 11 },
    grid: { left: 52, right: 14, top: 26, bottom: 26 },
    color: SERIES,
    tooltip: {
      trigger: "axis", backgroundColor: "#12121af2", borderColor: "#2e2e44", textStyle: { color: "#f4f4f5", fontSize: 11 },
      axisPointer: { lineStyle: { color: "#3a3a56" } }, extraCssText: "border-radius:10px;box-shadow:0 8px 28px #000a;",
    },
    // legend top-left, one row, scrolls instead of colliding with axis names or the panel edge
    legend: { type: "scroll", top: 0, left: 8, right: 8, itemGap: 14, textStyle: { color: C.text, fontSize: 11 }, itemWidth: 14,
      itemHeight: 3, icon: "rect", pageIconColor: C.text, pageIconInactiveColor: "#3a3a56", pageTextStyle: { color: C.axis } },
    xAxis: { type: "value", axisLine: { lineStyle: { color: C.grid } }, axisLabel: { color: C.axis, fontSize: 10 },
      splitLine: { show: false } },
    yAxis: { type: "value", axisLine: { show: false }, axisLabel: { color: C.axis, fontSize: 10 },
      splitLine: { lineStyle: { color: C.grid } }, scale: true },
    ...extra,
  };
}

export function EChart({ option, height = 220, className = "", onClick }: {
  option: echarts.EChartsOption; height?: number | string; className?: string;
  onClick?: (p: any) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inst = useRef<echarts.ECharts | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    const ch = echarts.init(ref.current, undefined, { renderer: "canvas" });
    inst.current = ch;
    const ro = new ResizeObserver(() => ch.resize());
    ro.observe(ref.current);
    return () => { ro.disconnect(); ch.dispose(); inst.current = null; };
  }, []);
  useEffect(() => {
    inst.current?.setOption(option, { notMerge: true, lazyUpdate: true });
  }, [option]);
  useEffect(() => {
    const ch = inst.current;
    if (!ch || !onClick) return;
    ch.on("click", onClick);
    return () => { ch.off("click", onClick); };
  }, [onClick]);
  return <div ref={ref} className={className} style={{ height, width: "100%" }} />;
}

export function Spark({ data, color = C.teal, height = 28, width = 110 }: { data: number[]; color?: string; height?: number; width?: number }) {
  if (!data.length) return <svg width={width} height={height} />;
  const min = Math.min(...data), max = Math.max(...data);
  const span = max - min || 1;
  const pts = data.map((v, i) => `${(i / Math.max(1, data.length - 1)) * (width - 2) + 1},${height - 2 - ((v - min) / span) * (height - 4)}`);
  return (
    <svg width={width} height={height} className="block">
      <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth={1.3} />
    </svg>
  );
}
