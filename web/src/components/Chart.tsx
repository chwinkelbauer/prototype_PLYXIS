import { useEffect, useRef } from "react";
import * as echarts from "echarts";

export const C = {
  plx: "#34d399", leg: "#f59e0b", teal: "#2dd4bf", opt: "#a78bfa", info: "#38bdf8", crit: "#f43f5e",
  warn: "#fbbf24", grid: "#1b2632", axis: "#56667a", text: "#8293a6",
};

export function base(extra: echarts.EChartsOption = {}): echarts.EChartsOption {
  return {
    animation: false,
    backgroundColor: "transparent",
    textStyle: { fontFamily: "Inter", color: C.text, fontSize: 11 },
    grid: { left: 52, right: 14, top: 26, bottom: 26 },
    tooltip: {
      trigger: "axis", backgroundColor: "#0c131b", borderColor: "#263443", textStyle: { color: "#d7e1eb", fontSize: 11 },
      axisPointer: { lineStyle: { color: "#34475a" } },
    },
    legend: { top: 0, right: 8, textStyle: { color: C.text, fontSize: 11 }, itemWidth: 14, itemHeight: 3, icon: "rect" },
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
