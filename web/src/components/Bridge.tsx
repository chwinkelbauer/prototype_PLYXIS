import { useMemo } from "react";
import { fmt } from "../api";
import { base, C, EChart } from "./Chart";

export type BridgeRow = { key: string; label: string; delta: number };

/** Operating-profit difference (PLYXIS − manual) by realised term, largest first. Bars in the colour of
 *  the world the term favours; the rest beyond `top` folds into "Other terms". */
export function BridgeChart({ rows, height = 220, top = 7 }: { rows: BridgeRow[]; height?: number; top?: number }) {
  const option = useMemo(() => {
    const head = rows.slice(0, top);
    const rest = rows.slice(top).reduce((s, r) => s + r.delta, 0);
    const all = Math.abs(rest) >= 1 ? [...head, { key: "other", label: "Other terms", delta: rest }] : head;
    const ordered = all.slice().reverse();                 // largest contribution on top
    // labels left, values in their own column on the right: nothing can collide with the bars
    return base({
      grid: { left: 150, right: 70, top: 6, bottom: 22 },
      legend: { show: false },
      tooltip: { trigger: "item", formatter: (p: any) => `${p.name}<br/><b>${fmt.usd(p.value)}</b> ${p.value >= 0 ? "in favour of PLYXIS" : "in favour of manual ops"}` },
      xAxis: { type: "value", axisLine: { show: false }, axisLabel: { color: C.axis, fontSize: 10, formatter: (v: number) => fmt.usd(v) }, splitLine: { lineStyle: { color: C.grid } } },
      yAxis: [
        { type: "category", data: ordered.map((r) => r.label), axisTick: { show: false }, axisLine: { lineStyle: { color: "#3a3a56" } },
          axisLabel: { color: C.text, fontSize: 11, width: 140, overflow: "truncate" }, splitLine: { show: false } },
        { type: "category", position: "right", data: ordered.map((r) => fmt.usd(r.delta)), axisTick: { show: false }, axisLine: { show: false },
          axisLabel: { color: "#f4f4f5", fontSize: 10.5, fontFamily: "JetBrains Mono" }, splitLine: { show: false } },
      ],
      series: [{
        type: "bar", name: "PLYXIS − manual", barWidth: 10, yAxisIndex: 0,
        data: ordered.map((r) => ({
          value: r.delta,
          itemStyle: { color: r.delta >= 0 ? C.plx : C.leg, borderRadius: r.delta >= 0 ? [0, 4, 4, 0] : [4, 0, 0, 4] },
        })),
      }],
    });
  }, [rows, top]);
  if (!rows.length) {
    return <div className="flex items-center justify-center text-dim text-[12px]" style={{ height }}>builds up as the two worlds diverge</div>;
  }
  return <EChart height={height} option={option} />;
}
