"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

export type CognitiveShift = {
  message_num: number;
  affective: number;
  cognitive: number;
  agency: number;
  dominant: string;
  timestamp: string;
  content: string;
};

const COLORS = {
  affective: "#e07b7b",   // soft red — emotional warmth
  cognitive: "#7baee0",   // blue — analytical cool
  agency:    "#a0c878",   // green — growth / action
};

const DOMINANT_BG: Record<string, string> = {
  affective: "rgba(224,123,123,0.12)",
  cognitive: "rgba(123,174,224,0.12)",
  agency:    "rgba(160,200,120,0.12)",
};

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const item = payload[0]?.payload;
  return (
    <div
      className="rounded-xl border border-white/10 p-3 text-xs"
      style={{ background: "#1c1c1c", maxWidth: 240 }}
    >
      <p className="text-white/50 mb-1">Message #{label}</p>
      {payload.map((p: any) => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.name}: <strong>{(p.value * 100).toFixed(0)}%</strong>
        </p>
      ))}
      {item?.dominant && (
        <p className="mt-1 text-white/40 uppercase tracking-wider text-[10px]">
          dominant: {item.dominant}
        </p>
      )}
      {item?.content && (
        <p className="mt-1 text-white/60 italic border-t border-white/10 pt-1 line-clamp-2">
          "{item.content}"
        </p>
      )}
    </div>
  );
}

// Renders a small colored dot on the line to mark the dominant shift
function DominantDot(props: any) {
  const { cx, cy, payload } = props;
  if (!payload?.dominant) return null;
  const color = COLORS[payload.dominant as keyof typeof COLORS] ?? "#888";
  return (
    <circle
      cx={cx}
      cy={cy}
      r={5}
      fill={color}
      stroke="#0f0f0f"
      strokeWidth={2}
    />
  );
}

export default function CognitiveShiftChart({ shifts }: { shifts: CognitiveShift[] }) {
  if (!shifts.length) {
    return (
      <p className="text-center text-white/30 py-8">No data yet.</p>
    );
  }

  // Determine the progression narrative
  const dominantCounts: Record<string, number> = { affective: 0, cognitive: 0, agency: 0 };
  shifts.forEach((s) => {
    if (s.dominant in dominantCounts) dominantCounts[s.dominant]++;
  });

  const progression = shifts.map((s) => s.dominant);
  const lastThree = progression.slice(-3).join(" → ");

  return (
    <div className="space-y-4">
      {/* Legend / narrative */}
      <div className="flex flex-wrap gap-3 items-center text-xs">
        {(["affective", "cognitive", "agency"] as const).map((k) => (
          <span
            key={k}
            className="flex items-center gap-1.5 px-2 py-1 rounded-full"
            style={{ background: DOMINANT_BG[k], color: COLORS[k] }}
          >
            <span className="w-2 h-2 rounded-full" style={{ background: COLORS[k] }} />
            {k.charAt(0).toUpperCase() + k.slice(1)}
            <span className="text-white/30 ml-1">({dominantCounts[k]})</span>
          </span>
        ))}
        {shifts.length >= 3 && (
          <span className="ml-auto text-white/30 italic">
            Recent arc: {lastThree}
          </span>
        )}
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={320}>
        <LineChart
          data={shifts}
          margin={{ top: 8, right: 16, left: -10, bottom: 8 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis
            dataKey="message_num"
            label={{ value: "Message #", position: "insideBottomRight", offset: -4, fill: "#ffffff44", fontSize: 11 }}
            tick={{ fill: "#ffffff44", fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: "#ffffff44", fontSize: 11 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0.5} stroke="rgba(255,255,255,0.08)" strokeDasharray="4 2" />

          <Line
            type="monotone"
            dataKey="affective"
            name="Affective"
            stroke={COLORS.affective}
            strokeWidth={2}
            dot={<DominantDot />}
            activeDot={{ r: 6 }}
          />
          <Line
            type="monotone"
            dataKey="cognitive"
            name="Cognitive"
            stroke={COLORS.cognitive}
            strokeWidth={2}
            dot={<DominantDot />}
            activeDot={{ r: 6 }}
          />
          <Line
            type="monotone"
            dataKey="agency"
            name="Agency"
            stroke={COLORS.agency}
            strokeWidth={2}
            dot={<DominantDot />}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Turn-by-turn breakdown */}
      <div className="mt-2 space-y-1 max-h-40 overflow-y-auto pr-1">
        {shifts.map((s) => (
          <div
            key={s.message_num}
            className="flex items-start gap-2 text-xs py-1.5 border-b border-white/[0.04]"
          >
            <span className="text-white/30 w-5 shrink-0">#{s.message_num}</span>
            <span
              className="shrink-0 px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wider font-medium"
              style={{
                background: DOMINANT_BG[s.dominant] ?? "rgba(255,255,255,0.05)",
                color: COLORS[s.dominant as keyof typeof COLORS] ?? "#888",
              }}
            >
              {s.dominant}
            </span>
            <span className="text-white/50 line-clamp-1 flex-1 italic">"{s.content}"</span>
            <span className="text-white/20 shrink-0">
              A:{(s.affective * 100).toFixed(0)} C:{(s.cognitive * 100).toFixed(0)} Ag:{(s.agency * 100).toFixed(0)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
