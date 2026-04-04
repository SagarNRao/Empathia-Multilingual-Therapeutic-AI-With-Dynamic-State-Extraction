"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export type CognitiveShift = {
  message_num: number;
  affective: number;
  cognitive: number;
  agency: number;
  dominant: string;
  timestamp: string;
  content: string;
};

interface CognitiveShiftChartProps {
  shifts: CognitiveShift[];
}

export default function CognitiveShiftChart({ shifts }: CognitiveShiftChartProps) {
  // Transform data for the chart
  const chartData = shifts.map((shift) => ({
    name: `Msg ${shift.message_num}`,
    affective: Math.round(shift.affective * 100),
    cognitive: Math.round(shift.cognitive * 100),
    agency: Math.round(shift.agency * 100),
  }));

  return (
    <div className="w-full h-full min-h-[400px] flex flex-col">
      <ResponsiveContainer width="100%" height={400}>
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis
            dataKey="name"
            stroke="rgba(255,255,255,0.4)"
            style={{ fontSize: "12px" }}
          />
          <YAxis
            stroke="rgba(255,255,255,0.4)"
            style={{ fontSize: "12px" }}
            domain={[0, 100]}
            label={{ value: "Score (%)", angle: -90, position: "insideLeft" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1a1a1a",
              border: "1px solid rgba(200,169,110,0.3)",
              borderRadius: "8px",
              color: "#e8e3d9",
            }}
            formatter={(value: number | string) => `${value}%`}
          />
          <Legend
            wrapperStyle={{ color: "#c9c3b8" }}
            iconType="line"
          />
          <Line
            type="monotone"
            dataKey="affective"
            stroke="#ff6b6b"
            dot={{ fill: "#ff6b6b", r: 4 }}
            activeDot={{ r: 6 }}
            strokeWidth={2}
            name="Affective"
          />
          <Line
            type="monotone"
            dataKey="cognitive"
            stroke="#4ecdc4"
            dot={{ fill: "#4ecdc4", r: 4 }}
            activeDot={{ r: 6 }}
            strokeWidth={2}
            name="Cognitive"
          />
          <Line
            type="monotone"
            dataKey="agency"
            stroke="#ffd93d"
            dot={{ fill: "#ffd93d", r: 4 }}
            activeDot={{ r: 6 }}
            strokeWidth={2}
            name="Agency"
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Data table below chart */}
      <div className="mt-6 flex-1 overflow-y-auto">
        <div className="text-sm text-white/60 mb-3">
          <p className="font-semibold text-white/80">Detailed Breakdown ({shifts.length} messages)</p>
        </div>
        <div className="space-y-2">
          {shifts.map((shift, idx) => (
            <div
              key={idx}
              className="text-xs bg-white/[0.02] border border-white/[0.05] rounded p-2"
            >
              <div className="flex justify-between items-start mb-1">
                <span className="font-medium text-[#c8a96e]">Message {shift.message_num}</span>
                <span className="text-white/40">{shift.dominant}</span>
              </div>
              <div className="text-white/50 mb-1 truncate">{shift.content}</div>
              <div className="grid grid-cols-3 gap-2 text-white/60">
                <div>
                  <span className="text-[#ff6b6b]">Affective: </span>
                  {(shift.affective * 100).toFixed(1)}%
                </div>
                <div>
                  <span className="text-[#4ecdc4]">Cognitive: </span>
                  {(shift.cognitive * 100).toFixed(1)}%
                </div>
                <div>
                  <span className="text-[#ffd93d]">Agency: </span>
                  {(shift.agency * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
