"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { NotesMap } from "@/app/page";

const CATEGORY_COLORS: Record<string, string> = {
  goal:          "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  dream:         "bg-violet-500/10 text-violet-400 border-violet-500/20",
  trauma:        "bg-rose-500/10 text-rose-400 border-rose-500/20",
  trigger:       "bg-orange-500/10 text-orange-400 border-orange-500/20",
  belief:        "bg-sky-500/10 text-sky-400 border-sky-500/20",
  relationship:  "bg-pink-500/10 text-pink-400 border-pink-500/20",
  win:           "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  blocker:       "bg-red-500/10 text-red-400 border-red-500/20",
  insight:       "bg-teal-500/10 text-teal-400 border-teal-500/20",
  personal_info: "bg-slate-500/10 text-slate-400 border-slate-500/20",
};

function categoryStyle(category: string) {
  const key = category?.toLowerCase() ?? "";
  return (
    CATEGORY_COLORS[key] ??
    "bg-white/[0.05] text-white/40 border-white/10"
  );
}

type Props = { notes: NotesMap };

export function NotesSidebar({ notes }: Props) {
  const entries = Object.values(notes);

  return (
    <aside className="w-72 shrink-0 border-l border-white/[0.06] bg-[#0c0c0c] flex flex-col">
      <div className="px-4 py-4 border-b border-white/[0.06] flex items-center justify-between">
        <span className="text-xs font-medium tracking-widest uppercase text-white/30">
          Profile
        </span>
        {entries.length > 0 && (
          <span className="text-xs text-[#c8a96e]/60 tabular-nums">
            {entries.length} {entries.length === 1 ? "note" : "notes"}
          </span>
        )}
      </div>

      <ScrollArea className="flex-1 p-3">
        {entries.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-40 text-center px-4 gap-2">
            <div className="w-8 h-8 rounded-full border border-white/10 flex items-center justify-center text-white/20 text-lg">
              ✦
            </div>
            <p className="text-xs text-white/20 leading-relaxed">
              Share something meaningful and Indy will remember it here.
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {entries.map((note, i) => (
              <Card
                key={i}
                className="bg-white/[0.03] border-white/[0.06] rounded-xl overflow-hidden"
              >
                <CardHeader className="px-3 pt-3 pb-1">
                  <div className="flex items-start justify-between gap-2">
                    <CardTitle className="text-xs font-medium text-[#e8e3d9] leading-snug">
                      {note.title}
                    </CardTitle>
                    <Badge
                      variant="outline"
                      className={`text-[10px] px-1.5 py-0 shrink-0 border ${categoryStyle(note.category)}`}
                    >
                      {note.category}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="px-3 pb-3">
                  <p className="text-[11px] text-white/40 leading-relaxed">
                    {note.content}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </ScrollArea>
    </aside>
  );
}
