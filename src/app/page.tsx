"use client";

import { useState, useRef, useEffect } from "react";
import { Send, BarChart3, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { NotesSidebar } from "@/components/notes-sidebar";
import dynamic from "next/dynamic";

const CognitiveShiftChart = dynamic(() => import("@/components/cognitive-shift-chart"), {
  ssr: false,
});

export type Note = {
  title: string;
  content: string;
  category: string;
};

export type NotesMap = Record<string, Note>;

export type Message = {
  role: "user" | "assistant";
  content: string;
};

export type CognitiveShift = {
  message_num: number;
  affective: number;
  cognitive: number;
  agency: number;
  dominant: string;
  timestamp: string;
  content: string;
};

const SESSION_ID = 1;

// Server Configuration
const CHAT_SERVER = "http://localhost:8000"; // Main server (LLM + session management)
// Note: The main server internally calls the NLP server at http://localhost:8001

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "hi bro what scene today",
    },
  ]);
  const [notes, setNotes] = useState<NotesMap>({});
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showGraph, setShowGraph] = useState(false);
  const [cognitiveShifts, setCognitiveShifts] = useState<CognitiveShift[]>([]);
  const [serverError, setServerError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function fetchCognitiveShifts() {
    try {
      setServerError(null);
      const res = await fetch(`${CHAT_SERVER}/sessions/${SESSION_ID}/cognitive-shifts`);
      if (!res.ok) throw new Error("Failed to fetch cognitive shifts");
      const data = await res.json();
      setCognitiveShifts(data.shifts);
      setShowGraph(true);
    } catch (error) {
      const errorMsg = `Error fetching cognitive shifts: ${error instanceof Error ? error.message : String(error)}`;
      console.error(errorMsg);
      setServerError(errorMsg);
    }
  }

  async function sendMessage() {
    const text = input.trim();
    if (!text || loading) return;

    const newMessages: Message[] = [
      ...messages,
      { role: "user", content: text },
    ];
    setMessages(newMessages);
    setInput("");
    setLoading(true);
    setServerError(null);

    try {
      const res = await fetch(`${CHAT_SERVER}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: SESSION_ID,
          message: text,
          history: newMessages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
        }),
      });

      if (!res.ok) throw new Error("API error");
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.reply },
      ]);

      // Merge new notes into existing ones
      if (data.extracted_notes?.notes) {
        const incoming: NotesMap = {};
        const raw = data.extracted_notes.notes;

        // Handle both array and object shapes
        if (Array.isArray(raw)) {
          raw.forEach((n: Note, i: number) => {
            incoming[n.title?.replace(/\s+/g, "_").toLowerCase() ?? `note_${i}`] = n;
          });
        } else {
          Object.assign(incoming, raw);
        }

        setNotes((prev) => ({ ...prev, ...incoming }));
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      setServerError(errorMsg);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Sorry, I couldn't connect to the server. Is the backend running?\n\nError: ${errorMsg}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="flex h-screen bg-[#0f0f0f] text-[#e8e3d9] font-sans overflow-hidden">
      {/* Chat area */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/[0.06] flex items-center gap-3">
          <div className={`w-2 h-2 rounded-full ${serverError ? 'bg-red-500' : 'bg-[#c8a96e]'} ${!serverError && 'animate-pulse'}`} />
          <span className="text-sm font-medium tracking-widest uppercase text-[#c8a96e]">
            Indy
          </span>
          {serverError && (
            <span className="text-xs text-red-400 ml-auto mr-2">
              ⚠️ Server: {serverError}
            </span>
          )}
          <span className="text-xs text-white/30 ml-auto">Session {SESSION_ID}</span>
          <Button
            onClick={fetchCognitiveShifts}
            variant="ghost"
            size="sm"
            className="text-[#c8a96e] hover:text-[#e8e3d9] hover:bg-white/[0.04]"
            title="View cognitive shift graph"
          >
            <BarChart3 className="w-4 h-4" />
          </Button>
        </div>

        {/* Messages */}
        <ScrollArea className="flex-1 px-6 py-6">
          <div className="max-w-2xl mx-auto space-y-6">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[78%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "bg-[#c8a96e]/10 text-[#e8e3d9] border border-[#c8a96e]/20 rounded-br-sm"
                      : "bg-white/[0.04] text-[#c9c3b8] border border-white/[0.06] rounded-bl-sm"
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="bg-white/[0.04] border border-white/[0.06] rounded-2xl rounded-bl-sm px-4 py-3">
                  <span className="flex gap-1">
                    {[0, 1, 2].map((i) => (
                      <span
                        key={i}
                        className="w-1.5 h-1.5 rounded-full bg-[#c8a96e]/60 animate-bounce"
                        style={{ animationDelay: `${i * 0.15}s` }}
                      />
                    ))}
                  </span>
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        </ScrollArea>

        {/* Input */}
        <div className="px-6 py-4 border-t border-white/[0.06]">
          <div className="max-w-2xl mx-auto flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="What's going on…"
              disabled={loading}
              className="flex-1 bg-white/[0.04] border-white/[0.08] text-[#e8e3d9] placeholder:text-white/20 focus-visible:ring-[#c8a96e]/40 focus-visible:border-[#c8a96e]/40 rounded-xl h-11"
            />
            <Button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              size="icon"
              className="h-11 w-11 rounded-xl bg-[#c8a96e] hover:bg-[#b8996e] text-[#0f0f0f] disabled:opacity-30 shrink-0"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <NotesSidebar notes={notes} />

      {/* Cognitive Shift Graph Modal */}
      {showGraph && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-[#1a1a1a] border border-white/[0.1] rounded-2xl p-6 max-w-4xl w-full max-h-[90vh] overflow-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-[#e8e3d9]">Cognitive Shift Over Time</h2>
              <button
                onClick={() => setShowGraph(false)}
                className="text-white/40 hover:text-white/60 transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            {cognitiveShifts.length > 0 ? (
              <CognitiveShiftChart shifts={cognitiveShifts} />
            ) : (
              <div className="text-center py-8 text-white/40">
                No cognitive shift data available yet
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
