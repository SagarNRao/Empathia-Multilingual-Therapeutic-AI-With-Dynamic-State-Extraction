"use client";
import {
  useState,
  useRef,
  useEffect,
  KeyboardEvent,
  ChangeEvent,
  FocusEvent,
} from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { askGroq } from "@/lib/actions";

type DominantMode = "affective" | "cognitive" | "agency";

interface NLPScores {
  affective: number;
  cognitive: number;
  agency: number;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  scores: NLPScores | null;
  dominant?: DominantMode | null;
}

interface ApiResponse {
  reply: string;
  scores?: NLPScores;
  dominant?: DominantMode;
}

export default function Chats() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false)

  const handleAction = async () => {
    setLoading(true)
    const result = await askGroq(prompt)
    setResponse(result)
    setLoading(false)

    console.log(result)
  }

  return (
    <>
      <Input onChange={(e) => setPrompt(e.target.value)}></Input>
      <Button onClick={handleAction}>Go</Button>
    </>
  );
}
