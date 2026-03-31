'use server';

import { createGroq } from '@ai-sdk/groq';
import { generateText } from 'ai';

const groq = createGroq({
  apiKey: process.env.GROQ_API_KEY,
});

export async function askGroq(userPrompt: string) {
  if (!userPrompt) return "Prompt is empty, buddy.";

  try {
    const { text } = await generateText({
      model: groq('llama-3.3-70b-versatile'), // Llama 3.3 is the 2026 sweet spot for Groq
      system: 'You are a concise, witty AI assistant.',
      prompt: userPrompt,
    });

    return text;
  } catch (error) {
    console.error("Groq Error:", error);
    return "Dammit, something went wrong with the API.";
  }
}