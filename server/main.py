'use client';

import { useState } from 'react';
import { askGroq } from './actions';

export default function GroqChatForm() {
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  // This function handles the form submission
  async function clientAction(formData: FormData) {
    setLoading(true);
    const result = await askGroq(formData);
    setResponse(result);
    setLoading(false);
  }

  return (
    <div className="max-w-md mx-auto p-6">
      {/* 1. Using the 'action' attribute is the modern Next.js way */}
      <form action={clientAction} className="flex flex-col gap-4">
        
        <label htmlFor="prompt" className="font-bold">Ask Groq:</label>
        
        <input 
          name="prompt" // Important: This matches formData.get('prompt')
          type="text"
          placeholder="Type something..."
          className="border rounded-lg p-3 text-black focus:ring-2 focus:ring-blue-500 outline-none"
          required
        />

        <button 
          type="submit" 
          disabled={loading}
          className="bg-black text-white p-3 rounded-lg font-medium hover:bg-gray-800 disabled:bg-gray-400 transition"
        >
          {loading ? "Thinking..." : "Send Message"}
        </button>
      </form>

      {/* Response Area */}
      {response && (
        <div className="mt-6 p-4 border-t-2 border-gray-100">
          <p className="text-gray-500 text-sm mb-1 uppercase tracking-wider">Response:</p>
          <div className="text-gray-800 leading-relaxed italic">
            "{response}"
          </div>
        </div>
      )}
    </div>
  );
}