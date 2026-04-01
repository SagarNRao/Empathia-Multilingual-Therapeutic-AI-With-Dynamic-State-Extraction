const API_URL = "http://localhost:8000/chat";

// Maintain conversation history across calls
let conversationHistory = [];
let sessionId = 1;

async function testIndyCopilot(userMessage) {
  console.log(`\n--- User: "${userMessage}" ---`);

  const payload = {
    session_id: sessionId,
    message: userMessage,
    history: conversationHistory  // Send full conversation history
  };

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Server Error: ${response.statusText}`);
    }

    const data = await response.json();

    // Add user message to history
    conversationHistory.push({
      role: "user",
      content: userMessage
    });

    // Add assistant response to history
    conversationHistory.push({
      role: "assistant",
      content: data.reply
    });

    // 1. Show Indy's Reply
    console.log("%cIndy says:", "color: #3b82f6; font-weight: bold;");
    console.log(data.reply);

    // 2. Show Cognitive Shift Analysis
    console.log("%cCognitive Shift Tracker:", "color: #10b981; font-weight: bold;");
    console.table(data.cognitive_shift.scores);
    console.log(`Dominant Mode: ${data.cognitive_shift.dominant}`);

    // 3. Show Extracted Notes (The AI's JSON memory)
    console.log("%cExtracted Notes:", "color: #f59e0b; font-weight: bold;");
    console.log(data.extracted_notes);

    // 4. Show current conversation history length
    console.log(`%cConversation history length: ${conversationHistory.length / 2} turns`, "color: #8b5cf6;");

    return data;

  } catch (error) {
    console.error("Fetch Error:", error);
  }
}

// Run the test with multiple messages in sequence
async function runFullTest() {
  await testIndyCopilot("dude i am so sad");
  await new Promise(r => setTimeout(r, 1000)); // Wait 1 sec between messages
  
  await testIndyCopilot("My name is bottleman");
  await new Promise(r => setTimeout(r, 1000));
  
  await testIndyCopilot("i need to learn how to backflip");
  await new Promise(r => setTimeout(r, 1000));
  
  await testIndyCopilot("quit");
  await new Promise(r => setTimeout(r, 1000));
  
  console.log("%c=== FULL CONVERSATION HISTORY ===", "color: #ec4899; font-weight: bold;");
  console.table(conversationHistory);
}

// Run the full test
runFullTest().catch(console.error);