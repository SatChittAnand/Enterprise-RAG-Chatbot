import { GoogleGenAI } from '@google/genai';
import { Chunk } from '../types';

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// Using the embedding model with TaskType
export const generateEmbeddings = async (
  texts: string[],
  taskType: 'RETRIEVAL_DOCUMENT' | 'RETRIEVAL_QUERY' = 'RETRIEVAL_DOCUMENT'
): Promise<number[][]> => {
  if (texts.length === 0) return [];
  
  const embeddings: number[][] = [];
  
  for (const text of texts) {
      if (!text || !text.trim()) {
        embeddings.push([]);
        continue;
      }
      try {
        // Correct usage for text-embedding-004 with task type
        const response = await ai.models.embedContent({
            model: 'text-embedding-004',
            contents: [{ parts: [{ text: text }] }],
            config: {
                taskType: taskType,
            }
        });
        if (response.embeddings?.[0]?.values) {
            embeddings.push(response.embeddings[0].values);
        } else {
            console.warn("No embedding returned");
            embeddings.push([]); 
        }
      } catch (e) {
          console.error("Embedding failed", e);
          embeddings.push([]);
      }
  }

  return embeddings;
};

export const generateRAGResponse = async (
  query: string,
  contextChunks: Chunk[],
  chatHistory: { role: 'user' | 'model'; content: string }[]
): Promise<{ text: string; auditTrail: string }> => {
  
  const contextText = contextChunks
    .map((c) => `[Source: ${c.docName}, Page ${c.pageNumber || 1}]: ${c.text}`)
    .join('\n\n');

  // If no context is found, return early to save API calls and be explicit
  if (contextChunks.length === 0) {
      return {
          text: "Information not found in the documents provided.",
          auditTrail: `**Query Processing:**\n- Input: "${query}"\n- Context Retrieval: Found 0 relevant chunks (Similarity below threshold).\n- Action: Stopped generation.`
      };
  }

  const systemInstruction = `You are a strict Enterprise RAG Assistant. 
  Your goal is to answer user questions strictly based on the provided Context.
  
  Context:
  ${contextText}
  
  Rules:
  1. Answer ONLY using the information from the Context above.
  2. If the answer is not present in the Context, you MUST say "Information not found in the documents provided."
  3. Do not use outside knowledge or hallucinate facts.
  4. At the end of your answer, list citations in the format (Document Name, Page X).
  5. Be professional and concise.`;

  // Filter history to last 5 turns to keep context window manageable
  const recentHistory = chatHistory.slice(-10);

  try {
    const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: [
            ...recentHistory.map(h => ({ role: h.role, parts: [{ text: h.content }] })),
            { role: 'user', parts: [{ text: query }] }
        ],
        config: {
            systemInstruction: systemInstruction,
            temperature: 0.1, // Low temperature for factual consistency
        }
    });

    const text = response.text || "No response generated.";
    
    // Create a mini audit log of what happened
    const auditTrail = `
**Query Processing:**
- Input: "${query}"
- Context Retrieval: Found ${contextChunks.length} relevant chunks.
- Model Used: gemini-2.5-flash
- Temperature: 0.1 (Strict Mode)

**Retrieved Context Snippets:**
${contextChunks.map(c => `- ${c.docName} (Pg ${c.pageNumber}): "${c.text.substring(0, 50)}..."`).join('\n')}
    `.trim();

    return { text, auditTrail };

  } catch (error) {
    console.error("Gemini Gen error:", error);
    return { 
        text: "I encountered an error while processing your request with the AI model.", 
        auditTrail: "Error calling Gemini API." 
    };
  }
};
