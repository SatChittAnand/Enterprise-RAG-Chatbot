export interface Document {
  id: string;
  name: string;
  type: 'pdf' | 'docx' | 'txt';
  uploadDate: Date;
  pageCount: number;
  status: 'processing' | 'ready' | 'error';
}

export interface Chunk {
  id: string;
  docId: string;
  docName: string;
  text: string;
  pageNumber?: number;
  embedding?: number[];
}

export interface RetrievalResult {
  chunk: Chunk;
  similarity: number;
}

export interface Message {
  id: string;
  role: 'user' | 'model';
  content: string;
  timestamp: Date;
  citations?: Chunk[]; // The chunks used to generate this answer
  auditLog?: string; // Step-by-step explanation
  retrievalResults?: RetrievalResult[]; // Data for RAG Graphs
}
