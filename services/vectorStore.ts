import { Chunk, RetrievalResult } from '../types';

class VectorStore {
  private chunks: Chunk[] = [];

  public addChunks(newChunks: Chunk[]) {
    this.chunks = [...this.chunks, ...newChunks];
  }

  public getDocumentChunks(docId: string): Chunk[] {
    return this.chunks.filter(c => c.docId === docId);
  }

  public clear() {
    this.chunks = [];
  }

  public search(queryEmbedding: number[], topK: number = 4): RetrievalResult[] {
    if (this.chunks.length === 0) return [];

    const results = this.chunks.map((chunk) => {
      if (!chunk.embedding) return { chunk, similarity: -1 };
      const similarity = this.cosineSimilarity(queryEmbedding, chunk.embedding);
      return { chunk, similarity };
    });

    // Sort by similarity descending
    results.sort((a, b) => b.similarity - a.similarity);

    return results.slice(0, topK);
  }

  private cosineSimilarity(vecA: number[], vecB: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}

export const vectorStore = new VectorStore();