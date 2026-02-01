import { Chunk, Document } from '../types';

// Declare globals loaded via CDN
declare const pdfjsLib: any;
declare const mammoth: any;

export const processFile = async (file: File): Promise<{ document: Document; chunks: Chunk[] }> => {
  const docId = crypto.randomUUID();
  let text = '';
  let pageMap: { text: string; page: number }[] = [];
  let pageCount = 1;

  try {
    if (file.type === 'application/pdf') {
      const arrayBuffer = await file.arrayBuffer();
      const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
      pageCount = pdf.numPages;

      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const tokenizedText = await page.getTextContent();
        const pageText = tokenizedText.items.map((item: any) => item.str).join(' ');
        text += pageText + '\n';
        pageMap.push({ text: pageText, page: i });
      }
    } else if (
      file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ) {
      const arrayBuffer = await file.arrayBuffer();
      const result = await mammoth.extractRawText({ arrayBuffer: arrayBuffer });
      text = result.value;
      // Mammoth doesn't give page numbers easily, assume page 1
      pageMap.push({ text: text, page: 1 });
    } else {
      // Text file
      text = await file.text();
      pageMap.push({ text: text, page: 1 });
    }
  } catch (error) {
    console.error('Error processing file:', error);
    throw new Error('Failed to parse file content.');
  }

  const chunks = chunkText(docId, file.name, pageMap);

  const document: Document = {
    id: docId,
    name: file.name,
    type: file.type.includes('pdf') ? 'pdf' : file.type.includes('word') ? 'docx' : 'txt',
    uploadDate: new Date(),
    pageCount: pageCount,
    status: 'ready',
  };

  return { document, chunks };
};

// Simple sliding window chunking with overlap
const CHUNK_SIZE = 500; // characters ~ 100-150 tokens
const OVERLAP = 100;

const chunkText = (docId: string, docName: string, pageMap: { text: string; page: number }[]): Chunk[] => {
  const chunks: Chunk[] = [];

  pageMap.forEach((pageData) => {
    let startIndex = 0;
    const text = pageData.text.replace(/\s+/g, ' ').trim(); // Clean whitespace

    if (text.length === 0) return;

    while (startIndex < text.length) {
      const end = Math.min(startIndex + CHUNK_SIZE, text.length);
      const chunkText = text.slice(startIndex, end);

      chunks.push({
        id: crypto.randomUUID(),
        docId,
        docName,
        text: chunkText,
        pageNumber: pageData.page,
      });

      startIndex += CHUNK_SIZE - OVERLAP;
      if (startIndex >= text.length - OVERLAP && chunks.length > 0) break; 
    }
  });

  return chunks;
};
