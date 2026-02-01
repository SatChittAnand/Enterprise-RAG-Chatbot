import React, { useState, useRef, useEffect } from 'react';
import { processFile } from './services/documentProcessor';
import { vectorStore } from './services/vectorStore';
import { generateEmbeddings, generateRAGResponse } from './services/geminiService';
import { Document, Message, Chunk } from './types';
import { AttachmentIcon, SendIcon, FileIcon, TrashIcon, InfoIcon, RobotIcon, PlusIcon, MenuIcon, PanelLeftCloseIcon, PanelLeftOpenIcon } from './components/Icons';

export default function App() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'model',
      content: 'Hello! Please upload your PDF, DOCX, or Text files to the left. I will analyze them and answer your questions strictly based on their content.',
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [showAuditFor, setShowAuditFor] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    const newDocs: Document[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      try {
        // 1. Parse File
        const { document, chunks } = await processFile(file);
        
        // 2. Generate Embeddings (Task Type: DOCUMENT)
        const texts = chunks.map(c => c.text);
        const embeddings = await generateEmbeddings(texts, 'RETRIEVAL_DOCUMENT');
        
        // 3. Attach embeddings to chunks
        chunks.forEach((chunk, index) => {
          chunk.embedding = embeddings[index];
        });

        // 4. Store in Vector DB
        vectorStore.addChunks(chunks);

        newDocs.push(document);
      } catch (error) {
        console.error(`Failed to process ${file.name}`, error);
        alert(`Error processing ${file.name}. See console.`);
      }
    }

    setDocuments(prev => [...prev, ...newDocs]);
    setIsUploading(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isProcessing) return;

    const userQuery = inputValue.trim();
    setInputValue('');
    
    // Add User Message
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: userQuery,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMsg]);
    setIsProcessing(true);

    try {
      // 1. Embed Query (Task Type: QUERY)
      const [queryEmbedding] = await generateEmbeddings([userQuery], 'RETRIEVAL_QUERY');
      
      // 2. Retrieve Relevant Chunks
      const retrievalResults = vectorStore.search(queryEmbedding, 5); // Top 5
      
      // Lowered threshold to 0.35 to catch more potentially relevant chunks
      const relevantChunks = retrievalResults
          .filter(r => r.similarity > 0.35) 
          .map(r => r.chunk);

      // 3. Generate Answer
      const history = messages.map(m => ({ role: m.role, content: m.content }));
      const { text, auditTrail } = await generateRAGResponse(userQuery, relevantChunks, history);

      const aiMsg: Message = {
        id: crypto.randomUUID(),
        role: 'model',
        content: text,
        timestamp: new Date(),
        citations: relevantChunks,
        auditLog: auditTrail,
        retrievalResults: retrievalResults.filter(r => r.similarity > 0.35) // Store stats for graph
      };

      setMessages(prev => [...prev, aiMsg]);

    } catch (error) {
      console.error(error);
      const errorMsg: Message = {
        id: crypto.randomUUID(),
        role: 'model',
        content: "I'm sorry, something went wrong while processing your request.",
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsProcessing(false);
    }
  };

  const clearDocuments = () => {
    vectorStore.clear();
    setDocuments([]);
    setMessages([{
      id: crypto.randomUUID(),
      role: 'model',
      content: 'Knowledge base cleared. Please upload new documents.',
      timestamp: new Date()
    }]);
  };

  return (
    <div className="flex h-screen bg-slate-900 overflow-hidden">
      {/* Sidebar - Documents */}
      {/* Added overflow-hidden to fix the toggle issue */}
      <div 
        className={`bg-slate-950 text-slate-100 flex flex-col shadow-2xl z-20 transition-all duration-300 ease-in-out border-r border-slate-800 overflow-hidden ${isSidebarOpen ? 'w-80 opacity-100' : 'w-0 opacity-0'}`}
      >
        <div className="p-4 border-b border-slate-800 bg-slate-950 flex items-center justify-between min-w-[20rem]">
          <div>
            <h1 className="text-lg font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
              Enterprise RAG
            </h1>
            <p className="text-[10px] text-slate-500 font-medium tracking-wide">SECURE DOCUMENT CHAT</p>
          </div>
          
          {/* Upload Button */}
          <div className="flex gap-2">
             <input
                type="file"
                multiple
                accept=".pdf,.docx,.txt"
                ref={fileInputRef}
                onChange={handleFileUpload}
                className="hidden"
              />
             <button 
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                title="Upload Documents"
                className="p-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-white transition-all shadow-lg hover:shadow-blue-500/30 disabled:opacity-50"
             >
                {isUploading ? <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"/> : <PlusIcon />}
             </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 custom-scrollbar min-w-[20rem]">
          <div className="flex justify-between items-center mb-3 px-1">
            <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Documents ({documents.length})</h2>
            {documents.length > 0 && (
                <button onClick={clearDocuments} className="text-[10px] text-red-400 hover:text-red-300 flex items-center gap-1 bg-red-900/20 px-2 py-1 rounded transition-colors">
                    <TrashIcon /> Clear All
                </button>
            )}
          </div>

          <div className="space-y-2">
            {documents.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-40 border-2 border-dashed border-slate-800 rounded-xl text-slate-600 gap-2">
                <FileIcon />
                <span className="text-xs">No files added</span>
              </div>
            ) : (
              documents.map(doc => (
                <div key={doc.id} className="group flex items-start gap-3 p-3 bg-slate-800/50 hover:bg-slate-800 rounded-xl border border-slate-700/50 transition-all hover:border-blue-500/30 hover:shadow-lg">
                  <div className="mt-1 p-2 rounded-lg bg-gradient-to-br from-blue-900 to-slate-800 text-blue-400 shadow-inner">
                    <FileIcon />
                  </div>
                  <div className="overflow-hidden">
                    <p className="text-sm font-medium truncate text-slate-200 group-hover:text-blue-200 transition-colors">{doc.name}</p>
                    <div className="flex items-center gap-2 mt-1">
                        <span className="text-[10px] bg-slate-900 text-slate-400 px-1.5 py-0.5 rounded border border-slate-700">{doc.type.toUpperCase()}</span>
                        <span className="text-[10px] text-slate-500">{doc.pageCount} pages</span>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col bg-gradient-to-br from-indigo-50 via-purple-50 to-white relative min-w-0">
        
        {/* Header - Made colorful */}
        <header className="h-16 bg-gradient-to-r from-white/80 to-indigo-50/80 backdrop-blur-md border-b border-indigo-100 flex items-center justify-between px-4 sticky top-0 z-10 transition-all shadow-sm">
          <div className="flex items-center gap-3">
            <button 
                onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                className="p-2 text-indigo-600 hover:text-indigo-700 hover:bg-indigo-100/50 rounded-lg transition-colors"
                title={isSidebarOpen ? "Close Sidebar" : "Show Documents"}
            >
                {isSidebarOpen ? <PanelLeftCloseIcon /> : <PanelLeftOpenIcon />}
            </button>
            <h2 className="font-bold text-lg bg-gradient-to-r from-indigo-600 via-purple-600 to-fuchsia-600 bg-clip-text text-transparent flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]"></span>
              Chat Session
            </h2>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-6 space-y-8 scroll-smooth">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`}>
              <div className={`max-w-[85%] lg:max-w-[75%] flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                
                {/* Avatar */}
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 shadow-lg border-2 ${
                    msg.role === 'user' 
                    ? 'bg-gradient-to-br from-fuchsia-600 to-purple-600 text-white border-white/20' 
                    : 'bg-white text-indigo-600 border-indigo-100'
                }`}>
                    {msg.role === 'user' ? <span className="text-[10px] font-bold">YOU</span> : <RobotIcon />}
                </div>

                {/* Message Content */}
                <div className="flex flex-col gap-2 min-w-0 flex-1">
                    <div className={`p-5 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap shadow-lg ${
                        msg.role === 'user' 
                        ? 'bg-gradient-to-br from-fuchsia-600 to-purple-600 text-white rounded-tr-sm' 
                        : 'bg-white text-slate-700 border border-indigo-100 rounded-tl-sm ring-1 ring-indigo-50'
                    }`}>
                        {msg.content}
                    </div>

                    {/* Citations & Audit Log (Model Only) */}
                    {msg.role === 'model' && msg.citations && msg.citations.length > 0 && (
                        <div className="bg-white/60 p-3 rounded-xl border border-indigo-100 mt-1 shadow-sm backdrop-blur-sm">
                             <div className="text-[10px] uppercase tracking-wider text-indigo-400 font-bold mb-2">Sources Referenced</div>
                             <div className="flex flex-wrap gap-2">
                                {msg.citations.slice(0, 3).map((chunk, idx) => (
                                    <span key={idx} className="text-xs bg-white text-indigo-600 px-2.5 py-1.5 rounded-lg border border-indigo-100 shadow-sm flex items-center gap-1 hover:bg-indigo-50 transition-colors cursor-help" title={chunk.text}>
                                        <FileIcon />
                                        <span className="truncate max-w-[150px]">{chunk.docName}</span>
                                        <span className="opacity-60 text-[10px] ml-1">Pg {chunk.pageNumber}</span>
                                    </span>
                                ))}
                             </div>
                        </div>
                    )}
                    
                    {msg.auditLog && (
                        <div className="flex items-center gap-2">
                             <button 
                                onClick={() => setShowAuditFor(showAuditFor === msg.id ? null : msg.id)}
                                className="text-[10px] font-medium flex items-center gap-1 text-slate-400 hover:text-indigo-500 transition-colors bg-white/80 px-2 py-1 rounded-full border border-indigo-100 shadow-sm"
                             >
                                <InfoIcon /> {showAuditFor === msg.id ? 'Hide Reasoning' : 'View Reasoning'}
                             </button>
                        </div>
                    )}
                    
                    {showAuditFor === msg.id && (
                         <div className="mt-2 p-4 bg-slate-900 text-slate-300 rounded-xl text-xs border border-slate-800 shadow-2xl relative overflow-hidden">
                             <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-indigo-500 to-purple-500"></div>
                             
                             {/* Relevance Graph */}
                             {msg.retrievalResults && msg.retrievalResults.length > 0 && (
                               <div className="mb-6">
                                  <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                                    <span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>
                                    Relevance Graph (Cosine Similarity)
                                  </h4>
                                  <div className="space-y-3">
                                    {msg.retrievalResults.map((result, idx) => {
                                        const score = (result.similarity * 100).toFixed(1);
                                        return (
                                          <div key={idx} className="flex flex-col gap-1">
                                            <div className="flex justify-between text-[10px] text-slate-400">
                                               <span className="truncate max-w-[200px]">{result.chunk.docName} (Pg {result.chunk.pageNumber})</span>
                                               <span className="font-mono text-indigo-300">{score}%</span>
                                            </div>
                                            <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                               <div 
                                                  className={`h-full rounded-full transition-all duration-500 ${result.similarity > 0.5 ? 'bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.5)]' : 'bg-blue-400/50'}`} 
                                                  style={{width: `${score}%`}}
                                               ></div>
                                            </div>
                                          </div>
                                        );
                                    })}
                                  </div>
                                  <div className="h-px bg-slate-800 my-4"></div>
                               </div>
                             )}

                             <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Step-by-Step Logic</h4>
                             <div className="font-mono whitespace-pre-wrap leading-relaxed opacity-90">
                                {msg.auditLog}
                             </div>
                         </div>
                    )}
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-6 bg-white/80 backdrop-blur-md border-t border-indigo-100">
          <div className="max-w-4xl mx-auto relative group">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              placeholder={documents.length > 0 ? "Ask a question about your documents..." : "Upload a document to start chatting..."}
              className="w-full pl-6 pr-14 py-4 rounded-2xl border border-indigo-100 bg-white focus:border-indigo-400 focus:ring-4 focus:ring-indigo-500/10 outline-none resize-none shadow-lg shadow-indigo-100/50 transition-all min-h-[64px] max-h-32 text-sm text-slate-700 placeholder:text-slate-400"
              rows={1}
              disabled={isProcessing || documents.length === 0}
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isProcessing || documents.length === 0}
              className="absolute right-3 bottom-3 p-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-indigo-500/30"
            >
              <SendIcon />
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
