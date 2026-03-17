import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { GoogleGenAI } from "@google/genai";
import { useDropzone } from 'react-dropzone';
import { 
  Book as BookIcon, 
  Upload, 
  Loader2, 
  FileText, 
  Trash2, 
  ChevronRight, 
  AlertCircle, 
  CheckCircle2, 
  List, 
  Search, 
  Info, 
  Database, 
  BarChart3, 
  PieChart as PieChartIcon,
  LayoutDashboard,
  FileJson,
  ShieldCheck
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  PieChart, 
  Pie,
  Cell,
  Legend
} from 'recharts';
import { localDB, type Podcast, type Episode, type Book, type Mention } from './db';
import { 
  BOOK_EXTRACTION_SCHEMA, 
  SYSTEM_INSTRUCTION, 
  type BookMention
} from './types';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

export default function App() {
  const [activeTab, setActiveTab] = useState<'upload' | 'dashboard' | 'books'>('dashboard');
  
  // Data State
  const [podcasts, setPodcasts] = useState<Podcast[]>([]);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [books, setBooks] = useState<Book[]>([]);
  const [mentions, setMentions] = useState<Mention[]>([]);
  
  // UI State
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Load Initial Data
  const refreshData = useCallback(async () => {
    try {
      const [p, b, m] = await Promise.all([
        localDB.getAll<Podcast>('podcasts'),
        localDB.getAll<Book>('books'),
        localDB.getAll<Mention>('mentions')
      ]);
      setPodcasts(p);
      setBooks(b.sort((a, b) => b.totalMentions - a.totalMentions));
      setMentions(m.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()));
    } catch (err) {
      console.error("Failed to load local data:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshData();
  }, [refreshData]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    setIsUploading(true);
    setError(null);

    try {
      for (const file of acceptedFiles) {
        const text = await file.text();
        const parsed = JSON.parse(text);
        const episodesArray = Array.isArray(parsed) ? parsed : [parsed];

        for (const epData of episodesArray) {
          const podcastId = epData.podcast?.podcast_id || 'unknown';
          const podcastName = epData.podcast?.podcast_name || 'Unknown Podcast';
          
          // Ensure Podcast exists
          const existingPodcast = await localDB.getById<Podcast>('podcasts', podcastId);
          if (!existingPodcast) {
            await localDB.put('podcasts', { 
              name: podcastName,
              id: podcastId,
              episodeCount: 0,
              totalBookMentions: 0
            });
          }

          // Create Episode
          const epId = epData.episode_id || Math.random().toString(36).substring(7);
          await localDB.put('episodes', {
            id: epId,
            podcastId,
            title: epData.episode_title || 'Untitled Episode',
            transcript: epData.episode_transcript || '',
            processed: false,
            bookCount: 0
          });
        }
      }
      await refreshData();
    } catch (err: any) {
      console.error(err);
      setError(`Upload failed: ${err.message || "Ensure format is correct."}`);
    } finally {
      setIsUploading(false);
    }
  }, [refreshData]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/json': ['.json'] },
    multiple: true
  } as any);

  const [selectedPodcastId, setSelectedPodcastId] = useState<string | null>(null);
  const [podcastEpisodes, setPodcastEpisodes] = useState<Episode[]>([]);

  useEffect(() => {
    if (selectedPodcastId) {
      localDB.getEpisodesByPodcast(selectedPodcastId).then(setPodcastEpisodes);
    } else {
      setPodcastEpisodes([]);
    }
  }, [selectedPodcastId, podcasts]);

  const handleDeletePodcast = async (id: string) => {
    if (!confirm("Are you sure you want to delete this podcast and all its episodes?")) return;
    try {
      await localDB.delete('podcasts', id);
      const eps = await localDB.getEpisodesByPodcast(id);
      for (const ep of eps) {
        await localDB.delete('episodes', ep.id);
      }
      await refreshData();
      if (selectedPodcastId === id) setSelectedPodcastId(null);
    } catch (err) {
      console.error("Delete failed:", err);
    }
  };
  const [processingProgress, setProcessingProgress] = useState<{ current: number, total: number } | null>(null);
  const [isBulkProcessing, setIsBulkProcessing] = useState(false);

  const processEpisode = async (episode: Episode) => {
    if (processingId && !isBulkProcessing) return;
    setProcessingId(episode.id);
    setProcessingProgress(null);
    setError(null);

    try {
      const CHUNK_SIZE = 100000; // ~100KB per chunk
      const transcript = episode.transcript;
      const chunks: string[] = [];
      
      for (let i = 0; i < transcript.length; i += CHUNK_SIZE) {
        chunks.push(transcript.slice(i, i + CHUNK_SIZE));
      }

      const allExtractedMentions: BookMention[] = [];
      setProcessingProgress({ current: 0, total: chunks.length });

      for (let i = 0; i < chunks.length; i++) {
        setProcessingProgress({ current: i + 1, total: chunks.length });
        const response = await ai.models.generateContent({
          model: "gemini-3-flash-preview",
          contents: chunks[i],
          config: {
            systemInstruction: SYSTEM_INSTRUCTION,
            responseMimeType: "application/json",
            responseSchema: BOOK_EXTRACTION_SCHEMA,
          },
        });

        const chunkMentions = JSON.parse(response.text || '[]') as BookMention[];
        allExtractedMentions.push(...chunkMentions);
      }

      // Deduplicate mentions by book title (case-insensitive)
      const uniqueMentionsMap = new Map<string, BookMention>();
      allExtractedMentions.forEach(m => {
        const key = m.book_title.toLowerCase().trim();
        if (!uniqueMentionsMap.has(key) || m.recommendation_strength_index > (uniqueMentionsMap.get(key)?.recommendation_strength_index || 0)) {
          uniqueMentionsMap.set(key, m);
        }
      });

      const finalMentions = Array.from(uniqueMentionsMap.values());

      for (const m of finalMentions) {
        const bookId = m.book_title.toLowerCase().replace(/[^a-z0-9]/g, '-');
        
        // Update Book Aggregation
        const existingBook = await localDB.getById<Book>('books', bookId);
        if (existingBook) {
          existingBook.totalMentions += 1;
          if (!existingBook.podcastIds.includes(episode.podcastId)) {
            existingBook.podcastIds.push(episode.podcastId);
          }
          await localDB.put('books', existingBook);
        } else {
          await localDB.put('books', {
            id: bookId,
            title: m.book_title,
            author: m.author_name,
            totalMentions: 1,
            podcastIds: [episode.podcastId]
          });
        }

        // Create Mention Record
        await localDB.put('mentions', {
          id: Math.random().toString(36).substring(7),
          bookTitle: m.book_title,
          episodeId: episode.id,
          podcastId: episode.podcastId,
          contextQuote: m.context_quote,
          mentionType: m.mention_type,
          strengthIndex: m.recommendation_strength_index,
          strengthReason: m.recommendation_reason,
          authorPresence: m.author_presence,
          timestamp: new Date().toISOString()
        });
      }

      // Mark Episode as Processed
      await localDB.put('episodes', { 
        ...episode,
        processed: true, 
        bookCount: finalMentions.length 
      });

      // Update Podcast Stats
      const podcast = await localDB.getById<Podcast>('podcasts', episode.podcastId);
      if (podcast) {
        podcast.episodeCount = (podcast.episodeCount || 0) + 1;
        podcast.totalBookMentions = (podcast.totalBookMentions || 0) + finalMentions.length;
        await localDB.put('podcasts', podcast);
      }

      await refreshData();
    } catch (err: any) {
      console.error(err);
      setError(`Failed to process "${episode.title}": ${err.message || 'Unknown error'}`);
      throw err; // Re-throw for bulk processing to stop or handle
    } finally {
      if (!isBulkProcessing) setProcessingId(null);
    }
  };

  const processAllEpisodes = async () => {
    if (!selectedPodcastId || isBulkProcessing) return;
    setIsBulkProcessing(true);
    setError(null);

    const unprocessed = podcastEpisodes.filter(ep => !ep.processed);
    
    try {
      for (const ep of unprocessed) {
        await processEpisode(ep);
      }
    } catch (err) {
      console.error("Bulk processing interrupted:", err);
    } finally {
      setIsBulkProcessing(false);
      setProcessingId(null);
      setProcessingProgress(null);
    }
  };

  const handleExport = async () => {
    try {
      const data = await localDB.exportAllData();
      const blob = new Blob([data], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `podcast_book_db_export_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export failed:", err);
      setError("Failed to export database.");
    }
  };

  // Charts Data
  const chartData = useMemo(() => {
    return books.slice(0, 10).map(b => ({
      name: b.title.length > 20 ? b.title.substring(0, 20) + '...' : b.title,
      mentions: b.totalMentions
    }));
  }, [books]);

  const typeData = useMemo(() => {
    const counts: Record<string, number> = {};
    mentions.forEach(m => {
      counts[m.mentionType] = (counts[m.mentionType] || 0) + 1;
    });
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [mentions]);

  const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444'];

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#f5f5f5]">
        <Loader2 className="w-8 h-8 text-emerald-600 animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#f5f5f5] text-[#1a1a1a] font-sans selection:bg-emerald-100 selection:text-emerald-900">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-xl border-b border-black/5 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-emerald-600 rounded-xl flex items-center justify-center shadow-lg shadow-emerald-600/20">
              <BookIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <span className="font-bold text-xl tracking-tight block">PodcastDB</span>
              <div className="flex items-center gap-1 text-[10px] text-emerald-600 font-bold uppercase tracking-widest">
                <ShieldCheck className="w-3 h-3" />
                Private & Local
              </div>
            </div>
          </div>

          <div className="flex items-center gap-1 bg-[#f5f5f5] p-1 rounded-2xl border border-black/5">
            {[
              { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard' },
              { id: 'upload', icon: Upload, label: 'Upload' },
              { id: 'books', icon: List, label: 'Library' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all",
                  activeTab === tab.id 
                    ? "bg-white text-emerald-600 shadow-sm" 
                    : "text-[#9e9e9e] hover:text-[#1a1a1a]"
                )}
              >
                <tab.icon className="w-4 h-4" />
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            ))}
          </div>

          <div className="hidden md:flex items-center gap-4">
            <div className="flex flex-col items-end">
              <span className="text-xs font-bold">Local Storage</span>
              <span className="text-[10px] text-[#9e9e9e]">No Cloud Sync</span>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-12">
        <AnimatePresence mode="wait">
          {activeTab === 'dashboard' && (
            <motion.div 
              key="dashboard"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-8"
            >
              {/* Stats Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {[
                  { label: 'Total Podcasts', value: podcasts.length, icon: Database, color: 'text-blue-600', bg: 'bg-blue-50' },
                  { label: 'Books Found', value: books.length, icon: BookIcon, color: 'text-emerald-600', bg: 'bg-emerald-50' },
                  { label: 'Total Mentions', value: mentions.length, icon: FileText, color: 'text-orange-600', bg: 'bg-orange-50' },
                  { label: 'Strong Recs', value: mentions.filter(m => m.mentionType === 'Strong Recommendation').length, icon: CheckCircle2, color: 'text-purple-600', bg: 'bg-purple-50' }
                ].map((stat, i) => (
                  <div key={i} className="bg-white p-6 rounded-3xl border border-black/5 shadow-sm">
                    <div className={cn("w-12 h-12 rounded-2xl flex items-center justify-center mb-4", stat.bg)}>
                      <stat.icon className={cn("w-6 h-6", stat.color)} />
                    </div>
                    <p className="text-xs font-bold text-[#9e9e9e] uppercase tracking-widest mb-1">{stat.label}</p>
                    <p className="text-3xl font-bold">{stat.value}</p>
                  </div>
                ))}
                
                <div className="bg-white p-6 rounded-3xl border border-black/5 shadow-sm flex flex-col justify-center items-center text-center group hover:border-emerald-200 transition-all cursor-pointer" onClick={handleExport}>
                  <div className="w-12 h-12 rounded-2xl bg-emerald-50 flex items-center justify-center mb-4 group-hover:bg-emerald-500 transition-all">
                    <FileJson className="w-6 h-6 text-emerald-600 group-hover:text-white transition-all" />
                  </div>
                  <p className="text-xs font-bold text-[#9e9e9e] uppercase tracking-widest mb-1">Export Data</p>
                  <p className="text-sm font-bold text-emerald-600">Download JSON for DBeaver</p>
                </div>

                <div className="bg-white p-6 rounded-3xl border border-black/5 shadow-sm flex flex-col justify-center items-center text-center group hover:border-emerald-200 transition-all cursor-pointer" onClick={async () => {
                  try {
                    const csv = await localDB.exportToCSV('mentions');
                    const blob = new Blob([csv], { type: 'text/csv' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `podcast_mentions_${new Date().toISOString().split('T')[0]}.csv`;
                    a.click();
                    URL.revokeObjectURL(url);
                  } catch (err) {
                    setError("Failed to export CSV.");
                  }
                }}>
                  <div className="w-12 h-12 rounded-2xl bg-emerald-50 flex items-center justify-center mb-4 group-hover:bg-emerald-500 transition-all">
                    <List className="w-6 h-6 text-emerald-600 group-hover:text-white transition-all" />
                  </div>
                  <p className="text-xs font-bold text-[#9e9e9e] uppercase tracking-widest mb-1">Export Mentions</p>
                  <p className="text-sm font-bold text-emerald-600">Download CSV for Analysis</p>
                </div>

                <div className="bg-white p-6 rounded-3xl border border-black/5 shadow-sm flex flex-col justify-center items-center text-center group hover:border-red-200 transition-all cursor-pointer" onClick={async () => {
                  if(confirm("Clear all local data? This cannot be undone.")) {
                    const db = await indexedDB.open('PodcastBookDB', 1);
                    db.onsuccess = (e) => {
                      const database = (e.target as any).result;
                      const transaction = database.transaction(['podcasts', 'episodes', 'books', 'mentions'], 'readwrite');
                      transaction.objectStore('podcasts').clear();
                      transaction.objectStore('episodes').clear();
                      transaction.objectStore('books').clear();
                      transaction.objectStore('mentions').clear();
                      transaction.oncomplete = () => refreshData();
                    };
                  }
                }}>
                  <div className="w-12 h-12 rounded-2xl bg-red-50 flex items-center justify-center mb-4 group-hover:bg-red-500 transition-all">
                    <Trash2 className="w-6 h-6 text-red-600 group-hover:text-white transition-all" />
                  </div>
                  <p className="text-xs font-bold text-[#9e9e9e] uppercase tracking-widest mb-1">Reset Database</p>
                  <p className="text-sm font-bold text-red-600">Clear All Local Data</p>
                </div>
              </div>

              {/* Charts Section */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm">
                  <div className="flex items-center justify-between mb-8">
                    <h3 className="font-bold flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-emerald-600" />
                      Top Mentioned Books
                    </h3>
                  </div>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                        <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#9e9e9e' }} />
                        <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#9e9e9e' }} />
                        <Tooltip 
                          contentStyle={{ borderRadius: '16px', border: 'none', boxShadow: '0 10px 30px rgba(0,0,0,0.1)' }}
                          cursor={{ fill: '#f9f9f9' }}
                        />
                        <Bar dataKey="mentions" fill="#10b981" radius={[6, 6, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm">
                  <div className="flex items-center justify-between mb-8">
                    <h3 className="font-bold flex items-center gap-2">
                      <PieChartIcon className="w-5 h-5 text-blue-600" />
                      Mention Types
                    </h3>
                  </div>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={typeData}
                          innerRadius={60}
                          outerRadius={100}
                          paddingAngle={8}
                          dataKey="value"
                        >
                          {typeData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip contentStyle={{ borderRadius: '16px', border: 'none' }} />
                        <Legend verticalAlign="bottom" height={36} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Recent Mentions Feed */}
              <div className="bg-white rounded-3xl border border-black/5 shadow-sm overflow-hidden">
                <div className="p-8 border-b border-black/5">
                  <h3 className="font-bold">Recent Mentions</h3>
                </div>
                <div className="divide-y divide-black/5">
                  {mentions.slice(0, 10).map((mention) => (
                    <div key={mention.id} className="p-6 hover:bg-[#f9f9f9] transition-colors group">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-lg font-bold group-hover:text-emerald-600 transition-colors">{mention.bookTitle}</span>
                            <span className={cn(
                              "text-[10px] font-bold uppercase px-2 py-0.5 rounded-md border",
                              mention.mentionType === 'Strong Recommendation' ? "bg-emerald-50 text-emerald-600 border-emerald-100" : "bg-gray-50 text-gray-500 border-gray-100"
                            )}>
                              {mention.mentionType}
                            </span>
                          </div>
                          <p className="text-xs text-[#9e9e9e] mb-3">
                            Mentioned in <span className="font-semibold text-[#1a1a1a]">{podcasts.find(p => p.id === mention.podcastId)?.name}</span>
                          </p>
                          {mention.contextQuote && (
                            <p className="text-sm italic text-[#4a4a4a] border-l-2 border-emerald-500/30 pl-4 py-1">
                              "{mention.contextQuote}"
                            </p>
                          )}
                        </div>
                        <div className="text-right">
                          <div className="flex items-center gap-1 text-emerald-600 font-bold mb-1">
                            <span className="text-sm">{mention.strengthIndex}/10</span>
                          </div>
                          <span className="text-[10px] text-[#9e9e9e]">{new Date(mention.timestamp).toLocaleDateString()}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'upload' && (
            <motion.div 
              key="upload"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              className="max-w-4xl mx-auto space-y-8"
            >
              <div className="bg-white rounded-3xl p-12 border border-black/5 shadow-sm text-center">
                <div 
                  {...getRootProps()} 
                  className={cn(
                    "border-2 border-dashed rounded-3xl p-16 transition-all cursor-pointer",
                    isDragActive ? "border-emerald-500 bg-emerald-50/50" : "border-black/5 hover:border-black/10 hover:bg-black/[0.01]"
                  )}
                >
                  <input {...getInputProps()} />
                  <div className="w-20 h-20 bg-emerald-50 rounded-2xl flex items-center justify-center mx-auto mb-6">
                    {isUploading ? <Loader2 className="w-10 h-10 text-emerald-600 animate-spin" /> : <Upload className="w-10 h-10 text-emerald-600" />}
                  </div>
                  <h3 className="text-xl font-bold mb-2">Upload Podcast JSON</h3>
                  <p className="text-[#9e9e9e] mb-6">Drop your episode list files here. Supports batch upload up to 200 files.</p>
                  <div className="flex items-center justify-center gap-2 text-xs font-bold text-emerald-600 uppercase tracking-widest">
                    <FileJson className="w-4 h-4" />
                    JSON Format Required
                  </div>
                </div>
              </div>

              {/* Pending Episodes List */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="bg-white rounded-3xl border border-black/5 shadow-sm overflow-hidden h-[600px] flex flex-col">
                  <div className="p-6 border-b border-black/5">
                    <h3 className="font-bold flex items-center gap-2">
                      <Database className="w-4 h-4 text-emerald-600" />
                      Podcasts
                    </h3>
                  </div>
                  <div className="flex-1 overflow-y-auto divide-y divide-black/5">
                    {podcasts.length === 0 && (
                      <div className="p-12 text-center text-[#9e9e9e]">
                        <p className="text-sm">No podcasts uploaded yet.</p>
                      </div>
                    )}
                    {podcasts.map(podcast => (
                      <div 
                        key={podcast.id} 
                        onClick={() => setSelectedPodcastId(podcast.id)}
                        className={cn(
                          "w-full p-4 text-left transition-all hover:bg-[#f9f9f9] flex items-center justify-between group cursor-pointer",
                          selectedPodcastId === podcast.id ? "bg-emerald-50/50 border-r-4 border-emerald-500" : ""
                        )}
                      >
                        <div>
                          <p className="font-bold text-sm">{podcast.name}</p>
                          <p className="text-[10px] text-[#9e9e9e] uppercase tracking-widest">
                            {podcast.episodeCount || 0} Processed
                          </p>
                        </div>
                        <button 
                          onClick={(e) => { e.stopPropagation(); handleDeletePodcast(podcast.id); }}
                          className="p-2 text-[#9e9e9e] hover:text-red-500 opacity-0 group-hover:opacity-100 transition-all"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="lg:col-span-2 bg-white rounded-3xl border border-black/5 shadow-sm overflow-hidden h-[600px] flex flex-col">
                  <div className="p-6 border-b border-black/5 flex items-center justify-between">
                    <h3 className="font-bold flex items-center gap-2">
                      <List className="w-4 h-4 text-emerald-600" />
                      Episodes
                    </h3>
                    <div className="flex items-center gap-4">
                      {selectedPodcastId && podcastEpisodes.some(ep => !ep.processed) && (
                        <button 
                          onClick={processAllEpisodes}
                          disabled={isBulkProcessing || !!processingId}
                          className="px-4 py-2 bg-emerald-600 text-white text-xs font-bold rounded-xl hover:bg-emerald-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                        >
                          {isBulkProcessing ? <Loader2 className="w-3 h-3 animate-spin" /> : <CheckCircle2 className="w-3 h-3" />}
                          Process All
                        </button>
                      )}
                      {selectedPodcastId && (
                        <span className="text-[10px] font-bold text-[#9e9e9e] uppercase tracking-widest">
                          {podcastEpisodes.length} Total
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex-1 overflow-y-auto divide-y divide-black/5">
                    {!selectedPodcastId ? (
                      <div className="h-full flex flex-col items-center justify-center text-[#9e9e9e] p-12 text-center">
                        <ChevronRight className="w-8 h-8 mb-4 opacity-20" />
                        <p className="text-sm">Select a podcast to view and process episodes.</p>
                      </div>
                    ) : podcastEpisodes.length === 0 ? (
                      <div className="p-12 text-center text-[#9e9e9e]">
                        <p className="text-sm">No episodes found for this podcast.</p>
                      </div>
                    ) : (
                      podcastEpisodes.map(episode => (
                        <div key={episode.id} className="p-6 flex items-center justify-between hover:bg-[#f9f9f9] transition-colors">
                          <div className="flex-1 min-w-0 pr-4">
                            <h4 className="font-bold text-sm truncate mb-1">{episode.title}</h4>
                            <div className="flex items-center gap-3">
                              <span className={cn(
                                "text-[8px] font-bold uppercase px-1.5 py-0.5 rounded",
                                episode.processed ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"
                              )}>
                                {episode.processed ? 'Processed' : 'Pending'}
                              </span>
                              <span className="text-[10px] text-[#9e9e9e]">
                                {Math.round(episode.transcript.length / 1000)}k characters
                              </span>
                            </div>
                          </div>
                          {!episode.processed && (
                            <button 
                              onClick={() => processEpisode(episode)}
                              disabled={!!processingId}
                              className="px-4 py-2 bg-emerald-600 text-white text-xs font-bold rounded-xl hover:bg-emerald-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                            >
                              {processingId === episode.id ? <Loader2 className="w-3 h-3 animate-spin" /> : <FileText className="w-3 h-3" />}
                              Process
                            </button>
                          )}
                          {episode.processed && (
                            <div className="flex items-center gap-1 text-emerald-600">
                              <CheckCircle2 className="w-4 h-4" />
                              <span className="text-xs font-bold">{episode.bookCount} Books</span>
                            </div>
                          )}
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'books' && (
            <motion.div 
              key="books"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-8"
            >
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div>
                  <h2 className="text-3xl font-bold tracking-tight">Book Library</h2>
                  <p className="text-[#9e9e9e]">Explore every book mentioned across your podcast collection.</p>
                </div>
                <div className="relative w-full md:w-96">
                  <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[#9e9e9e]" />
                  <input 
                    type="text" 
                    placeholder="Search titles, authors..." 
                    className="w-full pl-12 pr-6 py-4 bg-white rounded-2xl border border-black/5 shadow-sm focus:ring-2 focus:ring-emerald-500/20 transition-all"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {books.map((book) => (
                  <motion.div 
                    layout
                    key={book.id}
                    className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm hover:border-emerald-500/30 transition-all group"
                  >
                    <div className="flex items-start justify-between mb-6">
                      <div className="w-12 h-12 bg-emerald-50 rounded-2xl flex items-center justify-center group-hover:bg-emerald-600 transition-colors">
                        <BookIcon className="w-6 h-6 text-emerald-600 group-hover:text-white transition-colors" />
                      </div>
                      <div className="text-right">
                        <p className="text-2xl font-bold text-emerald-600">{book.totalMentions}</p>
                        <p className="text-[10px] font-bold text-[#9e9e9e] uppercase tracking-widest">Mentions</p>
                      </div>
                    </div>
                    <h3 className="text-xl font-bold mb-1 line-clamp-2">{book.title}</h3>
                    <p className="text-sm text-[#9e9e9e] mb-6">by {book.author || 'Unknown Author'}</p>
                    
                    <div className="flex items-center gap-2 pt-6 border-t border-black/5">
                      <div className="flex -space-x-2 overflow-hidden">
                        {[1, 2, 3].map(i => (
                          <div key={i} className="inline-block h-6 w-6 rounded-full ring-2 ring-white bg-[#f5f5f5] flex items-center justify-center text-[8px] font-bold text-[#9e9e9e]">
                            P{i}
                          </div>
                        ))}
                      </div>
                      <span className="text-[10px] font-bold text-[#9e9e9e] uppercase tracking-widest">
                        In {book.podcastIds.length} Podcasts
                      </span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Processing Overlay */}
      <AnimatePresence>
        {(processingId || isBulkProcessing) && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-black/20 backdrop-blur-sm flex items-center justify-center p-6"
          >
            <div className="bg-white rounded-3xl p-8 shadow-2xl border border-black/5 max-w-sm w-full text-center">
              <div className="w-16 h-16 bg-emerald-50 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <Loader2 className="w-8 h-8 text-emerald-600 animate-spin" />
              </div>
              <h3 className="text-xl font-bold mb-2">
                {isBulkProcessing ? 'Bulk Processing' : 'Processing Transcript'}
              </h3>
              <p className="text-[#9e9e9e] text-sm mb-6">
                {isBulkProcessing 
                  ? `Processing all episodes in sequence. Please keep this tab open.`
                  : `Gemini is analyzing the transcript for book mentions. This may take a moment.`}
              </p>
              
              {isBulkProcessing && (
                <div className="mb-6 p-4 bg-emerald-50 rounded-2xl">
                  <p className="text-xs font-bold text-emerald-600 uppercase tracking-widest mb-1">Overall Progress</p>
                  <p className="text-lg font-bold text-emerald-900">
                    {podcastEpisodes.filter(ep => ep.processed).length} / {podcastEpisodes.length} Episodes
                  </p>
                </div>
              )}

              {processingProgress && (
                <div className="space-y-2">
                  <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest text-[#9e9e9e]">
                    <span>Current Episode Progress</span>
                    <span>{Math.round((processingProgress.current / processingProgress.total) * 100)}%</span>
                  </div>
                  <div className="w-full h-2 bg-[#f5f5f5] rounded-full overflow-hidden">
                    <motion.div 
                      className="h-full bg-emerald-500"
                      initial={{ width: 0 }}
                      animate={{ width: `${(processingProgress.current / processingProgress.total) * 100}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-[#9e9e9e]">
                    Chunk {processingProgress.current} of {processingProgress.total}
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Toast */}
      <AnimatePresence>
        {error && (
          <motion.div 
            initial={{ opacity: 0, y: 100 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 100 }}
            className="fixed bottom-8 left-1/2 -translate-x-1/2 z-[100] bg-red-600 text-white px-6 py-4 rounded-2xl shadow-2xl flex items-center gap-3"
          >
            <AlertCircle className="w-5 h-5" />
            <span className="text-sm font-semibold">{error}</span>
            <button onClick={() => setError(null)} className="ml-4 p-1 hover:bg-white/10 rounded-lg transition-colors">
              <Trash2 className="w-4 h-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
