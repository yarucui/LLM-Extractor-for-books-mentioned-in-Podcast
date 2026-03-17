
const DB_NAME = 'PodcastBookDB';
const DB_VERSION = 1;

export interface Podcast {
  id: string;
  name: string;
  episodeCount?: number;
  totalBookMentions?: number;
}

export interface Episode {
  id: string;
  podcastId: string;
  title: string;
  transcript: string;
  processed: boolean;
  bookCount: number;
}

export interface Book {
  id: string;
  title: string;
  author: string | null;
  totalMentions: number;
  podcastIds: string[];
}

export interface Mention {
  id: string;
  bookTitle: string;
  episodeId: string;
  podcastId: string;
  contextQuote: string;
  mentionType: string;
  strengthIndex: number;
  strengthReason: string;
  authorPresence: string;
  timestamp: string;
}

class LocalDB {
  private db: IDBDatabase | null = null;

  async init(): Promise<IDBDatabase> {
    if (this.db) return this.db;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        if (!db.objectStoreNames.contains('podcasts')) {
          db.createObjectStore('podcasts', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('episodes')) {
          const episodeStore = db.createObjectStore('episodes', { keyPath: 'id' });
          episodeStore.createIndex('podcastId', 'podcastId', { unique: false });
        }
        if (!db.objectStoreNames.contains('books')) {
          db.createObjectStore('books', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('mentions')) {
          const mentionStore = db.createObjectStore('mentions', { keyPath: 'id', autoIncrement: true });
          mentionStore.createIndex('episodeId', 'episodeId', { unique: false });
          mentionStore.createIndex('bookTitle', 'bookTitle', { unique: false });
        }
      };

      request.onsuccess = (event) => {
        this.db = (event.target as IDBOpenDBRequest).result;
        resolve(this.db);
      };

      request.onerror = (event) => {
        reject((event.target as IDBOpenDBRequest).error);
      };
    });
  }

  async getAll<T>(storeName: string): Promise<T[]> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async put<T>(storeName: string, item: T): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.put(item);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async delete(storeName: string, id: string): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.delete(id);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async getById<T>(storeName: string, id: string): Promise<T | undefined> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.get(id);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getEpisodesByPodcast(podcastId: string): Promise<Episode[]> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction('episodes', 'readonly');
      const store = transaction.objectStore('episodes');
      const index = store.index('podcastId');
      const request = index.getAll(podcastId);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async exportAllData(): Promise<string> {
    const podcasts = await this.getAll<Podcast>('podcasts');
    const episodes = await this.getAll<Episode>('episodes');
    const books = await this.getAll<Book>('books');
    const mentions = await this.getAll<Mention>('mentions');

    return JSON.stringify({
      podcasts,
      episodes,
      books,
      mentions,
      exportDate: new Date().toISOString(),
      version: DB_VERSION
    }, null, 2);
  }

  async exportToCSV(storeName: string): Promise<string> {
    const data = await this.getAll<any>(storeName);
    if (data.length === 0) return '';

    const headers = Object.keys(data[0]);
    const csvRows = [
      headers.join(','),
      ...data.map((row: any) => 
        headers.map(fieldName => {
          const value = row[fieldName];
          const escaped = ('' + (value || '')).replace(/"/g, '""');
          return `"${escaped}"`;
        }).join(',')
      )
    ];

    return csvRows.join('\n');
  }
}

export const localDB = new LocalDB();
