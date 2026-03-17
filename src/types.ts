import { Type } from "@google/genai";

export interface BookMention {
  book_title: string;
  author_name: string | null;
  context_quote: string | null;
  mention_type: "Strong Recommendation" | "Casual Mention" | "Author Interview" | "Critical Mention";
  recommendation_strength_index: number;
  recommendation_reason: string;
  author_presence: boolean;
}

export interface Podcast {
  id: string;
  name: string;
  episodeCount: number;
  totalBookMentions: number;
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
  contextQuote: string | null;
  mentionType: string;
  strengthIndex: number;
  strengthReason: string;
  authorPresence: boolean;
  timestamp: string;
}

export const BOOK_EXTRACTION_SCHEMA = {
  type: Type.ARRAY,
  items: {
    type: Type.OBJECT,
    properties: {
      book_title: { type: Type.STRING },
      author_name: { type: Type.STRING, nullable: true },
      context_quote: { type: Type.STRING, nullable: true },
      mention_type: { 
        type: Type.STRING, 
        enum: ["Strong Recommendation", "Casual Mention", "Author Interview", "Critical Mention"] 
      },
      recommendation_strength_index: { type: Type.INTEGER },
      recommendation_reason: { type: Type.STRING },
      author_presence: { type: Type.BOOLEAN },
    },
    required: ["book_title", "mention_type", "recommendation_strength_index", "recommendation_reason", "author_presence"],
  },
};

export const SYSTEM_INSTRUCTION = `You are a precise information extraction system.
Extract books explicitly mentioned in a podcast episode transcript.
Rules:
1. Only extract books.
2. Exclude other media.
3. Set author_name to null unless explicitly stated.
4. Use exact wording for context_quote.
5. Assign recommendation_strength_index (1-10) and reason.
6. Set author_presence to true if the author is a guest.`;
