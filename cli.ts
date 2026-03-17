import fs from 'fs';
import path from 'path';
import { GoogleGenAI } from "@google/genai";
import dotenv from 'dotenv';
import { SYSTEM_INSTRUCTION, BOOK_EXTRACTION_SCHEMA } from './src/types';

dotenv.config();

const apiKey = process.env.GEMINI_API_KEY;

if (!apiKey) {
  console.error("Error: GEMINI_API_KEY not found in environment variables.");
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey });

async function processFile(filePath: string) {
  try {
    if (!fs.existsSync(filePath)) {
      console.error(`Error: File not found at ${filePath}`);
      return;
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    const episodes = JSON.parse(content);
    const episodesArray = Array.isArray(episodes) ? episodes : [episodes];

    console.log(`Found ${episodesArray.length} episodes. Starting extraction...`);

    const results = [];

    for (const ep of episodesArray) {
      console.log(`Processing: ${ep.title || 'Untitled Episode'}...`);
      
      const transcript = ep.transcript || '';
      if (!transcript) {
        console.warn(`Skipping ${ep.title}: No transcript found.`);
        continue;
      }

      // Chunking logic (similar to web app)
      const CHUNK_SIZE = 100000;
      const chunks = [];
      for (let i = 0; i < transcript.length; i += CHUNK_SIZE) {
        chunks.push(transcript.slice(i, i + CHUNK_SIZE));
      }

      const episodeMentions = [];
      for (let i = 0; i < chunks.length; i++) {
        process.stdout.write(`  Chunk ${i + 1}/${chunks.length}... `);
        const response = await ai.models.generateContent({
          model: "gemini-3-flash-preview",
          contents: chunks[i],
          config: {
            systemInstruction: SYSTEM_INSTRUCTION,
            responseMimeType: "application/json",
            responseSchema: BOOK_EXTRACTION_SCHEMA,
          },
        });
        
        const chunkMentions = JSON.parse(response.text || '[]');
        episodeMentions.push(...chunkMentions);
        console.log(`Done (${chunkMentions.length} books found)`);
      }

      results.push({
        episode: ep.title,
        mentions: episodeMentions
      });
    }

    const outputPath = `extraction_results_${Date.now()}.json`;
    fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
    console.log(`\nSuccess! Results saved to ${outputPath}`);

  } catch (error: any) {
    console.error("An error occurred:", error.message);
  }
}

const filePath = process.argv[2];
if (!filePath) {
  console.log("Usage: npm run cli <path-to-podcast-json>");
  process.exit(1);
}

processFile(filePath);
