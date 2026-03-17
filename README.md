# Podcast Book Extractor

A full-stack application to extract book mentions from podcast transcripts using Gemini AI.

## Local Setup

This project uses **Node.js** and **TypeScript**. There is no `main.py` as it is a web-based application with a CLI tool built in TypeScript.

### 1. Prerequisites
- Install [Node.js](https://nodejs.org/) (v18 or higher)
- A Gemini API Key

### 2. Installation
Open your terminal in your IDE and run:
```bash
npm install
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your API key:
```env
GEMINI_API_KEY=your_api_key_here
```

### 4. Running the Web App
To start the interactive dashboard:
```bash
npm run dev
```
Then open `http://localhost:3000` in your browser.

### 5. Running via Terminal (CLI)
You have two options for running via terminal:

#### Option A: TypeScript (Recommended)
```bash
npm run cli path/to/your/podcast.json
```

#### Option B: Python
```bash
pip install google-generativeai python-dotenv
python main.py path/to/your/podcast.json
```
Both options will process all episodes in the JSON file and save the results to a new JSON file.

## Project Structure
- `src/App.tsx`: Main React dashboard.
- `src/db.ts`: Local database (IndexedDB) logic.
- `cli.ts`: Terminal-based extraction tool.
- `src/types.ts`: Shared schemas and AI instructions.
