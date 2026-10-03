/**
 * ResearchPilotAI - Full-Stack Express Server with Gemini Multi-Agent Orchestration
 * Mounts Vite dev middlewares in development and exposes specialized AI agent endpoints.
 */

import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';
import { createServer as createViteServer } from 'vite';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: '15mb' }));

// Server-side initialization of GoogleGenAI SDK with required telemetry headers
const apiKey = process.env.GEMINI_API_KEY || '';
let ai: GoogleGenAI | null = null;
if (apiKey) {
  ai = new GoogleGenAI({
    apiKey: apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Multi-Agent Investigation Endpoint
app.post('/api/agents/generate-investigation', async (req, res) => {
  try {
    const { question, discipline, goal, csvData } = req.body;

    if (!question) {
      return res.status(400).json({ error: 'Research question is required.' });
    }

    if (!ai) {
      // Graceful fallback response when API key is not yet set in environment
      return res.json({
        fallback: true,
        message: 'GEMINI_API_KEY not provided; using high-fidelity local deterministic agent pipeline.',
      });
    }

    // Call Gemini 3.8 Flash with structured JSON output instructions
    const prompt = `You are ResearchPilotAI, an expert multi-agent scientific research workspace.
You are conducting a full 8-agent investigation for this user research question:
Research Question: "${question}"
Discipline: "${discipline || 'Auto Detect'}"
Goal: "${goal || 'Explore a question'}"

You must orchestrate the following 8 logical agents:
1. Research Planner (refines question, specifies discipline, objectives, subquestions, methodology, data requirements)
2. Literature Discovery Agent (real, authentic peer-reviewed literature - never fabricate citations, include real authors, year, journal, DOI if known, key finding)
3. Evidence Classifier (labels claims with VERIFIED_SOURCE, DATA_DERIVED, AI_INTERPRETATION, HYPOTHESIS, INSUFFICIENT_EVIDENCE)
4. Research Gap Detector (established knowledge, evidence gaps with cautious phrasing like "appears underexplored", contradictions, methodological gaps, population gaps, potential opportunities)
5. Hypothesis Generator (formulates testable PROPOSED HYPOTHESES with independent & dependent variables, methodology, expected relationship, limitations)
6. Research Investigator (synthesis: what is known, what evidence was found, what data shows, patterns, uncertainties, next steps)
7. Mini Paper Generator (14-section academic paper: Title, Abstract, Keywords, 1. Introduction, 2. Research Problem, 3. Research Question, 4. Objectives, 5. Literature Review, 6. Research Gap, 7. Hypothesis, 8. Methodology, 9. Data and Analysis, 10. Results, 11. Discussion, 12. Limitations, 13. Conclusion, 14. Future Research, References)
8. Research-to-Prototype Engine (Problem, Target Users, Proposed Solution, Core Features, User Journey, AI Component, Data Requirements, System Architecture, Technical Stack, Prototype Spec)

Return ONLY valid JSON matching this schema:
{
  "title": string,
  "discipline": string,
  "plan": {
    "refined_question": string,
    "discipline": string,
    "research_objectives": string[],
    "sub_questions": string[],
    "suggested_methodology": string,
    "required_evidence": string[],
    "data_requirements": string[]
  },
  "sources": [
    {
      "id": string,
      "title": string,
      "authors": string[],
      "year": number,
      "source": string,
      "urlOrDoi": string,
      "relevance": string,
      "keyFinding": string,
      "evidenceStatus": "VERIFIED_SOURCE",
      "isVerified": true
    }
  ],
  "evidenceClaims": [
    {
      "id": string,
      "claim": string,
      "status": "VERIFIED_SOURCE" | "DATA_DERIVED" | "AI_INTERPRETATION" | "HYPOTHESIS" | "INSUFFICIENT_EVIDENCE",
      "sourceIds": string[],
      "explanation": string,
      "confidence": "High" | "Moderate" | "Tentative" | "Insufficient"
    }
  ],
  "gaps": {
    "establishedKnowledge": [{ "title": string, "description": string, "supportingSources": string[] }],
    "evidenceGaps": [{ "gap": string, "whyItMatters": string, "cautiousNote": string }],
    "contradictions": [{ "topic": string, "sideA": string, "sideB": string, "resolutionStatus": string }],
    "methodologicalGaps": [{ "area": string, "limitationInLiterature": string, "suggestedApproach": string }],
    "populationGeographicGaps": [{ "underrepresentedDomain": string, "context": string }],
    "potentialOpportunities": [{ "opportunity": string, "feasibility": string }]
  },
  "hypotheses": [
    {
      "id": string,
      "hypothesis": string,
      "rationale": string,
      "supportingEvidence": string[],
      "independentVariable": string,
      "dependentVariable": string,
      "suggestedMethodology": string,
      "expectedRelationship": string,
      "limitations": string[],
      "statusLabel": "PROPOSED HYPOTHESIS"
    }
  ],
  "synthesis": {
    "whatIsKnown": string,
    "whatEvidenceFound": string,
    "whatDataShows": string,
    "patternsIdentified": string[],
    "whatRemainsUncertain": string,
    "whatResearchGapExists": string,
    "whatHypothesisEmerges": string,
    "whatShouldBeInvestigatedNext": string
  },
  "paper": {
    "title": string,
    "abstract": string,
    "keywords": string[],
    "sections": {
      "introduction": string,
      "researchProblem": string,
      "researchQuestion": string,
      "objectives": string[],
      "literatureReview": string,
      "researchGap": string,
      "hypothesis": string,
      "methodology": string,
      "dataAndAnalysis": string,
      "results": string,
      "discussion": string,
      "limitations": string,
      "conclusion": string,
      "futureResearch": string
    },
    "references": [
      { "id": string, "citationText": string, "url": string }
    ]
  },
  "prototype": {
    "feasible": true,
    "problem": string,
    "targetUsers": string,
    "proposedSolution": string,
    "prototypeType": "web-application" | "dashboard" | "prediction-system" | "simulation" | "monitoring-system",
    "coreFeatures": string[],
    "userJourney": string[],
    "aiComponent": string,
    "dataRequirements": string,
    "systemArchitecture": string,
    "technicalStack": string,
    "specificationDetails": string,
    "interactiveComponentId": "DynamicScenarioPrototype",
    "disclaimer": "RESEARCH PROTOTYPE — NOT A DEPLOYED PRODUCT"
  }
}`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
      },
    });

    const text = response.text || '{}';
    const parsedData = JSON.parse(text);
    return res.json(parsedData);
  } catch (error: any) {
    console.error('Agent Orchestration Error:', error);
    return res.status(500).json({
      error: 'Agent pipeline execution encountered an error.',
      details: error.message,
    });
  }
});

// Vite Middleware integration for development
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: process.env.DISABLE_HMR !== 'true',
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(Number(PORT), '0.0.0.0', () => {
    console.log(`ResearchPilotAI server active at http://localhost:${PORT}`);
  });
}

startServer();
