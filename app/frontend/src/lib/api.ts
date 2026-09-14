export interface RetrievedCase {
  case_id: string;
  customer_text: string;
  historical_response: string;
  resolution_type: string;
  intent: string;
  similarity: number;
  created_at: string;
}

export interface AnnaResult {
  customer_message: string;
  intent: string;
  intent_confidence: number;
  retrieved_cases: RetrievedCase[];
  evidence: {
    score: number;
    intent_agreement: number;
    resolution_agreement: number;
    similarity_strength: number;
    strong_precedent_ratio: number;
    strong_precedent_count: number;
    usable_case_count: number;
  };
  generation: {
    reply: string;
    confidence: number;
    grounded: boolean;
    escalate: boolean;
    reason: string;
  };
  decision: {
    decision: string;
    auto_handle: boolean;
    reason: string;
    risk_flags: string[];
    factors: Record<string, unknown>;
  };
}

const API_BASE = "http://127.0.0.1:8000";

export async function analyzeMessage(message: string): Promise<AnnaResult> {
  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "ANNA analysis failed.");
  }

  return response.json() as Promise<AnnaResult>;
}
