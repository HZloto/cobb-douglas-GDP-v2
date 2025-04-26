# agent.py  ───────────────────────────────────────────────────────────────
# Requires:  pip install google-genai
import base64
import os
from google import genai
from google.genai import types


def generate(prompt: str) -> None:
    """
    Streams a JSON object to stdout that describes one or more KPI-impact
    rules ready for the Cobb-Douglas simulator.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    # ── Updated system instructions ────────────────────────────────────
    system_txt = """
You are an expert macro-economist for Saudi Arabia.  
Translate *any* natural-language policy scenario into a **machine-readable**
list of impact-rules that our Cobb-Douglas simulator can ingest.

════════════════════════════════════════════════════════════════════
RULE FORMAT  (all numbers are **percent**, not decimals)
════════════════════════════════════════════════════════════════════
{
  "region": 3-letter code  | "all"
  "sector": ISIC letter    | "all"
  "kpi":    "Productivity" | "Workforce" | "Investments" | "all"
  "years":  {"2025":10, …, "2030":10}
}

• **Positive values** = percentage increases; **negative** = decreases.  
• **Target the narrowest sensible set of sectors. Never default to
  sector:"all" if the policy mentions — even implicitly — a specific
  sector or industry.**  
• If several distinct sectors are hit, create **multiple rule objects**,
  one per sector (or per region/sector pair if they differ).  
• Only use `"all"` when *every single ISIC sector* is clearly affected
  in the same direction **and** magnitude.

Quick word-to-sector hints
  tourism / hospitality       → "I"  (Accommodation & Food)
  hotels / restaurants        → "I"
  culture / entertainment     → "R"
  oil / crude / refinery      → "B"  (Mining & quarrying incl. oil)
  manufacturing / factory     → "C"
  construction / housing      → "F"
  retail / wholesale / shops  → "G"
  transport / logistics       → "H"
  health / hospitals          → "Q"
  education / schools         → "P"

  always try to match to a specific region unless the policy is country-wide.
  Regions dictionary: 
Asir - ASI
Al Bahah - ABA
Al Hudud ash Shamaliyah - AHU
Al Jawf - AJA
Al Madinah al Munawwara - AMA
Al Qasim - AQA
Ar Riyad - ARI
Ash Shargiyah - ASH
Ha'il - HAI
Jazan - JAZ
Makkah al Mukarramah - MAK
Najran - NAJ
Tabuk - TAB
 where the impact is expected to be most significant. Use only the 3-letter code.

Make sure to use more than one rule object if the policy affects multiple regions or sectors.
Make sure to think about the interaction of sectors. If the mining/oil gas sectors is affected, transportation and manufacturing sectors are also likely to be affected.
Your analysis should be based on the following:
Policy decisions have a direct impact on the economy, and the impact is expected to be significant.
these impacts are reflected in the productivity, workforce, and investment KPIs.
The impact is expected to be significant, and the analysis should reflect this.
════════════════════════════════════════════════════════════════════
OUTPUT JSON SHAPE  (return **exactly** this, nothing extra)
════════════════════════════════════════════════════════════════════
{
  "rules": [ … one or more rule objects … ],
  "reasoning": "≤ 150 words defending KPI choice, geo/sector targeting,
                direction, magnitude, and interaction of rules."
}

Return the JSON **only**.  No markdown, no prose outside `reasoning`.
"""

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["rules", "reasoning"],
            properties={
                "rules": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["region", "sector", "kpi", "years"],
                        properties={
                            "region": genai.types.Schema(type=genai.types.Type.STRING),
                            "sector": genai.types.Schema(type=genai.types.Type.STRING),
                            "kpi":    genai.types.Schema(type=genai.types.Type.STRING),
                            "years": genai.types.Schema(
                                type=genai.types.Type.OBJECT,
                                required=["2025", "2026", "2027", "2028", "2029", "2030"],
                                properties={
                                    "2025": genai.types.Schema(type=genai.types.Type.NUMBER),
                                    "2026": genai.types.Schema(type=genai.types.Type.NUMBER),
                                    "2027": genai.types.Schema(type=genai.types.Type.NUMBER),
                                    "2028": genai.types.Schema(type=genai.types.Type.NUMBER),
                                    "2029": genai.types.Schema(type=genai.types.Type.NUMBER),
                                    "2030": genai.types.Schema(type=genai.types.Type.NUMBER),
                                },
                            ),
                        },
                    ),
                ),
                "reasoning": genai.types.Schema(type=genai.types.Type.STRING),
            },
        ),
        system_instruction=[types.Part.from_text(text=system_txt)],
    )

    # Stream chunks directly to stdout
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")


if __name__ == "__main__":
    # Quick manual test
    sample = "Ban on religious tourism in Makkah from 2028."
    generate(sample)
