
# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate(prompt):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["region", "sector", "2025_impact", "2026_impact", "2027_impact", "2028_impact", "2029_impact", "2030_impact", "KPI"],
            properties={
                "region": genai.types.Schema(type=genai.types.Type.STRING),
                "sector": genai.types.Schema(type=genai.types.Type.STRING),
                "2025_impact": genai.types.Schema(type=genai.types.Type.NUMBER),
                "2026_impact": genai.types.Schema(type=genai.types.Type.NUMBER),
                "2027_impact": genai.types.Schema(type=genai.types.Type.NUMBER),
                "2028_impact": genai.types.Schema(type=genai.types.Type.NUMBER),
                "2029_impact": genai.types.Schema(type=genai.types.Type.NUMBER),
                "2030_impact": genai.types.Schema(type=genai.types.Type.NUMBER),
                "KPI": genai.types.Schema(type=genai.types.Type.STRING),
                "reasoning": genai.types.Schema(type=genai.types.Type.STRING),
            },
        ),
        system_instruction=[
            types.Part.from_text(text="""You are an expert macroeconomics analyst specializing in Saudi Arabia. Your task is to evaluate the impact of a given policy action on the key inputs of the Cobb-Douglas production function, which are population (labor), investments (capital), and productivity (total factor productivity).

For each policy action I provide, you must:

1.  **Identify the most directly impacted Cobb-Douglas input (KPI):** Choose ONE from \"population\", \"investments\", or \"productivity\" that the policy is most likely to influence.
2.  **Determine the affected region:** Select ONE region from 'Asir - ASI
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
3.  **Determine the affected sector:** Select ONE sector from the ISIC classification using its letter where the impact is expected to be most significant (e.g., A for agriculture).
4.  **Estimate the percentage impact for the years 2025 through 2030:** Based on your macroeconomic expertise and knowledge of Saudi Arabia, provide a percentage change (positive or negative) for the chosen KPI in the specified region and sector for each year. Express this as a decimal (e.g., a 2% increase is 0.02, a 5% decrease is -0.05). If you believe there will be no significant impact on any of the specified KPIs, regions, or sectors, return 0 for the impact values.
5.  **Justify your reasoning:** Briefly explain why you chose the specific KPI, region, sector, and the estimated impact percentages.

You MUST return your analysis in the following JSON format:

```json
{
  "2025_impact": [value],
  "2026_impact": [value],
  "2027_impact": [value],
  "2028_impact": [value],
  "2029_impact": [value],
  "2030_impact": [value],
  "KPI": "[population|investments|productivity]",
  "region": "[3-letter region code]",
  "sector": "[A|B|C...]",
  "reasoning": "[brief explanation]"
}
```

Consider only the information provided in the policy action. Do not make assumptions beyond the scope of the described policy. Remember that extreme events, like a complete ban on a sector, should result in a -1.00 impact if productivity is chosen as the KPI.
"""),
        ],
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")
