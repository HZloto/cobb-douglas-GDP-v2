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
After you have decided on your impact, you need to add to the rules the regional spillover effect.
Use the following rules to determine the impact on the regions:
region_from,region_to,spillover_pct
ABA,ASI,0.25
ABA,MAK,0.25
AHU,AJA,0.25
AHU,TAB,0.25
AJA,AHU,0.25
AJA,HAI,0.25
AJA,TAB,0.25
AMA,AQA,0.25
AMA,ARI,0.25
AMA,MAK,0.25
AMA,TAB,0.25
AQA,AMA,0.25
AQA,ARI,0.25
AQA,HAI,0.25
ARI,AMA,0.25
ARI,AQA,0.25
ARI,ASH,0.25
ARI,HAI,0.25
ARI,NAJ,0.25
ASH,ARI,0.25
ASH,ASI,0.25
ASH,NAJ,0.25
ASI,ABA,0.25
ASI,ASH,0.25
ASI,JAZ,0.25
ASI,NAJ,0.25
HAI,AJA,0.25
HAI,AQA,0.25
HAI,ARI,0.25
JAZ,ASI,0.25
JAZ,NAJ,0.25
MAK,ABA,0.25
MAK,AMA,0.25
NAJ,ARI,0.25
NAJ,ASH,0.25
NAJ,ASI,0.25
NAJ,JAZ,0.25
TAB,AHU,0.25
TAB,AJA,0.25
TAB,AMA,0.25
Consider all the other pairings to have a 0.0 spillover effect.
For example if your policy is a ban on religious tourism in Makkah from 2028, you would have the following rules:
Workforce would be reduced by a large amount in Makkah, and a small amount in Asir, Al Bahah, Al Hudud ash Shamaliyah, Al Jawf, Al Madinah al Munawwara, Al Qasim, Ar Riyad, Ash Shargiyah, Ha'il, Jazan, Najran and Tabuk.
Investments would be reduced to 0 in makkah and spilled over refgions. 
Some spillovers may be more complex, for example if foreigners are banned for working in finance in Riyadh, we may expect to see more finance workers in other regions with financial hubs
Also take into account the interaction of sectors. If the mining/oil gas sectors is affected, transportation and manufacturing sectors are also likely to be affected.
The spillover effect is expected to be significant, and the analysis should reflect this.
Here is the input/output spillover table for reference but you may use larger numbers if you think it is appropriate.
,,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T
,,"Agriculture, forestry and fishing",Extraction of crude petroleum and natural gas Mining and quarrying,Manufacturing,"Electricity, gas, and Water Supply; Waste Management Activities","Water supply; sewerage, waste management and remediation activities",Construction,Wholesale and retail trade; repair of motor vehicles and motorcycles,Transportation and Storage,Accommodation and food service activities,Information and communication,Financial and insurance activities,Real estate activities,"Professional, scientific and technical activities",Administrative and support service activities,Public administration and defence; compulsory social security,Education,Human health and social work activities,"Arts,Recreation and Other Service Activities",Other personal service activities and Activities of households as employers of domestic personnel,Activities of households
A,"Agriculture, forestry and fishing",0.006687565,0,0.001131964,0,0,0,0,0.000289318,0.001851619,0,0,0,0,0,0,0,0.000210427,0,0,
B,Extraction of crude petroleum and natural gas Mining and quarrying,0,0.003037535,0.009046397,0,0,0.000186954,1.88798E-05,9.15169E-05,0,0,0,0,0,0,0,0,0,0,0,
C,Manufacturing,0.057279557,0.005326989,0.06989611,0.056207118,0.056207118,0.056726946,0.003480758,0.045822713,0.036075922,0.025606966,0.00140319,0.001526069,0.017382574,0.010112619,0.012584879,0.016467149,0.025631582,0.023995042,0,
D,"Electricity, gas, and Water Supply; Waste Management Activities",0.011135893,0.010559284,0.025970085,0.012807466,0.012807466,0.006992615,0.029131831,0.006995975,0.043914804,0.00824223,0.0057091,0.023450073,0.005558289,0.012405512,0.061786942,0.014842505,0.021716759,0.043114799,0,
E,"Water supply; sewerage, waste management and remediation activities",0.011135893,0.010559284,0.025970085,0.012807466,0.012807466,0.006992615,0.029131831,0.006995975,0.043914804,0.00824223,0.0057091,0.023450073,0.005558289,0.012405512,0.061786942,0.014842505,0.021716759,0.043114799,0,
F,Construction,0,0.007080389,0.006670912,0.008961125,0.008961125,0.144125398,0.012872272,0.009396831,0.03981104,0.005090148,0.005905813,0.067124566,0.028519726,0.017221199,0.041366168,0.025735482,0.026310224,0.061115778,0,
G,Wholesale and retail trade; repair of motor vehicles and motorcycles,0.063088499,0.018354745,0.086609826,0.002304401,0.002304401,0.080266803,0.056858333,0.079527772,0.074358343,0.043762404,0.00066056,0.00090604,0.063356717,0.051281467,0.01165137,0.021142404,0.025097782,0.048062421,0,
H,Transportation and Storage,0.010897429,0.00088036,0.025523353,0.007785048,0.007785048,0.007335563,0.081939095,0.132640019,0.004812185,0.061996962,0.002582908,0.000160567,0.007497226,0.016037348,0.024695301,0.055217513,0.018392623,0.006657193,0,
I,Accommodation and food service activities,0,0,0,0.000517334,0.000517334,0,0,0.00892405,0,0,0,0,0,0,0.061820036,0.001179131,0.002204015,0.000424638,0,
J,Information and communication,0.00080401,0.002211336,0.001373996,0.001806776,0.001806776,0.000912963,0.006185005,0.01503069,0.008160237,0.085627689,0.058631613,0.006014533,0.094599758,0.004168946,0.032804544,0.00852005,0.027771068,0.031485806,0,
K,Financial and insurance activities,0.061841205,0.072022336,0.016574856,0.024876958,0.024876958,0.029542952,0.051253015,0.011969039,0.098647956,0.022558734,0.132355242,0.122959146,0.075848323,0.057161573,0.106833169,0.072327736,0.070858248,0.124930184,0,
L,Real estate activities,0,0,0.004038308,0.002557384,0.002557384,0.003764087,0.012758499,0.005949671,0.009617983,0.014746598,0.004247601,0.029041566,0.011819074,0.010719385,0.005322406,0.013596455,0.016089471,0.014944889,0,
M,"Professional, scientific and technical activities",0.013758867,0.014093539,0.00446695,0.000170904,0.000170904,0.007162731,0.040755683,0.028533285,0.029791016,0.068169927,0.014868199,0.043576405,0.066501097,0.063527862,0.009704475,0.029566622,0.032662943,0.0257951,0,
N,Administrative and support service activities,0,0,0.008781998,0.001393451,0.001393451,0.016263046,0.020500146,0.078927579,0.016148641,0.037603339,0.008185895,0.007055424,0.018917909,0.077793623,0.005304119,0.004484817,0.021475563,0.02309476,0,
O,Public administration and defence; compulsory social security,0,0.001155912,0.002413834,0.008558809,0.008558809,0.007800544,0.009175848,0.002603355,0.002204,0.013941373,0.000875093,0.00340697,0.020808294,0.038217168,0.02008783,0.031989181,0.025466306,0.003356171,0,
P,Education,0,0.00032084,0.000536912,0.002336855,0.002336855,0.000329536,0,9.81628E-05,0,7.83921E-05,0,0.002040466,0.004035586,0.0013991,0.003501811,0.001952239,5.84958E-06,0.000583897,0,
Q,Human health and social work activities,0,0,0.00026037,8.16696E-06,8.16696E-06,0.000309662,0.000402792,0.000112154,0.00054418,0.001036733,0,2.58859E-05,0.000585884,0.000315119,0,0.000151606,0.000533078,0.001180357,0,
R,"Arts,Recreation and Other Service Activities",0,0.000176565,0.000223315,0.000556901,0.000556901,0.000186527,0.001312755,0.000128669,0.006336062,0.000139701,0.003321913,0.001481273,0.001715114,0.020218069,0.000803955,0.000839656,0.010459012,0.011420547,0,
S,Other personal service activities and Activities of households as employers of domestic personnel,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
T,Activities of households,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
Make sure to always consider the sectors and regional implications in the context of the policy. 
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
