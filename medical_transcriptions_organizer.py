"""
You have been provided with an anonymized dataset of medical transcriptions organized by specialty, transcriptions.csv.

Use the OpenAI API to extract "age", "medical_specialty", and a new data field to store the recommended treatment extracted from each transcription.

Match each recommended treatment with the corresponding International Classification of Diseases (ICD) code, and save your answers in a pandas DataFrame named df_structured.


"""


# Import the necessary libraries
import pandas as pd
from openai import OpenAI
import json
import re

# Load the data
df = pd.read_csv("data/transcriptions.csv")
df.head()
print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

# Uutput: Loaded 5 rows. Columns: ['medical_specialty', 'transcription']

# Initialize the OpenAI client
client = OpenAI()


# Helper: call OpenAI to extract structured fields
def extract_fields(transcription: str, medical_specialty: str) -> dict:
    """
    Returns a dict with keys:
      age               – integer or None
      recommended_treatment – plain-text string
      icd_code          – e.g. "Z00.00"
      icd_description   – human-readable label for the code
    """
    prompt = f"""You are a clinical data-extraction assistant.
Given the medical transcription below (specialty: {medical_specialty}),
extract the following and return ONLY valid JSON with these exact keys:
  "age"                   : integer (patient age in years) or null if not mentioned
  "recommended_treatment" : string  (the primary recommended treatment/procedure)
  "icd_code"              : string  (the most appropriate ICD-10-CM code for the
                                      recommended treatment or primary diagnosis)
  "icd_description"       : string  (short official description for that ICD code)

Transcription:
\"\"\"
{transcription[:3000]}
\"\"\"

Return only JSON, no markdown fences, no extra text."""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )
    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    return json.loads(raw)


# Process each row
records = []
for idx, row in df.iterrows():
    try:
        result = extract_fields(
            transcription=str(row["transcription"]),
            medical_specialty=str(row["medical_specialty"]),
        )
        records.append({
            "age":                   result.get("age"),
            "medical_specialty":     row["medical_specialty"],
            "recommended_treatment": result.get("recommended_treatment"),
            "icd_code":              result.get("icd_code"),
            "icd_description":       result.get("icd_description"),
        })
    except Exception as e:
        print(f"Row {idx} failed: {e}")
        records.append({
            "age":                   None,
            "medical_specialty":     row.get("medical_specialty"),
            "recommended_treatment": None,
            "icd_code":              None,
            "icd_description":       None,
        })

# Build the structured DataFrame
df_structured = pd.DataFrame(records, columns=[
    "age",
    "medical_specialty",
    "recommended_treatment",
    "icd_code",
    "icd_description",
])

print("\ndf_structured sample:")
print(df_structured.head(10).to_string(index=False))
print(f"\nShape: {df_structured.shape}")
print(f"Null counts:\n{df_structured.isnull().sum()}")

"""
output:

df_structured sample:
 age          medical_specialty                                                                                                                                                                                                         recommended_treatment icd_code                                                                       icd_description
  23       Allergy / Immunology                                                                                                                                                             Samples of Nasonex two sprays in each nostril given for three weeks    J30.1                                                       Allergic rhinitis due to pollen
  41                 Orthopedic                                                                                                                                                                                                              Operative fixation S86.011A                         Traumatic rupture of right Achilles tendon, initial encounter
  30                 Bariatrics                                                                                                                                                                     Laparoscopic antecolic antegastric Roux-en-Y gastric bypass  0DTY0Z0 Bypass Stomach to Jejunum with Synthetic Substitute, Percutaneous Endoscopic Approach
  50 Cardiovascular / Pulmonary Neck exploration; tracheostomy; urgent flexible bronchoscopy via tracheostomy site; removal of foreign body, tracheal metallic stent material; dilation distal trachea; placement of #8 Shiley single cannula tracheostomy tube    J39.8                                   Other specified diseases of upper respiratory tract
  66                    Urology                                                                                                                                                                                                    Benign prostatic hypertrophy    N40.1                                                          Benign prostatic hyperplasia

Shape: (5, 5)
Null counts:
age                      0
medical_specialty        0
recommended_treatment    0
icd_code                 0
icd_description          0
dtype: int64
"""
