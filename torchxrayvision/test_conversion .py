"""
Test file to see how raw probabilities convert to chatbot text
"""
import numpy as np

# Your exact output from test.py
raw_predictions = {
    'Atelectasis': np.float32(0.54051274), 
    'Consolidation': np.float32(0.17454077), 
    'Infiltration': np.float32(0.51858664), 
    'Pneumothorax': np.float32(0.10749136), 
    'Edema': np.float32(0.0010575246), 
    'Emphysema': np.float32(0.5004622), 
    'Fibrosis': np.float32(0.5155582), 
    'Effusion': np.float32(0.12558937), 
    'Pneumonia': np.float32(0.043546), 
    'Pleural_Thickening': np.float32(0.5325583), 
    'Cardiomegaly': np.float32(0.46688202), 
    'Nodule': np.float32(0.5094887), 
    'Mass': np.float32(0.56457794), 
    'Hernia': np.float32(0.99355507), 
    'Lung Lesion': np.float32(0.0022418485), 
    'Fracture': np.float32(0.5131295), 
    'Lung Opacity': np.float32(0.36164907), 
    'Enlarged Cardiomediastinum': np.float32(0.10844509)
}

def classify_risk(probability):
    """Convert probability to risk level"""
    if probability >= 0.75:
        return "Very High Risk", "🔴"
    elif probability >= 0.50:
        return "High Risk", "🟠"
    elif probability >= 0.30:
        return "Moderate Risk", "🟡"
    elif probability >= 0.15:
        return "Low Risk", "🟢"
    else:
        return "Very Low Risk", "⚪"

print("="*70)
print("RAW MODEL OUTPUT (What you currently have)")
print("="*70)
for disease, prob in raw_predictions.items():
    print(f"{disease:30} {prob:.6f}")

print("\n\n")
print("="*70)
print("FORMATTED OUTPUT (With percentages and risk levels)")
print("="*70)
sorted_diseases = sorted(raw_predictions.items(), key=lambda x: x[1], reverse=True)

for disease, prob in sorted_diseases:
    risk_level, emoji = classify_risk(prob)
    print(f"{emoji} {disease:30} {prob*100:6.1f}%  |  {risk_level}")

print("\n\n")
print("="*70)
print("CHATBOT CONVERSATIONAL RESPONSE (What user will see)")
print("="*70)

# Get top finding
top_disease, top_prob = sorted_diseases[0]
risk_level, emoji = classify_risk(top_prob)

chatbot_response = f"""
{emoji} **SIGNIFICANT FINDING DETECTED**

I've detected a very high probability ({top_prob*100:.1f}%) of **{top_disease}** (Hiatal Hernia).

📋 **What this means:** Part of the stomach pushes through the diaphragm into the chest cavity.

⚠️ **Action Required:** Gastroenterology consultation if you're experiencing symptoms like:
   • Heartburn
   • Difficulty swallowing
   • Chest pain
   • Acid reflux

💡 **Next Steps:**
   1. Schedule an appointment with a gastroenterologist
   2. Keep track of any digestive symptoms
   3. Avoid large meals before bedtime
   4. Elevate your head while sleeping

**Important:** While this requires medical follow-up, hiatal hernias are common 
and often manageable with lifestyle changes and medication. Please consult your 
doctor for proper evaluation and treatment plan.

---

**Other Notable Findings:**
• {sorted_diseases[1][0]}: {sorted_diseases[1][1]*100:.1f}%
• {sorted_diseases[2][0]}: {sorted_diseases[2][1]*100:.1f}%
• {sorted_diseases[3][0]}: {sorted_diseases[3][1]*100:.1f}%
"""

print(chatbot_response)

print("\n\n")
print("="*70)
print("JSON FORMAT (For API Response)")
print("="*70)

import json

json_response = {
    "success": True,
    "summary": {
        "top_finding": top_disease,
        "probability": float(top_prob),
        "percentage": round(float(top_prob) * 100, 1),
        "risk_level": risk_level
    },
    "all_findings": [
        {
            "disease": disease,
            "probability": float(prob),
            "percentage": round(float(prob) * 100, 1),
            "risk_level": classify_risk(prob)[0],
            "emoji": classify_risk(prob)[1]
        }
        for disease, prob in sorted_diseases
    ]
}

print(json.dumps(json_response, indent=2))

print("\n\n")
print("="*70)
print("SIMPLE SUMMARY (For quick chatbot reply)")
print("="*70)

simple_summary = f"""
Your X-ray analysis is complete! Here's what I found:

🔍 **Primary Finding:** {top_disease} ({top_prob*100:.1f}% confidence)

📊 **Top 3 Conditions Detected:**
1. {emoji} {sorted_diseases[0][0]} - {sorted_diseases[0][1]*100:.1f}%
2. {classify_risk(sorted_diseases[1][1])[1]} {sorted_diseases[1][0]} - {sorted_diseases[1][1]*100:.1f}%
3. {classify_risk(sorted_diseases[2][1])[1]} {sorted_diseases[2][0]} - {sorted_diseases[2][1]*100:.1f}%

💬 Would you like to know more about any of these findings? 
I can explain what they mean and what steps you should take.

Type "explain {top_disease}" or "what should I do?" to learn more.
"""

print(simple_summary)
