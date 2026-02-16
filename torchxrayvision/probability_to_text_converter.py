import torchxrayvision as xrv
import skimage.io
import torch
import torchvision
import numpy as np
from datetime import datetime

class MedicalReportGenerator:
    """Convert probability scores to human-readable medical reports"""
    
    def __init__(self):
        # Disease information database
        self.disease_info = {
            'Atelectasis': {
                'name': 'Atelectasis',
                'description': 'Partial or complete collapse of lung tissue',
                'severity': 'moderate',
                'action': 'Consult a pulmonologist for breathing exercises and treatment'
            },
            'Consolidation': {
                'name': 'Lung Consolidation',
                'description': 'Air spaces filled with fluid or tissue',
                'severity': 'high',
                'action': 'Medical evaluation needed - may require antibiotics'
            },
            'Infiltration': {
                'name': 'Pulmonary Infiltration',
                'description': 'Abnormal substances in lung tissue',
                'severity': 'moderate',
                'action': 'Further imaging and clinical correlation recommended'
            },
            'Pneumothorax': {
                'name': 'Pneumothorax',
                'description': 'Collapsed lung due to air leakage',
                'severity': 'critical',
                'action': 'URGENT: Seek emergency medical care immediately'
            },
            'Edema': {
                'name': 'Pulmonary Edema',
                'description': 'Excess fluid in the lungs',
                'severity': 'high',
                'action': 'Medical attention needed within 24 hours'
            },
            'Emphysema': {
                'name': 'Emphysema',
                'description': 'Damaged air sacs in the lungs',
                'severity': 'moderate',
                'action': 'Consult pulmonologist for management plan'
            },
            'Fibrosis': {
                'name': 'Pulmonary Fibrosis',
                'description': 'Scarring and thickening of lung tissue',
                'severity': 'moderate',
                'action': 'Specialist evaluation and monitoring required'
            },
            'Effusion': {
                'name': 'Pleural Effusion',
                'description': 'Fluid buildup around the lungs',
                'severity': 'moderate',
                'action': 'Pulmonologist consultation for possible drainage'
            },
            'Pneumonia': {
                'name': 'Pneumonia',
                'description': 'Infection causing lung inflammation',
                'severity': 'high',
                'action': 'Immediate medical attention and antibiotics needed'
            },
            'Pleural_Thickening': {
                'name': 'Pleural Thickening',
                'description': 'Thickened lining around the lungs',
                'severity': 'low',
                'action': 'Monitor and follow up with your doctor'
            },
            'Cardiomegaly': {
                'name': 'Cardiomegaly',
                'description': 'Enlarged heart',
                'severity': 'moderate',
                'action': 'Cardiology consultation recommended'
            },
            'Nodule': {
                'name': 'Pulmonary Nodule',
                'description': 'Small round growth in the lung',
                'severity': 'moderate',
                'action': 'Follow-up CT scan needed for characterization'
            },
            'Mass': {
                'name': 'Lung Mass',
                'description': 'Larger growth in lung tissue',
                'severity': 'high',
                'action': 'Immediate specialist evaluation and possible biopsy'
            },
            'Hernia': {
                'name': 'Hiatal Hernia',
                'description': 'Part of stomach pushes through diaphragm',
                'severity': 'moderate',
                'action': 'Gastroenterology consultation if symptomatic'
            },
            'Lung Lesion': {
                'name': 'Lung Lesion',
                'description': 'Abnormal area in lung tissue',
                'severity': 'moderate',
                'action': 'Further investigation with CT scan recommended'
            },
            'Fracture': {
                'name': 'Rib Fracture',
                'description': 'Broken rib bone',
                'severity': 'moderate',
                'action': 'Pain management and rest; heals naturally'
            },
            'Lung Opacity': {
                'name': 'Lung Opacity',
                'description': 'Area of increased density in lung',
                'severity': 'moderate',
                'action': 'Clinical correlation and possible follow-up imaging'
            },
            'Enlarged Cardiomediastinum': {
                'name': 'Enlarged Cardiomediastinum',
                'description': 'Widened central chest area',
                'severity': 'moderate',
                'action': 'Cardiac evaluation recommended'
            }
        }
    
    def classify_risk_level(self, probability):
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
    
    def generate_conversational_response(self, predictions):
        """Generate natural language response for chatbot"""
        
        # Sort diseases by probability
        sorted_diseases = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Get top finding
        top_disease, top_prob = sorted_diseases[0]
        risk_level, emoji = self.classify_risk_level(top_prob)
        info = self.disease_info.get(top_disease, {})
        
        # Build conversational response
        response = f"🏥 **X-Ray Analysis Complete**\n\n"
        
        # Main finding
        if top_prob >= 0.70:
            response += f"{emoji} **SIGNIFICANT FINDING DETECTED**\n\n"
            response += f"I've detected a very high probability ({top_prob*100:.1f}%) of **{info.get('name', top_disease)}**.\n\n"
            response += f"📋 What this means: {info.get('description', 'Abnormal finding detected.')}\n\n"
            response += f"⚠️ **Action Required:** {info.get('action', 'Please consult a healthcare professional.')}\n\n"
            response += "**This requires prompt medical attention. Please schedule an appointment with your doctor as soon as possible.**"
        
        elif top_prob >= 0.40:
            response += f"{emoji} **Moderate Finding**\n\n"
            response += f"The analysis shows moderate signs of **{info.get('name', top_disease)}** ({top_prob*100:.1f}% probability).\n\n"
            response += f"📋 What this means: {info.get('description', 'Abnormal finding detected.')}\n\n"
            response += f"💡 **Recommendation:** {info.get('action', 'Consult your healthcare provider.')}\n\n"
            response += "While not immediately critical, it's recommended to follow up with your doctor for proper evaluation."
        
        else:
            response += f"{emoji} **Good News!**\n\n"
            response += f"The analysis shows relatively low probability of serious conditions. The highest detection was {info.get('name', top_disease)} at {top_prob*100:.1f}%.\n\n"
            response += "However, if you're experiencing symptoms, please don't hesitate to consult a healthcare professional. This AI analysis is a screening tool and not a substitute for professional medical diagnosis."
        
        return response
    
    def generate_detailed_report(self, predictions):
        """Generate detailed medical report"""
        
        # Sort diseases by probability
        sorted_diseases = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        report = "\n" + "="*60 + "\n"
        report += "           DETAILED MEDICAL ANALYSIS REPORT\n"
        report += "="*60 + "\n\n"
        report += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Total Conditions Analyzed: {len(predictions)}\n"
        report += "-"*60 + "\n\n"
        
        # Top 5 findings
        report += "TOP 5 FINDINGS:\n"
        report += "-"*60 + "\n\n"
        
        for i, (disease, prob) in enumerate(sorted_diseases[:5], 1):
            risk_level, emoji = self.classify_risk_level(prob)
            info = self.disease_info.get(disease, {})
            
            report += f"{i}. {emoji} {info.get('name', disease)}\n"
            report += f"   Probability: {prob*100:.1f}% | Risk Level: {risk_level}\n"
            report += f"   Description: {info.get('description', 'N/A')}\n"
            report += f"   Recommendation: {info.get('action', 'N/A')}\n\n"
        
        # All findings table
        report += "\n" + "-"*60 + "\n"
        report += "COMPLETE ANALYSIS (All Conditions):\n"
        report += "-"*60 + "\n\n"
        report += f"{'Condition':<30} {'Probability':<12} {'Risk Level':<15}\n"
        report += "-"*60 + "\n"
        
        for disease, prob in sorted_diseases:
            risk_level, emoji = self.classify_risk_level(prob)
            disease_name = self.disease_info.get(disease, {}).get('name', disease)
            report += f"{disease_name:<30} {prob*100:>6.1f}%      {emoji} {risk_level}\n"
        
        report += "\n" + "="*60 + "\n"
        report += "DISCLAIMER: This analysis is for educational purposes only.\n"
        report += "Always consult qualified healthcare professionals for medical\n"
        report += "diagnosis and treatment decisions.\n"
        report += "="*60 + "\n"
        
        return report
    
    def generate_json_response(self, predictions):
        """Generate structured JSON for API responses"""
        
        sorted_diseases = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_disease, top_prob = sorted_diseases[0]
        
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "top_finding": self.disease_info.get(top_disease, {}).get('name', top_disease),
                "probability": float(top_prob),
                "percentage": round(float(top_prob) * 100, 1),
                "risk_level": self.classify_risk_level(top_prob)[0]
            },
            "findings": []
        }
        
        for disease, prob in sorted_diseases:
            risk_level, emoji = self.classify_risk_level(prob)
            info = self.disease_info.get(disease, {})
            
            response["findings"].append({
                "disease": disease,
                "disease_name": info.get('name', disease),
                "probability": float(prob),
                "percentage": round(float(prob) * 100, 1),
                "risk_level": risk_level,
                "emoji": emoji,
                "description": info.get('description', ''),
                "action": info.get('action', '')
            })
        
        return response


def analyze_xray(image_path):
    """Main function to analyze X-ray and generate reports"""
    
    print("🔬 Loading image and model...\n")
    
    # Load and preprocess image
    img = skimage.io.imread(image_path)
    img = xrv.datasets.normalize(img, 255)
    
    # Handle color images
    if len(img.shape) == 3:
        img = img.mean(2)[None, ...]
    else:
        img = img[None, ...]
    
    # Transform
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img = transform(img)
    img = torch.tensor(img).unsqueeze(0)
    
    # Load model and predict
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    out = model(img)[0].detach().numpy()
    
    # Create predictions dictionary
    predictions = dict(zip(model.pathologies, out))
    
    # Generate reports
    generator = MedicalReportGenerator()
    
    print("="*60)
    print("RAW OUTPUT (What the model returns):")
    print("="*60)
    for disease, prob in predictions.items():
        print(f"{disease}: {prob:.6f}")
    
    print("\n\n")
    print("="*60)
    print("CONVERSATIONAL RESPONSE (For Chatbot):")
    print("="*60)
    conversational = generator.generate_conversational_response(predictions)
    print(conversational)
    
    print("\n\n")
    detailed_report = generator.generate_detailed_report(predictions)
    print(detailed_report)
    
    # JSON format
    json_response = generator.generate_json_response(predictions)
    
    return {
        'raw_predictions': predictions,
        'conversational_response': conversational,
        'detailed_report': detailed_report,
        'json_response': json_response
    }


# Example usage
if __name__ == "__main__":
    # Analyze the X-ray
    results = analyze_xray("images/a.jpeg")
    
    # You can access different formats:
    print("\n\n" + "="*60)
    print("FOR CHATBOT - Use this response:")
    print("="*60)
    print(results['conversational_response'])
    
    # Save JSON for API
    import json
    with open('analysis_result.json', 'w') as f:
        json.dump(results['json_response'], f, indent=2)
    
    print("\n✅ JSON response saved to 'analysis_result.json'")
