# 🚨 DisasterSOS AI

### Social Media Emotion & Emergency Intelligence for Disaster Response

DisasterSOS AI is an AI-powered decision-support system designed to analyze disaster-related social media posts and identify potentially urgent situations.

During floods, earthquakes, and other disasters, emergency agencies may receive a large number of social media messages. Manually reviewing every message can make it difficult to identify urgent requests quickly.

This project analyzes text reports and extracts useful emergency intelligence such as emotion, sentiment, emergency intent, urgency, help category, location, medical emergency indicators, and an overall priority score.

---

## 🎯 Problem Statement

During a disaster, people often use social media to request help, report injuries, describe unsafe conditions, or provide information about affected areas.

Emergency responders may receive hundreds or thousands of messages, making manual prioritization difficult.

DisasterSOS AI aims to help organize these reports and highlight potentially critical messages for human verification.

---

## 💡 Proposed Solution

The system processes disaster-related text and performs multiple NLP tasks:

- Emotion detection
- Sentiment analysis
- Emergency/help intent detection
- Urgency classification
- Help-category identification
- Location extraction
- Number of affected people detection
- Medical emergency detection
- Priority scoring
- Emergency dashboard visualization

The system is designed as **decision support for emergency responders**, not as an autonomous rescue decision-maker.

---

## 🔄 System Workflow

```text
Social Media / Disaster Report
             ↓
      Text Preprocessing
             ↓
       NLP Analysis
             ↓
 ┌───────────┼────────────┐
 ↓           ↓            ↓
Emotion   Sentiment    Emergency
Detection  Analysis      Intent
             ↓
       Urgency Analysis
             ↓
      Help Category
             ↓
   Location / Entity Extraction
             ↓
       Priority Scoring
             ↓
     Emergency Dashboard
