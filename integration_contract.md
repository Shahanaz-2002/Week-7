# Integration Contract – Clinical Insight Engine (Skin Disease Domain)

## Overview

This document defines the data exchange format between the **Semantic Retrieval Engine** and the **Insight Engine** for a skin disease analysis system. It ensures consistency, modularity, and seamless integration across components.

---

## 1. Input to Insight Engine

The Insight Engine receives the output from the Retrieval Engine in the form of a list of top-K similar skin disease cases.

### Data Structure

* **Type:** List of Dictionaries
* **Description:** Each dictionary represents one retrieved dermatological case.

---

### Fields Description

| Field Name | Type        | Description                                                                |
| ---------- | ----------- | -------------------------------------------------------------------------- |
| case_id    | string      | Unique identifier for the case                                             |
| similarity | float       | Similarity score between input case and retrieved case (range: 0–1)        |
| features   | dict / list | Extracted features (image features, texture, lesion characteristics, etc.) |
| diagnosis  | string      | Skin disease diagnosis                                                     |
| outcome    | string      | Treatment outcome or condition status                                      |

---

### Example Input (Skin Disease Cases)

```python
[
  {
    "case_id": "SD001",
    "similarity": 0.92,
    "features": [0.34, 0.67, 0.21],
    "diagnosis": "Melanoma",
    "outcome": "Biopsy Confirmed"
  },
  {
    "case_id": "SD002",
    "similarity": 0.88,
    "features": [0.30, 0.60, 0.25],
    "diagnosis": "Melanoma",
    "outcome": "Under Treatment"
  },
  {
    "case_id": "SD003",
    "similarity": 0.81,
    "features": [0.28, 0.55, 0.20],
    "diagnosis": "Benign Nevus",
    "outcome": "No Treatment Required"
  }
]
```

---

## 2. Processing in Insight Engine

The Insight Engine performs the following steps:

1. Extract diagnoses from retrieved cases
2. Apply majority voting to determine predicted skin condition
3. Compute confidence score using similarity values
4. Generate explanation based on similar dermatological cases

---

## 3. Output from Insight Engine

The Insight Engine returns a structured response containing the predicted skin disease, confidence score, and explanation.

---

### Data Structure

* **Type:** Dictionary

---

### Fields Description

| Field Name  | Type   | Description                        |
| ----------- | ------ | ---------------------------------- |
| prediction  | string | Predicted skin disease             |
| confidence  | float  | Confidence score (0–1)             |
| explanation | string | Explanation based on similar cases |

---

### Example Output

```python
{
  "prediction": "Melanoma",
  "confidence": 0.87,
  "explanation": "Prediction is based on 3 similar skin lesion cases, where 2 cases are diagnosed as Melanoma with high similarity (>0.85)."
}
```

---

## 4. Assumptions

* Retrieval Engine returns top-K similar dermatological cases (default K = 3)
* Similarity scores are normalized between 0 and 1
* Each case includes valid diagnosis and outcome information

---

## 5. Error Handling

If no similar cases are found:

```python
{
  "prediction": null,
  "confidence": 0.0,
  "explanation": "No similar skin disease cases found."
}
```

---

## 6. Notes

* Similarity computation is handled only by the Retrieval Engine
* Insight Engine does not perform similarity calculations
* All modules must strictly follow this format for compatibility
* Designed specifically for dermatological (skin disease) analysis systems

---
