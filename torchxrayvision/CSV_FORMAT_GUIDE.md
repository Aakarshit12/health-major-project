# CSV File Format for Model Evaluation

## Required Format

Your CSV file must have the following structure:

### Required Column
- **`filename`**: The name of the image file (must match files in your `images/` directory)

### Label Columns
Each pathology/condition should be a separate column with binary values:
- **`1`** = Pathology is present (positive)
- **`0`** = Pathology is absent (negative)  
- **`NaN`** or empty = Unknown/not labeled

## Standard Pathology Columns

The model supports these 18 pathologies (you can include all or a subset):

1. `Atelectasis`
2. `Consolidation`
3. `Infiltration`
4. `Pneumothorax`
5. `Edema`
6. `Emphysema`
7. `Fibrosis`
8. `Effusion`
9. `Pneumonia`
10. `Pleural_Thickening`
11. `Cardiomegaly`
12. `Nodule`
13. `Mass`
14. `Hernia`
15. `Lung Lesion`
16. `Fracture`
17. `Lung Opacity`
18. `Enlarged Cardiomediastinum`

## Example CSV

```csv
filename,Atelectasis,Consolidation,Infiltration,Pneumothorax,Edema,Emphysema,Fibrosis,Effusion,Pneumonia,Pleural_Thickening,Cardiomegaly,Nodule,Mass,Hernia
img_001.png,1,0,0,0,0,0,0,0,0,0,0,0,0,0
img_002.png,0,1,1,0,0,0,0,0,0,0,0,0,0,0
img_003.png,0,0,0,0,0,0,0,1,0,0,1,0,0,0
img_004.png,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

## Notes

- **Filename must match exactly** (including extension: .jpg, .jpeg, .png, etc.)
- You **don't need all pathology columns** - include only the ones you have labels for
- If a pathology column is missing, it will be treated as NaN (unknown)
- You can have **multiple pathologies** per image (multi-label classification)
- **Empty cells** or **NaN** values mean "unknown/not labeled" for that pathology

## Minimal Example

If you only have labels for a few pathologies:

```csv
filename,Pneumonia,COVID-19,Cardiomegaly
img_001.png,1,0,0
img_002.png,0,1,1
img_003.png,0,0,0
```

**Note:** If you use custom pathology names (like "COVID-19"), the model will still work, but it will only evaluate against pathologies that match the model's built-in pathology list.


