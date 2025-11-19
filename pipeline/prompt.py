import numpy as np

def create_prompt(train_df: str, train_summary: str, target: str, y: np.array, test_sample: str) -> str:
    
    target_info = {
        'pH': {
            'description': 'soil pH (acidity/alkalinity)',
            'constraint': 'typically ranges from 3.5 to 8.5 in forest soils',
            'interpretation': 'Lower values = more acidic. Surface soils often more acidic than subsurface due to organic matter decomposition.'
        },
        'SOM': {
            'description': 'soil organic matter content (%)',
            'constraint': 'non-negative floating point number (0-100%)',
            'interpretation': 'Higher at surface, decreases with depth. Strongly related to soc (soil organic carbon).'
        }
    }

    if target in target_info:
        target_desc = target_info[target]['description']
        target_constraint = target_info[target]['constraint']
        target_interp = target_info[target]['interpretation']

    prompt = f"""
    You are an expert soil scientist specializing in forest soils. You are given a training dataset and a test sample.
    Your task is to predict the {target} of the test sample.

    TARGET VARIABLE:
    - Variable: {target}
    - Description: {target_desc}
    - Constraint: {target_constraint}
    - Interpretation: {target_interp}
    - Training data range: {np.min(y):.2f} to {np.max(y):.2f}
    (mean: {np.mean(y):.2f}, std: {np.std(y):.2f})

    CONTEXT:
    - Location: Lamont Forest Campus, New York (mixed temperate forest)
    - Sampling design: 11×11 grid (121 points total), 10 meters between points
    - Coordinates: Format is (x,y) where values are in 10-meter intervals
    - Depth classes:
    * Depth_mid = 5 represents 0-10cm depth (surface soil)
    * Depth_mid = 15 represents 10-20cm depth (subsurface soil)
    - Sampling period: Summer 2025
    - Train/test split: Both depths from the same location are always in the same set (train OR test, never split)
    - Training data statistics (from df.describe()):
    * {target} range: {np.min(y):.2f} to {np.max(y):.2f}
    * {target} mean: {np.mean(y):.2f}, std: {np.std(y):.2f}

    INSTRUCTIONS:

    Step 1: Identify test sample context
    - Extract coordinates from test sample ID
    - Note: The test location will NOT appear in training data (both depths excluded to prevent data leakage)
    - Identify the 5 nearest spatial neighbors by Euclidean distance in coordinate space

    Step 2: Understand feature relationships
    - Review which features show strongest correlation with {target} in the summary statistics
    - Consider soil properties (texture, organic carbon, bulk density) as primary predictors
    - Consider vegetation indices and topography as secondary predictors
    - Note: Surface soils (5cm) are typically more acidic than subsurface (15cm) due to organic matter

    Step 3: Develop predictions using three approaches
    1. POINT-BASED: Use test sample's feature values and global feature-target relationships from training data
    2. SPATIAL: Weight predictions from 5 nearest neighbors (inverse distance weighting recommended)
    3. DEPTH-PATTERN: Analyze typical depth gradients in training data (compare Depth_mid=5 vs 15 across all locations)
    - If predicting for Depth_mid=15, consider typical change from 5→15cm
    - If predicting for Depth_mid=5, consider typical surface soil patterns

    Step 4: Synthesize final prediction
    - Integrate insights from all three approaches
    - Weight approaches based on data availability and relevance
    - Provide your single best estimate with confidence level

    OUTPUT FORMAT:

    Analysis:
    <Describe test sample context, nearest neighbors found, and key features observed>

    Reasoning:
    <Explain which approach(es) you weighted most heavily and why>

    Final Prediction: <single best estimate>
    Confidence: <High/Medium/Low - with brief justification>

    Supporting Estimates:
    1. Point-based approach: <estimate1> - <reasoning>
    2. Spatial approach (5-NN): <estimate2> - <reasoning>
    3. Depth-pattern approach: <estimate3> - <reasoning based on general depth trends>

    Feature Importance Scores (0-100):
    <feature>: <score>
    <feature>: <score>
    ...

    Key Features Explanation (scores > 50 only):
    <feature> (<score>): <why this feature was critical for this prediction>

    TRAINING DATASET:
    {train_df}

    TRAINING SUMMARY STATISTICS:
    {train_summary}

    TEST SAMPLE:
    {test_sample}

    IMPORTANT REMINDERS:
    - The test location does NOT appear in training data (to prevent data leakage)
    - Use spatial interpolation from nearest neighbors as primary approach
    - Apply general depth patterns from training data (not location-specific depth info)
    - Ensure prediction respects the constraint: {target_constraint}
    - Never predict outside the training data range without strong justification
    - If highly uncertain, default to training mean but explain uncertainty
    - Consider soil-forming factors: topography, vegetation, parent material, depth
    """

    return prompt
