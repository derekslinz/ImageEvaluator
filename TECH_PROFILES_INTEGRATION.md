# Technical Profiles Integration

## Overview

The image evaluation system now uses comprehensive technical profiles that define context-specific thresholds for 10 different photography types. This replaces the previous simple CONTEXT_PROFILES with a more sophisticated numeric parameter system.

## Profile Structure

Each profile in `TECH_PROFILES` contains:

### Sharpness Parameters
- `sharpness_crit`: Below this → "critically low" warning
- `sharpness_soft`: Below this → "lower sharpness" warning  
- `sharpness_post_heavy`: Heavy penalty threshold for post-processing potential
- `sharpness_post_soft`: Soft penalty threshold; above this → bonus

### Clipping Parameters
- `clip_warn`: Percentage where text warnings start
- `clip_penalty_mid`: Medium penalty threshold
- `clip_penalty_high`: Heavy penalty threshold
- `clip_bonus_max`: Upper bound for "very low clipping" bonus

### Color Cast Parameters
- `color_cast_threshold`: Channel mean difference for detecting cast
- `color_cast_penalty`: Points deducted if non-neutral cast detected

### Other Parameters
- `post_base`: Base score for post-processing potential (default 70)
- `brightness_range`: Expected brightness tuple (min, max)
- `name`: Human-readable profile name

## 10 Photography Contexts

### 1. studio_photography
**Studio Photography**
- Most strict profile
- Critical sharpness required (<30 = critical, <60 = soft)
- Minimal clipping tolerance (warn >5%, bonus <0.5%)
- Near-neutral color expected (15.0 threshold, 8 point penalty)
- Bright images (200-245)

### 2. macro_food
**Macro/Food Photography**
- Very high sharpness demands (<35 = critical, <65 = soft)
- Strict highlight control (warn >3%)
- Moderate color cast tolerance (18.0 threshold, 6 point penalty)
- Bright but natural (180-240)

### 3. portrait_neutral
**Portrait (Neutral/Studio)**
- High sharpness for face/eyes (<25 = critical, <55 = soft)
- Moderate clipping tolerance (warn >6%)
- Sensitive to skin tone casts (12.0 threshold, 10 point penalty)
- Medium brightness (100-200)

### 4. portrait_highkey
**High-Key Portrait**
- Relaxed sharpness (<20 = critical, <50 = soft)
- Very tolerant of highlight clipping (warn >12%, penalty >40%)
- Intentional warm bias accepted (18.0 threshold, 4 point penalty)
- Very bright (180-255)

### 5. landscape
**Landscape/Nature**
- High overall sharpness (<30 = critical, <60 = soft)
- Tight clipping control (warn >4%, bonus <0.5%)
- Moderate color cast tolerance for golden/blue hour (18.0 threshold, 5 point penalty)
- Medium brightness (100-200)

### 6. street_documentary
**Street/Documentary**
- Tolerant of motion blur (<20 = critical, <45 = soft)
- High clipping tolerance for contrast (warn >10%, penalty >45%)
- Accepts varied lighting moods (22.0 threshold, 4 point penalty)
- Wide brightness range (50-200)

### 7. sports_action
**Sports/Action/Wildlife**
- Sharp subject preferred (<25 = critical, <55 = soft)
- Moderate clipping tolerance for stadium lights (warn >8%, penalty >35%)
- Accepts messy lighting (20.0 threshold, 6 point penalty)
- Wide brightness range (80-220)

### 8. concert_night
**Concert/Night/Low-Light**
- Very tolerant of blur (<15 = critical, <40 = soft)
- Huge clipping tolerance for point lights (warn >20%, penalty >70%)
- Color casts expected and accepted (30.0 threshold, 0 penalty)
- Dark images (20-120)

### 9. architecture_realestate
**Architecture/Real Estate**
- High sharpness for edges/details (<30 = critical, <65 = soft)
- Moderate clipping (small window highlights OK, warn >4%)
- Color neutrality important (15.0 threshold, 8 point penalty)
- Medium-bright (120-220)

### 10. fineart_creative
**Fine Art/Creative/Experimental**
- Sharpness not critical (<10 = critical, <25 = soft)
- Extreme clipping may be intentional (warn >25%, penalty >90%)
- Strong color casts intentional (25.0 threshold, 0 penalty)
- Full brightness range (0-255)

## Integration Points

### 1. Context Classification
```python
# Quick vision model pass to determine context
image_context = classify_image_context(image_path, encoded_image, ollama_host_url, model)
```

### 2. Technical Analysis
```python
# Pass context to technical analysis
technical_metrics = analyze_image_technical(image_path, iso_value, context=image_context)
```

### 3. Warning Generation
```python
# Generate context-aware warnings
warnings = assess_technical_metrics(technical_metrics, context=image_context)
```

### 4. Post-Processing Potential
```python
# Calculate with context-specific thresholds
post_potential = compute_post_process_potential(technical_metrics, context=image_context)
```

### 5. Enhanced Prompt
```python
# Inform AI evaluator about context and expectations
enhanced_prompt = create_enhanced_prompt(prompt, exif_data, technical_metrics)
```

## Key Functions

### `get_profile(context: str) -> Dict`
Retrieves the technical profile for a given context with fallback to `studio_photography`.

### `assess_technical_metrics(technical_metrics: Dict, context: str) -> List[str]`
Generates context-aware warnings using profile thresholds:
- Sharpness warnings based on `sharpness_crit` and `sharpness_soft`
- Clipping warnings based on `clip_warn`
- Color cast warnings only if `color_cast_penalty > 0`
- Brightness range validation

### `compute_post_process_potential(technical_metrics: Dict, context: str) -> int`
Calculates 0-100 score for post-processing improvement potential:
- Starts with `post_base` (typically 70)
- Sharpness penalties/bonuses using `sharpness_post_heavy` and `sharpness_post_soft`
- Clipping penalties using `clip_penalty_mid`, `clip_penalty_high`, and `clip_bonus_max`
- Color cast penalty using `color_cast_penalty`

## Benefits

1. **Context-Appropriate Evaluation**: Concert photos aren't penalized for highlight clipping from stage lights
2. **Numeric Precision**: Clear thresholds instead of vague "high/medium/low" priorities
3. **Transparent Scoring**: AI evaluator sees exact thresholds and penalties
4. **Easy Tuning**: Adjust individual numeric parameters without refactoring code
5. **Extensible**: Add new contexts by defining a new profile dictionary

## Migration from Old System

The previous `CONTEXT_PROFILES` used qualitative terms:
- `sharpness_priority`: 'critical', 'high', 'medium', 'low'
- `noise_tolerance`: 'critical', 'low', 'medium', 'high'

New `TECH_PROFILES` use numeric thresholds:
- `sharpness_crit`: 10-35 (varies by context)
- `clip_warn`: 3.0-25.0 (percentage)
- `color_cast_penalty`: 0-10 (points)

This provides more granular control and consistent scoring across contexts.

## Default Context

If context classification fails or returns unknown value, the system defaults to `studio_photography` (most strict profile) rather than being overly permissive.
