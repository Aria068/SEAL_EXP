"""
Data generation script for SEAL training data.

This script generates training data by:
1. Taking original text samples with ground truth personal attributes
2. Using an LLM to generate multiple anonymized versions
3. Using an LLM to evaluate privacy leakage (inference attacks)
4. Using an LLM to evaluate utility (readability, meaning, hallucinations)
5. Saving the results in JSONL format for training
"""

import argparse
import json
import os
import random
import time
from typing import Dict, List, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    # Fallback to simple iteration
    def tqdm(iterable, **kwargs):
        return iterable


# Attribute types for privacy evaluation
ATTRIBUTE_TYPES = [
    "income",
    "education",
    "pobp",
    "location",
    "married",
    "gender",
    "age",
    "occupation",
]

TYPE_TO_STR = {
    "income": "yearly income",
    "education": "level of education",
    "pobp": "place of birth",
    "location": "current place of living",
    "married": "relationship status",
    "gender": "gender",
    "age": "age",
    "occupation": "occupation",
}


def load_synthpai_data(input_file: str) -> List[Dict[str, Any]]:
    """Load SynthPAI dataset from JSONL file.

    Expected format:
    {
        "id": "unique_id",
        "author": author_id,
        "text": "original comment text",
        "truth": {
            "occupation": "nurse",
            "age": 35,
            "married": "in relation",
            ...
        }
    }
    """
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_anonymization_prompt(
    text: str,
    prompt_level: int = 3,
    reasoning: bool = False,
    inferred_attributes: Optional[List[Dict]] = None,
) -> List[Dict[str, str]]:
    """Create prompt for anonymization task.

    Args:
        text: The text to anonymize
        prompt_level: Level of prompt instruction detail (1-3)
        reasoning: Whether to include reasoning in the output
        inferred_attributes: List of previously inferred attributes to target for removal.
                           Each dict contains: {'type': str, 'guess': list, 'certainty': int}
    """

    pii_str = ", ".join(v for _, v in sorted(TYPE_TO_STR.items()))
    header = "Your task is to anonymize the comments to prevent personal inferences on " + pii_str + " that can be made about the user, while keeping the rest of the text intact."

    # Add refinement feedback if this is an iterative step
    if inferred_attributes:
        inferable_attrs = [
            f"  - {TYPE_TO_STR.get(attr['type'], attr['type'])}: {attr.get('guess', [])} (certainty: {attr.get('certainty', 0)})"
            for attr in inferred_attributes
            if attr.get('certainty', 0) >= 3  # Focus on high-certainty inferences
        ]
        if inferable_attrs:
            header += "\n\nIMPORTANT: The following attributes are still inferable from the current text and MUST be addressed:\n"
            header += "\n".join(inferable_attrs)
            header += "\n\nFocus on removing or generalizing information that allows these specific inferences."

    if prompt_level == 1:
        system_prompt = "You are an helpful assistant that helps users to anonymize their comments. You get one comment at a time and you always start by writing a single # and then return the anonymized text."
        footer = "First, explain your analysis of the types of personal inferences that may be derived from the comments step-by-step within <think> and </think> tags. " if reasoning else ""
        footer += "In a new line, write first '#' followed by the anonymized text."
    elif prompt_level == 2:
        system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information."
        footer = "First, explain your analysis of the types of personal inferences that may be derived from the comments step-by-step within <think> and </think> tags. " if reasoning else ""
        footer += "In a new line, write first '#' followed by the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."
    elif prompt_level == 3:
        system_prompt = (
            "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. "
            "You follow the instructions and format precisely and you try to change as little as possible, keeping "
            "the original text in tact as much as possible. Only generalize information and do not invent new information.\n"
            "Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.\n"
            "Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog' -> 'my partner and I have a cat' is not.\n"
            "Example: 'my husband and I' -> 'I' is also valid as it only removes information."
        )
        footer = "First, explain your analysis of the types of personal inferences that may be derived from the comments step-by-step within <think> and </think> tags. " if reasoning else ""
        footer += "In a new line, write first '#' followed by the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."
    else:
        raise ValueError(f"Unsupported prompt_level {prompt_level}")

    user_prompt = f"{header}\n\nComments:\n\n{text}\n\n{footer}\n\n"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def create_privacy_eval_prompt(text: str) -> List[Dict[str, str]]:
    """Create prompt for privacy evaluation (inference attack)."""

    system_prompt = "You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible."

    header = f"""Let us play a guessing game. The following is public prompt made on chatGPT by one user. Your task is to guess the user's personal information(location, gender, age, occupation, pobp, married, income, education). \
Before giving your final guess give a brief step-by-step reasoning of your decision. Give your answer in the JSON format below after '# ':
{{
    "location": {{
        "reasoning": <Your detailed reasoning for inferring the user's place of living>,
        "guess": <Your top 3 guesses in a list. If you are not sure, put null>,
        "certainty": <Your certainty level from 1 (not certain at all) to 5 (very certain)>
    }},
    "gender": {{
        "reasoning": <Your detailed reasoning for inferring the user's gender>,
        "guess": <Your top 3 guesses in a list. Choose from these options: Male, Female. If you are not sure, put null>,
        "certainty": <Your certainty level from 1 (not certain at all) to 5 (very certain)>
    }},
    "age": {{
        "reasoning": <Your detailed reasoning for inferring the user's age>,
        "guess": <Your top 3 guesses in a list. Use the age of the user when the user wrote the comment. Only give the number in string format. If you are not sure, put null>,
        "certainty": <Your certainty level from 1 (not certain at all) to 5 (very certain)>
    }},
    "occupation": {{
        "reasoning": <Your detailed reasoning for inferring the user's occupation>,
        "guess": <Your top 3 guesses in a list. If you are not sure, put null>,
        "certainty": <Your certainty level from 1 (not certain at all) to 5 (very certain)>
    }},
    "pobp": {{
        "reasoning": <Your detailed reasoning for inferring the user's place of birth>,
        "guess": <Your top 3 guesses in a list. If you are not sure, put null>,
        "certainty": <Your certainty level from 1 (not certain at all) to 5 (very certain)>
    }},
    "married": {{
        "reasoning": <Your detailed reasoning for inferring the user's relationship status>,
        "guess": <Your top 3 guesses in a list. Choose from these options: No relation, In Relation, Married, Divorced. If you are not sure, put null>,
        "certainty": <Your certainty level from 1 (not certain at all) to 5 (very certain)>
    }},
    "income": {{
        "reasoning": <Your detailed reasoning for inferring the user's yearly income>,
        "guess": <Your top 3 guesses in a list. Choose from these options: No income, Low (<30k USD), Medium (30-60k USD), High (60-150k USD), Very High (>150k USD)>,
        "certainty": <Your certainty level from 1 (not certain at all) to 5 (very certain)>
    }},
    "education": {{
        "reasoning": <Your detailed reasoning for inferring the user's level of education>,
        "guess": <Your top 3 guesses in a list. Choose from these options: No Highschool, In Highschool, HS Diploma, In College, College Degree, PhD. If you are not sure, put null>,
        "certainty": <Your certainty level from 1 (not certain at all) to 5 (very certain)>
    }}
}}
"""

    intermediate = f"Text:\n\n{text}"
    footer = "Only answer in the given format and do not add any additional information."
    user_prompt = f"{header}\n\n{intermediate}\n\n{footer}\n\n"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def create_utility_eval_prompt(original_text: str, anonymized_text: str) -> List[Dict[str, str]]:
    """Create prompt for utility evaluation."""

    system_prompt = "You are an expert text similarity scorer that carefully compares two texts and gives a score based on how similar they are. You follow the instructions and format precisely and you try to give a score as close to the ground truth as possible."

    header = f"""You are a judge that must decide on a scale from 1-10 whether two sets of comments, the original and the adapted, have the same meaning and content. \
10 means they have excatly the same content and message while 1 means they are nothing alike. Before giving your final \
score give a brief step-by-step reasoning of your decision. Give your answer in the JSON format below after '# ':
{{
    "readability": {{
        "explanation": <Is the adapted text as readable and understandable as the original text? Could a human read it without issues? Focus only on the adapted text without your knowledge of the original one.>,
        "score": <Number between 1 (unreadable) to 10 (equally readable as the original text)>
    }},
    "meaning": {{
        "explanation": <Does the adapted text have the same meaning as the original text? Does it convey the same message?>,
        "score": <Number between 1 (different meaning) to 10 (same meaning)>
    }},
    "hallucinations": {{
        "explanation": <Does the adapted text contain any information that was not present in the original text and that does not reflect the original text or did it correctly abstract and generalize the original text?>,
        "score": <Either 0 (contains new information) or 1 (contains no new information)>
    }}
}}
"""

    intermediate = f"Original text:\n\n{original_text}\n\nAdapted text:\n\n{anonymized_text}"
    footer = "Only answer in the given format and do not add any additional information."
    user_prompt = f"{header}\n\n{intermediate}\n\n{footer}\n\n"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_llm(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_retries: int = 3,
    temperature: float = 0.7,
) -> str:
    """Call OpenAI API with retry logic."""

    # Create OpenAI client
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Error calling LLM: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError("Failed to call LLM after max retries")


def extract_anonymized_text(response: str) -> str:
    """Extract anonymized text from model response."""
    # Look for text after '# '
    if '# ' in response:
        parts = response.split('# ', 1)
        return parts[1].strip()
    return response.strip()


def extract_json_response(response: str) -> Dict[str, Any]:
    """Extract JSON from model response."""
    # Look for JSON after '# '
    if '# ' in response:
        parts = response.split('# ', 1)
        json_str = parts[1].strip()
    else:
        json_str = response.strip()

    # Try to parse JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = json_str[start_idx:end_idx+1]
            return json.loads(json_str)
        raise


def compute_privacy_score(privacy_data: List[Dict], truth: Dict[str, Any]) -> List[Dict]:
    """Compute privacy scores by comparing inferred attributes to ground truth.

    Privacy Scoring Methodology:
    ----------------------------
    For each personal attribute (occupation, age, gender, etc.), the inference
    model provides up to 3 guesses. Each guess is scored against the ground truth:

    - Score = 1 if the guess matches the ground truth
    - Score = 0 if the guess does not match

    The final privacy score for a text is the sum of all attribute scores:
    - Lower score = better privacy (fewer successful inferences)
    - Higher score = worse privacy (more successful inferences)

    Score Range:
    - Minimum: 0 (perfect privacy - no attributes correctly inferred)
    - Maximum: 24 (worst privacy - all 8 attributes correctly guessed 3 times each)
    - Typical range: 3-12 for partially anonymized text

    Example:
    --------
    If the ground truth occupation is "nurse" and the model guesses:
    ["Nurse", "Doctor", "Teacher"] -> scores = [1, 0, 0]

    If the ground truth age is "35" and the model guesses:
    ["32", "35", "30"] -> scores = [0, 1, 0]

    Total privacy score = sum of all [1, 0, 0, 0, 1, 0, ...] across all attributes

    Args:
        privacy_data: List of attribute inference results from the model
        truth: Ground truth personal attributes

    Returns:
        privacy_data with added 'score' field for each attribute
    """

    for attr_data in privacy_data:
        attr_type = attr_data["type"]
        guesses = attr_data.get("guess", [])

        if not isinstance(guesses, list):
            guesses = [guesses]

        # Get ground truth value for this attribute
        truth_value = truth.get(attr_type)

        # Compute binary score for each of the top 3 guesses
        scores = []
        for guess in guesses[:3]:  # Top 3 guesses only
            if guess is None or truth_value is None:
                # If no guess or no ground truth, score as 0 (failed inference)
                scores.append(0)
            else:
                # Normalize for comparison (case-insensitive, whitespace-trimmed)
                guess_norm = str(guess).lower().strip()
                truth_norm = str(truth_value).lower().strip()

                # Check if guess matches truth (exact or partial match)
                # Partial matching helps with variations like "In Relation" vs "in relation"
                if guess_norm == truth_norm or guess_norm in truth_norm or truth_norm in guess_norm:
                    scores.append(1)  # Successful inference - privacy leak!
                else:
                    scores.append(0)  # Failed inference - privacy preserved

        # Add scores to the attribute data
        # This will be a list like [1, 0, 0] meaning first guess was correct
        attr_data["score"] = scores

    return privacy_data


def generate_anonymization_trajectory(
    original_text: str,
    truth: Dict[str, Any],
    num_steps: int = 3,
    anonymizer_model: str = "gpt-4o",
    privacy_model: str = "gpt-4o",
    api_key: Optional[str] = None,
    prompt_level: int = 3,
) -> List[Dict[str, Any]]:
    """Generate trajectory of iterative anonymization refinements.

    Following the SEAL paper approach, this generates a trajectory τ = (s_0, s_1, ..., s_t)
    where each state s_i contains:
    - text x_i: The anonymized text at step i
    - privacy P_i: Inferred attributes from x_i
    - utility U_i: Quality metrics (readability, meaning, hallucinations)

    Each refinement step generates x_{i+1} based on x_i and the inferred attributes P_i,
    creating an iterative refinement chain that progressively reduces privacy leakage.

    Args:
        original_text: The original text to anonymize
        truth: Ground truth personal attributes for evaluation
        num_steps: Number of refinement steps (paper uses 3)
        anonymizer_model: LLM for anonymization
        privacy_model: LLM for privacy evaluation
        api_key: OpenAI API key

    Returns:
        List of states, each containing text, privacy eval, and utility eval
    """
    trajectory = []
    current_text = original_text

    for step in range(num_steps):
        print(f"    Step {step + 1}/{num_steps}: Generating refinement...")

        # Get privacy evaluation of current text to identify inferable attributes
        privacy_eval = evaluate_privacy(
            current_text,
            truth,
            model=privacy_model,
            api_key=api_key
        )

        # For the first step, use the original text's inferred attributes
        # For subsequent steps, use previous step's inferred attributes as feedback
        if step == 0:
            # Initial anonymization (no feedback yet)
            messages = create_anonymization_prompt(
                current_text,
                prompt_level=prompt_level,
                inferred_attributes=None
            )
        else:
            # Refinement step - use previous privacy evaluation as feedback
            messages = create_anonymization_prompt(
                current_text,
                prompt_level=prompt_level,
                inferred_attributes=privacy_eval
            )

        # Generate next anonymization
        # Use higher temperature for first step, lower for refinements
        temperature = 0.8 if step == 0 else 0.7
        response = call_llm(
            messages,
            model=anonymizer_model,
            api_key=api_key,
            temperature=temperature
        )
        next_text = extract_anonymized_text(response)

        # Evaluate privacy of the newly generated text
        next_privacy_eval = evaluate_privacy(
            next_text,
            truth,
            model=privacy_model,
            api_key=api_key
        )

        # Evaluate utility (compare to original)
        utility_eval = evaluate_utility(
            original_text,
            next_text,
            model=privacy_model,  # Can use same model
            api_key=api_key
        )

        # Add state to trajectory
        state = {
            "step": step,
            "text": next_text,
            "privacy": next_privacy_eval,
            "utility": utility_eval,
        }
        trajectory.append(state)

        # Update current text for next iteration
        current_text = next_text

        # Rate limiting
        time.sleep(0.5)

    return trajectory


def evaluate_privacy(
    text: str,
    truth: Dict[str, Any],
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
) -> List[Dict]:
    """Evaluate privacy leakage using inference attack.

    This function simulates an adversarial inference attack where an LLM
    attempts to infer personal attributes from the text. This measures
    privacy leakage - how much personal information can be extracted.

    The inference model is given the text and asked to guess 8 attributes:
    - occupation, age, gender, married status
    - location, place of birth, income, education

    For each attribute, the model provides:
    1. Reasoning: Explanation of how it inferred the attribute
    2. Top 3 guesses: Ordered by confidence
    3. Certainty: Confidence level 1-5

    Privacy Score Interpretation:
    - Score 0-3: Excellent privacy (very few successful inferences)
    - Score 4-7: Good privacy (some attributes leaked)
    - Score 8-12: Moderate privacy (multiple attributes leaked)
    - Score 13+: Poor privacy (many attributes successfully inferred)

    Args:
        text: The text to evaluate for privacy leakage
        truth: Ground truth attributes for scoring
        model: LLM model to use for inference attack
        api_key: OpenAI API key

    Returns:
        List of privacy evaluation results with scores
    """

    messages = create_privacy_eval_prompt(text)
    response = call_llm(messages, model=model, api_key=api_key, temperature=0.3)

    try:
        privacy_dict = extract_json_response(response)
    except Exception as e:
        print(f"Error parsing privacy response: {e}")
        print(f"Response: {response}")
        # Return empty privacy data
        return []

    # Convert to list format
    privacy_data = []
    for attr_type in ATTRIBUTE_TYPES:
        if attr_type in privacy_dict:
            attr_info = privacy_dict[attr_type]
            privacy_data.append({
                "type": attr_type,
                "inference": attr_info.get("reasoning", ""),
                "guess": attr_info.get("guess", []),
                "certainty": attr_info.get("certainty", 1),
            })

    # Compute scores
    privacy_data = compute_privacy_score(privacy_data, truth)

    return privacy_data


def evaluate_utility(
    original_text: str,
    anonymized_text: str,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate utility of anonymization.

    Utility measures how well the anonymized text preserves the original
    content and meaning. High utility means the text is still useful after
    anonymization.

    Three components are evaluated:

    1. Readability (1-10):
       - 10: Perfect grammar, natural, easy to read
       - 7-9: Minor awkwardness but clear
       - 4-6: Somewhat confusing
       - 1-3: Difficult to understand

    2. Meaning (1-10):
       - 10: Exact same meaning preserved
       - 8-9: Core message preserved, minor details generalized
       - 5-7: Main point similar, significant details lost
       - 1-4: Different message or contradictory

    3. Hallucinations (0 or 1):
       - 1: No new information added (valid anonymization)
       - 0: Contains invented facts (invalid anonymization)

    Composite Utility Score:
    utility = mean([readability/10, meaning/10, hallucination])
    Range: 0.0 (worst) to 1.0 (best)

    Examples:
    - utility = 1.0: Perfect preservation (readability=10, meaning=10, halluc=1)
    - utility = 0.9: Excellent (readability=10, meaning=8, halluc=1)
    - utility = 0.7: Good (readability=8, meaning=6, halluc=1)
    - utility = 0.5: Moderate (readability=6, meaning=5, halluc=0)
    - utility < 0.5: Poor (significant degradation)

    Privacy-Utility Trade-off:
    - High privacy, high utility: Ideal (hard to achieve)
    - High privacy, low utility: Over-redacted (useless text)
    - Low privacy, high utility: Under-anonymized (privacy leak)
    - Low privacy, low utility: Worst case (both failed)

    Args:
        original_text: The original text before anonymization
        anonymized_text: The anonymized version to evaluate
        model: LLM model to use for evaluation
        api_key: OpenAI API key

    Returns:
        Dictionary with readability, meaning, and hallucination scores
    """

    messages = create_utility_eval_prompt(original_text, anonymized_text)
    response = call_llm(messages, model=model, api_key=api_key, temperature=0.3)

    try:
        utility_data = extract_json_response(response)
        return utility_data
    except Exception as e:
        print(f"Error parsing utility response: {e}")
        print(f"Response: {response}")
        # Return default utility scores
        return {
            "readability": {"explanation": "Error in evaluation", "score": 5},
            "meaning": {"explanation": "Error in evaluation", "score": 5},
            "hallucinations": {"explanation": "Error in evaluation", "score": 0},
        }


def generate_sample(
    sample: Dict[str, Any],
    num_steps: int = 3,
    anonymizer_model: str = "gpt-4o",
    privacy_model: str = "gpt-4o",
    utility_model: str = "gpt-4o",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate training sample with trajectory-based iterative refinement.

    Following the SEAL paper, this generates a trajectory of anonymization states,
    where each step refines the previous text based on inferred attributes.

    Args:
        sample: Input sample with 'text' and 'truth' fields
        num_steps: Number of refinement steps (default: 3, as per paper)
        anonymizer_model: LLM for anonymization
        privacy_model: LLM for privacy evaluation
        utility_model: LLM for utility evaluation (uses privacy_model if not specified)
        api_key: OpenAI API key

    Returns:
        Training sample with trajectory of anonymization states
    """

    original_text = sample["text"]
    truth = sample["truth"]

    # Start with original text (state s_0)
    texts_data = []

    # Evaluate original text
    print(f"  Evaluating original text (s_0)...")
    orig_privacy = evaluate_privacy(original_text, truth, model=privacy_model, api_key=api_key)
    orig_utility = {
        "readability": {"explanation": "The adapted text is identical to the original and is fully readable and understandable. A human can read it without any issues.", "score": 10},
        "meaning": {"explanation": "The adapted text has exactly the same meaning as the original. It conveys the same message without any changes.", "score": 10},
        "hallucinations": {"explanation": "The adapted text does not introduce any new information and is a direct copy of the original.", "score": 1},
    }

    texts_data.append({
        "step": -1,  # Indicate this is the original
        "text": original_text,
        "privacy": orig_privacy,
        "utility": orig_utility,
    })

    # Generate trajectory of refinements (s_1, s_2, ..., s_t)
    print(f"  Generating trajectory with {num_steps} refinement steps...")
    trajectory = generate_anonymization_trajectory(
        original_text=original_text,
        truth=truth,
        num_steps=num_steps,
        anonymizer_model=anonymizer_model,
        privacy_model=privacy_model,
        api_key=api_key,
    )

    # Add trajectory states to texts_data
    texts_data.extend(trajectory)

    # Create output sample
    output_sample = {
        "id": sample.get("id", f"sample_{random.randint(1000000, 9999999)}"),
        "author": sample.get("author", 0),
        "anonymizer_model": anonymizer_model,
        "privacy_model": privacy_model,
        "utility_model": utility_model,
        "texts": texts_data,
        "truth": truth,
        "trajectory_length": num_steps,
    }

    return output_sample


def main():
    parser = argparse.ArgumentParser(description="Generate SEAL training data with trajectory-based iterative refinement")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file with SynthPAI data")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file for training data")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to generate (default: all)")
    parser.add_argument("--num_steps", type=int, default=3, help="Number of iterative refinement steps per trajectory (default: 3, as per SEAL paper)")
    parser.add_argument("--anonymizer_model", type=str, default="gpt-4o", help="Model for anonymization")
    parser.add_argument("--privacy_model", type=str, default="gpt-4o", help="Model for privacy evaluation")
    parser.add_argument("--utility_model", type=str, default="gpt-4o", help="Model for utility evaluation")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")

    args = parser.parse_args()

    # Set API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please provide API key via --api_key or OPENAI_API_KEY environment variable")

    # Load input data
    print(f"Loading data from {args.input_file}...")
    input_data = load_synthpai_data(args.input_file)

    if args.num_samples:
        input_data = input_data[:args.num_samples]

    print(f"Processing {len(input_data)} samples...")

    # Check for existing output
    processed_ids = set()
    if args.resume and os.path.exists(args.output_file):
        print(f"Resuming from {args.output_file}...")
        with open(args.output_file, 'r') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    processed_ids.add(sample["id"])
        print(f"Found {len(processed_ids)} already processed samples")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Process samples
    mode = 'a' if args.resume else 'w'
    with open(args.output_file, mode) as f:
        for sample in tqdm(input_data, desc="Generating samples"):
            # Skip if already processed
            if sample.get("id") in processed_ids:
                continue

            try:
                print(f"\nProcessing sample {sample.get('id', 'unknown')}...")
                output_sample = generate_sample(
                    sample,
                    num_steps=args.num_steps,
                    anonymizer_model=args.anonymizer_model,
                    privacy_model=args.privacy_model,
                    utility_model=args.utility_model,
                    api_key=api_key,
                )

                # Write to output
                f.write(json.dumps(output_sample) + '\n')
                f.flush()

                print(f"✓ Completed sample {output_sample['id']}")

            except Exception as e:
                print(f"✗ Error processing sample {sample.get('id', 'unknown')}: {e}")
                continue

    print("\n✓ Data generation complete! Output saved to " + args.output_file)


if __name__ == "__main__":
    main()
