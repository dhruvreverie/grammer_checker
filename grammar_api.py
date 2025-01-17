from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from unsloth import FastLanguageModel
import difflib
from typing import Optional, List, Set, Dict, Tuple
import torch.nn.functional as F

app = FastAPI(title="Hindi Grammar Correction API", 
              description="API for correcting grammar with confidence scores and change classification")

# Load models
max_seq_length = 500
dtype = None
load_in_4bit = False
model_name = "/root/grammer_hindi/llama32_3b_model"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)

class HindiGrammarChangeClassifier:
    def __init__(self):
        # Common Hindi pronouns
        self.pronouns = {
            'personal': ['मैं', 'तुम', 'वह', 'हम', 'आप', 'वे', 'मुझे', 'तुम्हें', 'उसे', 'हमें', 'आपको', 'उन्हें'],
            'possessive': ['मेरा', 'तेरा', 'उसका', 'हमारा', 'आपका', 'उनका'],
            'demonstrative': ['यह', 'वह', 'ये', 'वे']
        }
        
        # Gender-specific verb and adjective endings
        self.gender_pairs = [
            ('ी', 'ा'),   
            ('ीं', 'े'),   
            ('ती', 'ता'), 
            ('ती', 'ते'), 
            ('यी', 'या'), 
            ('यीं', 'ये'), 
            ('िन', 'ा'),   
            ('इन', 'ा')   
        ]
        
        # Common verb forms
        self.verb_forms = [
            'ना', 'ता', 'ती', 'ते', 'रहा', 'रही', 'रहे',
            'गा', 'गी', 'गे', 'कर', 'के', 'हूँ', 'है', 'हैं'
        ]
        
        # Common postpositions with gender variations
        self.postpositions = {
            'masculine': ['का', 'के', 'वाला', 'वाले'],
            'feminine': ['की', 'वाली'],
            'neutral': ['को', 'से', 'में', 'पर', 'तक', 'ने']
        }

    def classify_changes(self, original: str, corrected: str) -> Dict[str, List[Tuple[str, str]]]:
        changes = {
            'pronoun_changes': [],
            'gender_changes': [],
            'verb_form_changes': [],
            'postposition_changes': [],
            'spelling_changes': [],
            'word_order_changes': [],
            'other_corrections': []  # For unclassified changes
        }
        
        orig_words = original.split()
        corr_words = corrected.split()
        
        # Check for word order changes first
        if orig_words != corr_words and len(orig_words) == len(corr_words):
            changes['word_order_changes'].append((original, corrected))
        
        for orig_word, corr_word in zip(orig_words, corr_words):
            if orig_word != corr_word:
                change_classified = False
                
                # Check gender changes first as they can overlap with other categories
                if self._is_gender_change(orig_word, corr_word):
                    changes['gender_changes'].append((orig_word, corr_word))
                    change_classified = True
                
                # Check pronoun changes
                if self._is_pronoun_change(orig_word, corr_word):
                    changes['pronoun_changes'].append((orig_word, corr_word))
                    change_classified = True
                
                # Check verb form changes (non-gender related)
                if self._is_verb_form_change(orig_word, corr_word) and not self._is_gender_change(orig_word, corr_word):
                    changes['verb_form_changes'].append((orig_word, corr_word))
                    change_classified = True
                
                # Check postposition changes
                if self._is_postposition_change(orig_word, corr_word):
                    changes['postposition_changes'].append((orig_word, corr_word))
                    change_classified = True
                
                # If the change wasn't classified in any category, check for spelling
                if not change_classified and self._is_spelling_change(orig_word, corr_word):
                    changes['spelling_changes'].append((orig_word, corr_word))
                    change_classified = True
                
                # If still not classified, add to other_corrections
                if not change_classified:
                    changes['other_corrections'].append((orig_word, corr_word))
        
        return changes
    
    def _is_gender_change(self, orig_word: str, corr_word: str) -> bool:
        """Enhanced gender change detection for verb and adjective agreement"""
        # Remove punctuation for comparison
        orig_clean = orig_word.rstrip('।,!?.')
        corr_clean = corr_word.rstrip('।,!?.')
        
        # Check verb and adjective gender pairs
        for fem, masc in self.gender_pairs:
            # Check feminine to masculine change
            if orig_clean.endswith(fem) and corr_clean.endswith(masc):
                return True
            # Check masculine to feminine change
            if orig_clean.endswith(masc) and corr_clean.endswith(fem):
                return True
        
        # Check postposition gender changes
        if (orig_clean in self.postpositions['masculine'] and corr_clean in self.postpositions['feminine']) or \
           (orig_clean in self.postpositions['feminine'] and corr_clean in self.postpositions['masculine']):
            return True
        
        return False
    
    def _is_spelling_change(self, orig_word: str, corr_word: str) -> bool:
        """Detect pure spelling changes (not gender/grammar related)"""
        # If the words are similar but not identical, and not a grammar change
        return (not self._is_gender_change(orig_word, corr_word) and
                not self._is_pronoun_change(orig_word, corr_word) and
                not self._is_verb_form_change(orig_word, corr_word) and
                not self._is_postposition_change(orig_word, corr_word))
    
    def _is_pronoun_change(self, orig_word: str, corr_word: str) -> bool:
        """Check if the change involves pronouns"""
        all_pronouns = [p for sublist in self.pronouns.values() for p in sublist]
        return orig_word in all_pronouns or corr_word in all_pronouns
    
    def _is_verb_form_change(self, orig_word: str, corr_word: str) -> bool:
        """Check if the change involves verb forms (non-gender related)"""
        return any(
            (orig_word.endswith(form) or corr_word.endswith(form))
            for form in self.verb_forms
        ) and not self._is_gender_change(orig_word, corr_word)
    
    def _is_postposition_change(self, orig_word: str, corr_word: str) -> bool:
        """Check if the change involves postpositions"""
        all_postpositions = (self.postpositions['masculine'] + 
                           self.postpositions['feminine'] + 
                           self.postpositions['neutral'])
        return orig_word in all_postpositions or corr_word in all_postpositions

# Initialize classifier
hindi_classifier = HindiGrammarChangeClassifier()

# Pydantic Models
class SentenceInput(BaseModel):
    sentences: List[str]

class WordChanges(BaseModel):
    added_words: List[str]
    removed_words: List[str]
    word_additions: int
    word_removals: int

class GrammarChanges(BaseModel):
    pronoun_changes: List[List[str]]
    gender_changes: List[List[str]]
    verb_form_changes: List[List[str]]
    postposition_changes: List[List[str]]
    spelling_changes: List[List[str]]
    word_order_changes: List[List[str]]

class ConfidenceMetrics(BaseModel):
    token_confidence: float
    entropy_confidence: float
    topk_confidence: float
    word_changes: WordChanges
    grammar_changes: GrammarChanges

class CorrectionResult(BaseModel):
    original: str
    corrected: str
    confidence_metrics: ConfidenceMetrics

class CorrectionOutput(BaseModel):
    corrections: List[CorrectionResult]

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

def analyze_word_changes(original: str, corrected: str) -> dict:
    """Analyze word additions and removals between original and corrected text"""
    original_words = set(original.lower().split())
    corrected_words = set(corrected.lower().split())
    
    added = corrected_words - original_words
    removed = original_words - corrected_words
    
    return {
        'added_words': sorted(list(added)),
        'removed_words': sorted(list(removed)),
        'word_additions': len(added),
        'word_removals': len(removed)
    }

def get_comprehensive_confidence(model, tokenizer, original: str, corrected: str) -> dict:
    """Calculate confidence metrics for the correction"""
    
    # Token Probabilities Confidence
    inputs = tokenizer(corrected, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
        max_probs = torch.max(probs, dim=-1).values
        token_conf = max_probs.mean().item() * 100
        
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        max_entropy = -torch.log(torch.tensor(1.0 / probs.size(-1)))
        entropy_conf = (1 - entropy / max_entropy).mean().item() * 100
        
        k = 5
        top_k_probs = torch.topk(probs, k=k, dim=-1).values
        topk_conf = (top_k_probs[:,:,0] / top_k_probs.sum(dim=-1)).mean().item() * 100

    # Word Changes Analysis
    word_changes = analyze_word_changes(original, corrected)
    
    # Grammar Changes Analysis
    grammar_changes_dict = hindi_classifier.classify_changes(original, corrected)
    grammar_changes = {
        k: [list(change) for change in v] for k, v in grammar_changes_dict.items()
    }

    return {
        'token_confidence': round(token_conf, 2),
        'entropy_confidence': round(entropy_conf, 2),
        'topk_confidence': round(topk_conf, 2),
        'word_changes': word_changes,
        'grammar_changes': grammar_changes
    }

def correct_sentence(sent: str) -> str:
    """Correct a single sentence"""
    max_new_tokens = 500
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Correct any grammar mistakes in the sentence while preserving its original meaning.",
                f"{sent}",
                "",
            )
        ],
        return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    text = tokenizer.batch_decode(outputs)
    text = ''.join(text)
    start_marker = '### Response:\n'
    start_index = text.find(start_marker) + len(start_marker)
    end_index = text.find('<eos>', start_index)
    response = text[start_index:end_index].strip()
    response = response.replace("<|end_of_text|", "").strip()
    return response

@app.post("/correct_grammar", response_model=CorrectionOutput)
async def correct_grammar(input_data: SentenceInput):
    if not input_data.sentences:
        raise HTTPException(status_code=400, detail="No sentences provided")
    
    corrections = []
    
    for sentence in input_data.sentences:
        if not sentence.strip():
            continue
            
        corrected_sentence = correct_sentence(sentence)
        confidence_metrics = get_comprehensive_confidence(
            model, tokenizer, sentence, corrected_sentence
        )
        
        correction = CorrectionResult(
            original=sentence,
            corrected=corrected_sentence,
            confidence_metrics=ConfidenceMetrics(**confidence_metrics)
        )
        
        corrections.append(correction)
    
    if not corrections:
        raise HTTPException(status_code=400, detail="No valid sentences provided")
    
    return CorrectionOutput(corrections=corrections)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)