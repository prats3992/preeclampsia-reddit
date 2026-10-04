"""Study parameters: subreddit weights, LLM-suggested keyword lists and collection limits.

Subreddit and keyword lists were proposed independently by three LLMs (Claude Sonnet 4.5,
Gemini 2.5 Pro, GPT-5); their raw answers are kept in ``docs/llm_suggestions/``.
"""

import pandas as pd

# Subreddit weights based on relevance to pre-eclampsia
# Higher weight = more focused on the topic
SUBREDDIT_WEIGHTS = {
    # High relevance - dedicated or high-risk pregnancy subreddits (1.0)
    'preeclampsia': {'weight': 1.0, 'llm': ['claude', 'gemini', 'gpt5'], 'focus': 'dedicated'},
    'highriskpregnancy': {'weight': 1.0, 'llm': ['gemini'], 'focus': 'dedicated'},
    'NICUParents': {'weight': 0.9, 'llm': ['claude', 'gemini'], 'focus': 'high'},
    
    # Medium-high relevance - pregnancy with complications focus (0.7-0.8)
    'PregnancyAfter35': {'weight': 0.8, 'llm': ['claude'], 'focus': 'high'},
    'CautiousBB': {'weight': 0.8, 'llm': ['claude'], 'focus': 'high'},
    'birthstories': {'weight': 0.7, 'llm': ['gemini'], 'focus': 'medium'},
    'GestationalDiabetes': {'weight': 0.7, 'llm': ['gemini'], 'focus': 'medium'},
    'InfertilityBabies': {'weight': 0.7, 'llm': ['gemini'], 'focus': 'medium'},
    'Infertility': {'weight': 0.6, 'llm': ['gpt5'], 'focus': 'medium'},
    
    # Medium relevance - general pregnancy subreddits (0.5-0.6)
    'BabyBumps': {'weight': 0.6, 'llm': ['claude', 'gemini', 'gpt5'], 'focus': 'medium'},
    'pregnant': {'weight': 0.6, 'llm': ['claude', 'gemini'], 'focus': 'medium'},
    'pregnancy': {'weight': 0.6, 'llm': ['gpt5'], 'focus': 'medium'},
    'beyondthebump': {'weight': 0.5, 'llm': ['claude', 'gemini'], 'focus': 'medium'},
    'BabyBumpsandBeyondAu': {'weight': 0.5, 'llm': ['claude'], 'focus': 'medium'},
    
    # Lower relevance - broad parenting/medical subreddits (0.3-0.4)
    'Mommit': {'weight': 0.4, 'llm': ['claude', 'gpt5'], 'focus': 'broad'},
    'Parenting': {'weight': 0.4, 'llm': ['gpt5'], 'focus': 'broad'},
    'ParentingInBulk': {'weight': 0.4, 'llm': ['claude'], 'focus': 'broad'},
    'AskDocs': {'weight': 0.4, 'llm': ['gemini', 'gpt5'], 'focus': 'broad'},
    'ObGyn': {'weight': 0.3, 'llm': ['gpt5'], 'focus': 'broad'},
    'Nursing': {'weight': 0.3, 'llm': ['gpt5'], 'focus': 'broad'},
    'TwoXChromosomes': {'weight': 0.3, 'llm': ['gpt5'], 'focus': 'broad'},
}

# Keyword categorization by LLM with priority weights
KEYWORDS_BY_LLM = {
    'claude': {
        'core_terms': [
            'preeclampsia', 'pre-eclampsia', 'pre eclampsia',
            'eclampsia', 'HELLP syndrome', 'gestational hypertension',
            'toxemia', 'pregnancy-induced hypertension', 'PIH',
            'superimposed preeclampsia'
        ],
        'symptoms': [
            'high blood pressure pregnancy', 'protein in urine pregnancy',
            'proteinuria', 'severe swelling pregnancy', 'edema pregnancy',
            'vision changes pregnancy', 'blurred vision',
            'right upper quadrant pain', 'severe headache pregnancy',
            'elevated liver enzymes'
        ],
        'monitoring': [
            'magnesium sulfate', 'mag sulfate', '24-hour urine collection',
            'early delivery', 'premature birth', 'induced labor preeclampsia',
            'NST monitoring', 'blood pressure monitoring'
        ],
        'outcomes': [
            'preterm delivery', 'NICU admission', 'postpartum preeclampsia',
            'emergency c-section'
        ]
    },
    'gemini': {
        'core_terms': [
            'preeclampsia', 'pre-eclampsia', 'toxemia', 'HELLP',
            'Eclampsia', 'Postpartum preeclampsia', 'High BP'
        ],
        'symptoms': [
            'High blood pressure', 'Swelling', 'Headache',
            'Protein in urine', 'Proteinuria', 'Blurry vision',
            'Seeing spots', 'Upper abdominal pain', 'Rib pain',
            'Elevated liver enzymes'
        ],
        'monitoring': [
            'Magnesium sulfate', 'Mag drip', 'Labetalol',
            'Nifedipine', 'Procardia', 'Bed rest',
            'Induction', 'Induced', 'Early delivery'
        ],
        'diagnostic': [
            'Diagnosed with preeclampsia', 'Signs of preeclampsia',
            'Worried about preeclampsia', '24-hour urine',
            'Non-stress test', 'NST'
        ]
    },
    'gpt5': {
        'core_terms': [
            'preeclampsia', 'pre-eclampsia', 'eclampsia',
            'HELLP syndrome', 'gestational hypertension',
            'high blood pressure during pregnancy', 'protein in urine'
        ],
        'symptoms': [
            'swelling', 'headache', 'vision problems',
            'dizziness', 'blurred vision', 'upper abdominal pain',
            'nausea', 'edema'
        ],
        'monitoring': [
            'magnesium sulfate', 'bed rest', 'induced labor',
            'C-section', 'monitoring blood pressure', 'prenatal checkups'
        ],
        'emotional': [
            'scared', 'anxious', 'doctor didn\'t listen',
            'NICU baby', 'support group', 'recovery',
            'postpartum', 'birth story'
        ],
        'context': [
            'third trimester', 'high-risk pregnancy',
            'complications', 'hospital stay'
        ]
    }
}

# Keyword weights by category
KEYWORD_WEIGHTS = {
    'core_terms': 1.0,      # Direct medical terminology
    'symptoms': 0.8,        # Symptom descriptions
    'monitoring': 0.7,      # Treatment and monitoring
    'diagnostic': 0.7,      # Diagnostic experiences
    'outcomes': 0.6,        # Complication outcomes
    'emotional': 0.5,       # Emotional/support language
    'context': 0.5          # Contextual terms
}

# r/preeclampsia was created on 2013-06-29; posts on/after COVID_START are "Post-COVID".
COVID_START = pd.Timestamp('2020-03-01')

DATA_COLLECTION_CONFIG = {
    'base_posts_per_subreddit': 10_000,  # scaled by subreddit weight
    'max_comments_per_post': 50,
    'min_comment_length': 50,            # characters
}

# Seconds to pause between subreddits to stay well inside Reddit's rate limit
RATE_LIMIT_SECONDS = 2


def get_post_limit_for_subreddit(subreddit):
    """Posts to request for a subreddit: base limit x weight, clamped to [50, 10 000]."""
    weight = SUBREDDIT_WEIGHTS.get(subreddit, {}).get('weight', 0.3)
    post_limit = int(DATA_COLLECTION_CONFIG['base_posts_per_subreddit'] * weight)
    return max(50, min(post_limit, 10_000))
