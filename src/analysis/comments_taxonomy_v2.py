#!/usr/bin/env python3
"""
Comments Interaction Taxonomy Classifier (12 Categories)
Standalone pipeline for reddit_comments.csv analysis.
"""
import re, json, os, pandas as pd, numpy as np
from collections import Counter, defaultdict

INPUT = "final_outputs/comments_analysis/comments_preprocessed.csv"
OUTPUT_DIR = "final_outputs/comments_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 12-Category Taxonomy ────────────────────────────────────────────────────
# Each category has a weighted keyword dictionary and a set of regex patterns.
TAXONOMY = {
    "support_empathy": {
        "weight": 1,
        "keywords": {
            # Direct empathy
            "i feel you": 3, "i understand": 3, "i get it": 2, "same here": 2,
            "me too": 2, "relatable": 2, "you are not alone": 3, "not alone": 2,
            "hugs": 2, "sending love": 3, "sending hugs": 3, "i am here": 2,
            "here for you": 3, "with you": 1, "we are with you": 2,
            "sab theek hoga": 2, "tension mat le": 2, "koi baat nahi": 1,
            "chinta mat karo": 1, "fikar mat karo": 1, "it is okay": 2,
            "it is alright": 2, "it will be okay": 2, "you will be fine": 2,
            "take care": 1, "stay safe": 1, "be safe": 1, "look after yourself": 2,
            "self care": 2, "self-care": 2, "mental health matters": 3,
            "therapy helps": 2, "seek help": 2, "reach out": 2,
            "please talk": 2, "talk to someone": 2, "you deserve help": 3,
            "you deserve better": 2, "you deserve peace": 2, "not your fault": 2,
            "not your mistake": 2, "do not blame yourself": 2, "do not blame": 1,
            # Emotional support words
            "empathy": 2, "empathetic": 2, "compassion": 2, "compassionate": 2,
            "sympathy": 1, "sympathetic": 1, "understanding": 1, "caring": 1,
            "kind": 1, "kindness": 1, "gentle": 1, "gentle with yourself": 2,
            "love yourself": 2, "self love": 1, "self-love": 1, "accept yourself": 2,
            "forgive yourself": 2, "healing": 1, "heal": 1, "recovery": 1,
            "recover": 1, "better days": 1, "brighter days": 1, "this too shall pass": 2,
            "time heals": 1, "give yourself time": 2, "be patient": 1, "patience": 1,
        },
    },
    "advice_guidance": {
        "weight": 1,
        "keywords": {
            # Direct advice
            "do this": 2, "try this": 2, "focus on": 2, "practice": 1, "recommend": 2,
            "suggest": 2, "strategy": 2, "plan": 1, "schedule": 1, "routine": 1,
            "tip": 1, "trick": 1, "hack": 1, "advice": 2, "guidance": 2,
            "counsel": 1, "should": 1, "must": 1, "need to": 1, "have to": 1,
            "better to": 1, "would suggest": 2, "my advice": 2, "what worked for me": 2,
            "in my experience": 2, "i suggest": 2, "i recommend": 2, "try doing": 2,
            "make sure": 1, "ensure": 1, "avoid": 1, "do not do": 1,
            "rather than": 1, "instead of": 1, "focus": 1, "prioritize": 1,
            "dedicate": 1, "devote": 1, "time table": 2, "timetable": 2,
            "planning": 1, "organize": 1, "manage time": 2, "self study": 2,
            "pyq": 1, "previous year": 1, "ncert": 1, "notes": 1, "revision": 1,
            "revise": 1, "memorize": 1, "understand concept": 1, "solve": 1,
            "mock test": 1, "test series": 1, "coaching": 1, "batch": 1,
            "online": 1, "youtube": 1, "unacademy": 1, "physics wallah": 1,
            "pw": 1, "allen": 1, "akash": 1, "resonance": 1, "vibrant": 1,
            "study material": 1, "resources": 1, "book": 1, "reference": 1,
            "formula": 1, "derivation": 1, "shortcut": 1, "method": 1,
            "technique": 1, "approach": 1, "way to": 1, "how to": 1,
            "what to": 1, "when to": 1, "where to": 1, "which": 1,
            "choose": 1, "select": 1, "pick": 1, "opt": 1, "go for": 1,
            "aim for": 1, "target": 1, "goal": 1, "objective": 1, "aim": 1,
            "aspiration": 1, "therapy": 1, "counselor": 1, "counsellor": 1,
            "therapist": 1, "psychiatrist": 1, "psychologist": 1, "medication": 1,
            "meds": 1, "exercise": 1, "meditation": 1, "yoga": 1, "jogging": 1,
            "sleep well": 1, "eat well": 1, "diet": 1, "hydration": 1,
            "hydrate": 1, "drink water": 1, "breathing exercise": 1, "deep breath": 1,
            "grounding technique": 1, "mindfulness": 1, "journaling": 1, "journal": 1,
            "write down": 1, "talk to friend": 1, "talk to family": 1,
            "call helpline": 2, "helpline": 2, "crisis helpline": 2, "samaritans": 2,
            "vandrevala": 2, "icall": 2, "tele manas": 2, "aasha": 2,
        },
    },
    "personal_experience": {
        "weight": 1,
        "keywords": {
            "i went through": 3, "i have been through": 3, "i experienced": 2,
            "i faced": 2, "i had the same": 2, "same thing happened": 2,
            "happened to me": 2, "i can relate": 2, "i relate": 2,
            "my story": 2, "my experience": 2, "when i was": 1, "back when i": 1,
            "i used to": 1, "i was in": 1, "i dropped": 2, "i took a drop": 2,
            "i failed": 2, "i failed jee": 2, "i failed neet": 2, "i got backlog": 2,
            "i got backlogs": 2, "i was depressed": 2, "i had depression": 2,
            "i had anxiety": 2, "i was anxious": 2, "i attempted": 2,
            "i tried to": 1, "i survived": 2, "i overcame": 2, "i recovered": 2,
            "got through it": 2, "made it through": 2, "pulled through": 2,
            "things got better": 2, "it gets better": 2, "it got better": 2,
            "now i am": 1, "currently i": 1, "now i work": 1, "now i study": 1,
            "i am at": 1, "i am in": 1, "i am doing": 1, "i am feeling": 1,
            "last year": 1, "two years ago": 1, "three years ago": 1,
            "last sem": 1, "last semester": 1, "during my": 1, "in my first": 1,
            "in my second": 1, "in my third": 1, "in my final": 1,
            "in 12th": 1, "in 11th": 1, "in 10th": 1, "after 12th": 1,
            "after boards": 1, "after jee": 1, "after neet": 1, "after my": 1,
            "mujhe bhi": 1, "mere sath bhi": 2, "mere saath bhi": 2,
            "maine bhi": 1, "meri bhi": 1, "mujhe bhi yahi": 2, "same bhai": 1,
            "same bhai same": 2, "bhai same": 1, "yaar same": 1,
        },
    },
    "encouragement_motivation": {
        "weight": 1,
        "keywords": {
            "do not give up": 3, "never give up": 3, "keep going": 2, "keep pushing": 2,
            "keep trying": 2, "hang in there": 2, "push through": 2, "stay strong": 3,
            "you can do it": 3, "you will do it": 3, "you are capable": 2,
            "you are talented": 2, "you are smart": 2, "you are strong": 2,
            "you are brave": 2, "you are not a failure": 2, "not a failure": 2,
            "not a loser": 2, "not worthless": 2, "not useless": 2, "you matter": 2,
            "you are important": 2, "your life matters": 2, "life matters": 2,
            "worth living": 2, "life is worth": 2, "worth it": 1, "it is worth it": 2,
            "believe in yourself": 3, "believe in you": 2, "have faith": 2,
            "have confidence": 2, "confidence": 1, "self belief": 1, "self-belief": 1,
            "do not lose hope": 2, "do not lose": 1, "do not quit": 2, "do not stop": 1,
            "rise again": 2, "get back up": 2, "stand up": 1, "fight back": 1,
            "stronger": 1, "become stronger": 2, "grow from this": 2, "learn from this": 1,
            "this will make you": 2, "you will grow": 2, "you will learn": 1,
            "one day at a time": 2, "small steps": 1, "step by step": 1,
            "progress not perfection": 2, "you are enough": 2, "you are worthy": 2,
            "you deserve happiness": 2, "deserve to be happy": 2, "deserve happiness": 2,
            "deserve peace": 2, "deserve love": 2, "all the best": 1, "best of luck": 1,
            "good luck": 1, "break a leg": 1, "you got this": 2, "you have got this": 2,
            "ho jayega": 2, "sab thik ho jayega": 2, "tum kar sakte ho": 2,
            "tu kar sakta hai": 2, "himmat mat harna": 2, "himmat rakh": 2,
            "shabash": 2, "dabang": 1, "wah": 1, "proud of you": 2, "well done": 1,
            "good job": 1, "congratulations": 1, "happy for you": 1, "glad to hear": 1,
            "relief": 1, "comfort": 1, "motivate": 1, "encourage": 1, "inspire": 1,
            "inspiration": 1, "proud": 1, "love": 1, "care": 1, "empower": 1,
            "empowering": 1, "lift you up": 2, "pick you up": 1, "carry on": 1,
            "keep your head up": 2, "head up": 1, "chin up": 1, "do not look down": 1,
        },
    },
    "information_resources": {
        "weight": 1,
        "keywords": {
            "link": 1, "check out": 1, "here is": 1, "here is a": 1, "resource": 1,
            "guide": 1, "tutorial": 1, "video": 1, "article": 1, "post": 1,
            "reddit": 1, "subreddit": 1, "forum": 1, "community": 1, "group": 1,
            "app": 1, "website": 1, "platform": 1, "tool": 1, "software": 1,
            "book recommendation": 1, "book i recommend": 1, "youtube channel": 1,
            "channel": 1, "playlist": 1, "course": 1, "online course": 1,
            "mooc": 1, "coursera": 1, "udemy": 1, "edx": 1, "nptel": 1,
            "unacademy": 1, "pw vip": 1, "physics wallah": 1, "allen": 1,
            "akash": 1, "resonance": 1, "vibrant": 1, "motion": 1,
            "byjus": 1, "vedantu": 1, "whitehat jr": 1, "gradeup": 1,
            "testbook": 1, "career360": 1, "shiksha": 1, "collegedunia": 1,
            "glassdoor": 1, "linkedin": 1, "naukri": 1, "internshala": 1,
            "indeed": 1, "angelList": 1, "angel list": 1, "hackerRank": 1,
            "hackerrank": 1, "hackerEarth": 1, "hackerearth": 1, "codeChef": 1,
            "codechef": 1, "codeForces": 1, "codeforces": 1, "leetcode": 1,
            "dsa sheet": 1, "striver": 1, "love babbar": 1, "apna college": 1,
            "take u forward": 1, "neet pg": 1, "jee mains": 1, "jee advanced": 1,
            "upsc": 1, "cat exam": 1, "gate exam": 1, "nift": 1, "nid": 1,
            "clat": 1, "nlu": 1, "nit": 1, "iit": 1, "iiit": 1, "nit trichy": 1,
            "nit warangal": 1, "nit surathkal": 1, "nit calicut": 1,
            "bits pilani": 1, "bits": 1, "vit vellore": 1, "vit": 1, "srm": 1,
            "manipal": 1, "thapar": 1, "lpu": 1, "amity": 1, "bennett": 1,
        },
    },
    "neutral_discussion": {
        "weight": 1,
        "keywords": {},  # Will be detected by absence of other categories + question marks + factual language
        "patterns": [
            r'\?',  # Questions
        ],
    },
    "humor_meme_coping": {
        "weight": 1,
        "keywords": {
            "lol": 2, "lmao": 2, "rofl": 2, "haha": 1, "hehe": 1, "hahaha": 1,
            "lolol": 1, "xd": 1, "xddd": 1, "kekw": 1, "bruh": 1, "bro": 1,
            "bhai": 1, "dekho": 1, "dekh": 1, "batao": 1, "bata": 1, "sun": 1,
            "suno": 1, "abey": 1, "oye": 1, "arre": 1, "arey": 1, "bhaiya": 1,
            "cooked": 2, "bodied": 1, "ratio": 1, "based": 1, "chad": 1,
            "sigma": 1, "npc": 1, "sus": 1, "imposter": 1, "among us": 1,
            "meme": 1, "funny": 1, "joke": 1, "sarcasm": 1, "sarcastic": 1,
            "ironically": 1, "irony": 1, "satire": 1, "parody": 1, "mock": 1,
            "tease": 1, "banter": 1, "troll": 1, "trolling": 1, "shitpost": 1,
            "shitposting": 1, "hilarious": 1, "comedy": 1, "comedian": 1,
            "laugh": 1, "laughing": 1, "lmfao": 1, "roflmao": 1, "pepe": 1,
            "wojak": 1, "doge": 1, "cheems": 1, "virgin": 1, "alpha": 1,
            "beta": 1, "gigachad": 1, "giga chad": 1, "nerd": 1, "geek": 1,
            "noob": 1, "pro": 1, "epic": 1, "legend": 1, "god": 1, "king": 1,
            "queen": 1, "goat": 1, "savage": 1, "fire": 1, "lit": 1, "dope": 1,
            "sick": 1, "insane": 1, "crazy": 1, "wild": 1, "mad": 1, "unreal": 1,
            "legendary": 1, "gg": 1, "wp": 1, "rekt": 1, "pwned": 1, "ez": 1,
            "easy": 1, "clap": 1, "applause": 1, "slow clap": 1, "facepalm": 1,
            "palm": 1, "smh": 1, "shaking my head": 1, "rip": 1, "f": 1,
            "press f": 1, "f in chat": 1, "rip bozo": 1, "bozo": 1, "clown": 1,
            "joker": 1, "circus": 1, "show": 1, "entertainment": 1, "bakchod": 1,
            "bakchodi": 1, "chutiyapa": 1, "chutiyagiri": 1, "mazaak": 1,
            "mazak": 1, "joke": 1, "mazaakiya": 1, "hasi": 1, "hansi": 1,
            "hass": 1, "hassa": 1, "thug life": 1, "thug": 1, "gigachad": 1,
            "chad": 1, "sigma male": 1, "sigma grindset": 1, "grind": 1,
            "hustle": 1, "cooked": 2, "bodied": 1, "destroyed": 1,
            "annihilated": 1, "obliterated": 1, "massacred": 1, "murdered": 1,
            "executed": 1, "slaughtered": 1, "genocide": 1, "exterminated": 1,
            "erased": 1, "deleted": 1, "cancelled": 1, "canceled": 1,
            "emoji heavy": 0,  # Detected separately
        },
    },
    "dismissive_minimizing": {
        "weight": 1,
        "keywords": {
            "not a big deal": 3, "no big deal": 3, "just study": 2, "just focus": 2,
            "stop complaining": 3, "stop whining": 3, "get over it": 3, "move on": 2,
            "it is normal": 2, "everyone goes through": 2, "happens to everyone": 2,
            "you are overreacting": 3, "overreacting": 2, "too sensitive": 2,
            "dramatic": 2, "attention seeking": 2, "itna kya ro raha hai": 3,
            "zada mat soch": 2, "common hai": 2, "sabke sath hota hai": 2,
            "koi nahi": 1, "itna bhi kya": 2, "zyada mat soch": 2, "chill kar": 1,
            "tension mat le": 1, "sab thik ho jayega": 1, "ho jayega": 1,
            "easy hai": 1, "simple hai": 1, "bas kar": 2, "kya fark padta hai": 2,
            "does not matter": 2, "who cares": 2, "so what": 2, "whatever": 1,
            "meh": 1, "not that serious": 2, "not serious": 2, "chill": 1,
            "relax": 1, "calm down": 1, "do not overthink": 1, "overthink": 1,
            "jada mat": 2, "itna bhi": 2, "koi baat nahi": 1, "no worries": 1,
            "it is fine": 1, "it is okay": 1, "you will be fine": 1,
            "it is not that bad": 2, "could be worse": 2, "at least": 1,
            "think positive": 1, "positive soch": 1, "be positive": 1,
            "positivity": 1, "optimistic": 1, "life goes on": 1, "time heals": 1,
            "move forward": 1, "forget it": 1, "let it go": 1,
            "do not worry about it": 1, "it is not important": 2, "insignificant": 2,
            "trivial": 1, "minor": 1, "small issue": 1, "not a problem": 1,
            "not an issue": 1, "you will survive": 1, "you will live": 1,
            "suck it up": 2, "deal with it": 2, "tough luck": 2, "life is hard": 1,
            "everyone has problems": 2, "others have it worse": 2, "be grateful": 1,
            "count your blessings": 1, "at least you": 1, "at least you have": 1,
            "at least you are": 1, "snap out of it": 2, "pull yourself together": 2,
            "toughen up": 2, "man up": 2, "grow up": 2, "act mature": 1,
            "not that deep": 2, "not deep": 1, "first world problem": 2,
            "first world problems": 2, "privileged": 1, "ungrateful": 1,
            "overdramatic": 1, "melodramatic": 1, "hypersensitive": 1,
            "crybaby": 1, "cry baby": 1, "snowflake": 1, "triggered": 1,
            "triggered much": 1, "fragile": 1, "weak": 1, "soft": 1,
            "softie": 1, "pussy": 1, "coward": 1, "spineless": 1,
        },
    },
    "toxic_abusive": {
        "weight": 1,
        "keywords": {
            "shut up": 3, "stfu": 3, "idiot": 2, "stupid": 2, "dumb": 2,
            "loser": 2, "worthless": 2, "useless": 2, "kill yourself": 5,
            "kys": 5, "fuck off": 2, "fuck you": 2, "fuck": 1, "shit": 1,
            "asshole": 2, "bitch": 2, "bastard": 2, "moron": 2, "retard": 2,
            "die": 1, "go die": 2, "chutiya": 2, "chutiye": 2, "bhosdike": 2,
            "madarchod": 2, "bc": 1, "mc": 1, "bkl": 1, "gandu": 2,
            "randi": 2, "saala": 1, "harami": 1, "kamine": 1, "nalayak": 1,
            "bewakoof": 1, "ullu": 1, "hate": 1, "garbage": 1, "trash": 1,
            "cringe": 1, "pathetic": 1, "disgusting": 1, "worst": 1,
            "terrible": 1, "awful": 1, "suck": 1, "sucks": 1, "hate you": 1,
            "get lost": 1, "fuck yourself": 2, "motherfucker": 2, "cunt": 2,
            "slut": 2, "whore": 2, "dick": 1, "pussy": 1, "cock": 1,
            "ass": 1, "piss off": 1, "piss": 1, "damn": 1, "hell": 1,
            "crap": 1, "bullshit": 1, "bs": 1, "nonsense": 1, "rubbish": 1,
            "waste": 1, "fail": 1, "failure": 1, "disappoint": 1,
            "disappointed": 1, "shame": 1, "shameful": 1, "embarrass": 1,
            "embarrassing": 1, "ridiculous": 1, "absurd": 1, "stupid": 2,
            "dumb": 2, "fool": 1, "foolish": 1, "idiotic": 1, "lame": 1,
            "weak": 1, "pathetic": 1, "hopeless": 1, "good for nothing": 2,
            "kuch nahi kar sakta": 2, "kuch nahi ukhad sakta": 2, "bekar hai": 1,
            "bekar": 1, "faltu": 1, "bakwaas": 1, "bakchodi": 1,
            "chutiyapa": 1, "chutiyagiri": 1, "bhosdiwala": 1, "bhosdike": 2,
            "madarjaat": 1, "maaki": 1, "bahenchod": 1, "bhenchod": 1,
            "chod": 1, "loda": 1, "lund": 1, "gand": 1, "gaand": 1,
            "chut": 1, "bhosdi": 1, "randi": 2, "randwe": 1, "chutmarike": 1,
            "jhantu": 1, "jhatu": 1, "katua": 1, "abuse": 1, "toxic": 1,
            "negativity": 1, "negative": 1, "hateful": 1, "harass": 1,
            "harassment": 1, "bully": 1, "bullying": 1, "insult": 1,
            "insulting": 1, "offend": 1, "offensive": 1, "aggressive": 1,
            "attack": 1, "threat": 1, "threaten": 1, "kill": 1, "murder": 1,
            "no one cares": 2, "nobody cares": 2, "no one likes you": 2,
            "nobody likes you": 2, "no one loves you": 2, "you are alone": 2,
            "deserve to die": 3, "should die": 3, "must die": 3, "die already": 3,
            "worthless piece": 2, "piece of shit": 2, "trash human": 2,
            "human garbage": 2, "disgusting human": 2, "disgusting person": 2,
            "pathetic excuse": 2, "excuse of a": 2, "waste of space": 2,
            "waste of oxygen": 2, "oxygen thief": 2, "breathing air": 1,
        },
    },
    "blame_criticism": {
        "weight": 1,
        "keywords": {
            "your fault": 2, "it is your fault": 2, "you did this": 2,
            "you caused": 2, "you brought this": 2, "you asked for it": 2,
            "you deserve it": 2, "karma": 1, "what goes around": 1,
            "comes around": 1, "actions have consequences": 2, "consequences": 1,
            "you had it coming": 2, "serves you right": 2, "served you right": 2,
            "told you so": 2, "i told you": 2, "i warned you": 2, "should have listened": 2,
            "should have studied": 2, "should have worked": 2, "lazy": 2,
            "procrastinated": 2, "procrastination": 1, "did not try": 2,
            "did not work hard": 2, "not hard working": 2, "lack of effort": 2,
            "no effort": 2, "half assed": 2, "half-assed": 2, "half ass": 2,
            "not serious": 2, "not taking it seriously": 2, "joking around": 1,
            "wasting time": 2, "time waste": 1, "timepass": 1, "time pass": 1,
            "distracted": 1, "distraction": 1, "phone addiction": 2, "social media": 1,
            "instagram": 1, "snapchat": 1, "tiktok": 1, "reels": 1, "youtube shorts": 1,
            "shorts": 1, "gaming": 1, "pubg": 1, "bgmi": 1, "free fire": 1,
            "fortnite": 1, "valorant": 1, "minecraft": 1, "genshin": 1,
            "not focused": 2, "lack focus": 2, "no focus": 2, "no discipline": 2,
            "lack discipline": 2, "no dedication": 2, "lack dedication": 2,
            "not dedicated": 2, "not committed": 2, "no commitment": 2,
            "uncommitted": 1, "unmotivated": 1, "not motivated": 1, "no motivation": 1,
            "weak mindset": 2, "weak mentality": 2, "loser mentality": 2,
            "loser mindset": 2, "excuse": 1, "excuses": 1, "making excuses": 2,
            "full of excuses": 2, "always excuses": 2, "victim mentality": 2,
            "playing victim": 2, "victim card": 2, "pity party": 1,
            "self pity": 1, "self-pity": 1, "pity yourself": 1, "stop feeling sorry": 2,
            "stop feeling bad": 2, "stop whining": 2, "stop crying": 2, "man up": 2,
            "grow up": 2, "be a man": 2, "act like an adult": 2, "immature": 1,
            "childish": 1, "baby": 1, "crybaby": 1, "drama": 1, "drama queen": 1,
            "drama king": 1, "attention seeker": 1, "attention whore": 2,
            "validation seeker": 1, "too much drama": 1, "overreacting": 1,
            "over reacting": 1, "over-reacting": 1, "creating drama": 1,
        },
    },
    "self_disclosure": {
        "weight": 1,
        "keywords": {
            "i feel": 2, "i am feeling": 2, "i am going through": 2, "i am struggling": 2,
            "i struggle with": 2, "i have": 1, "i am dealing with": 2, "dealing with": 1,
            "i suffer from": 2, "i suffer": 1, "i have been diagnosed": 2,
            "diagnosed with": 1, "my doctor": 1, "my therapist": 1, "my counselor": 1,
            "my psychiatrist": 1, "my psychologist": 1, "medication": 1, "meds": 1,
            "i take": 1, "i am on": 1, "i am taking": 1, "antidepressant": 1,
            "anti depressant": 1, "anti-depressant": 1, "anxiety med": 1,
            "anxiety medication": 1, "sleeping pill": 1, "sleeping pills": 1,
            "therapy session": 1, "therapy helped": 1, "therapy is helping": 1,
            "i see a therapist": 2, "i go to therapy": 2, "i am in therapy": 2,
            "i started therapy": 2, "i am seeing": 1, "i am attending": 1,
            "i have anxiety": 2, "i have depression": 2, "i have ocd": 2,
            "i have adhd": 2, "i have bipolar": 2, "i have ptsd": 2,
            "i have panic": 1, "i have panic attacks": 2, "panic attack": 1,
            "panic attacks": 1, "anxiety attack": 1, "anxiety attacks": 1,
            "i feel empty": 2, "i feel numb": 2, "i feel nothing": 2,
            "i feel lost": 2, "i feel broken": 2, "i feel trapped": 2,
            "i feel stuck": 2, "i feel hopeless": 2, "i feel helpless": 2,
            "i feel worthless": 2, "i feel useless": 2, "i feel pathetic": 2,
            "i feel like a failure": 2, "i feel like garbage": 2, "i feel like trash": 2,
            "i feel like dying": 3, "i feel like ending": 2, "i feel like giving up": 2,
            "i feel like quitting": 2, "i feel like i": 1, "i feel like a burden": 2,
            "burden": 1, "i am a burden": 2, "i am burden": 1, "i am a waste": 2,
            "i am waste": 1, "i am garbage": 2, "i am trash": 2, "i am pathetic": 2,
            "i am useless": 2, "i am worthless": 2, "i am a failure": 2, "i am failure": 1,
            "i am depressed": 2, "i am sad": 1, "i am unhappy": 1, "i am miserable": 1,
            "i am lonely": 1, "i am alone": 1, "i am scared": 1, "i am afraid": 1,
            "i am worried": 1, "i am anxious": 1, "i am stressed": 1, "i am tired": 1,
            "i am exhausted": 1, "i am burnt out": 2, "i am burned out": 2,
            "i am overwhelmed": 1, "i am overthinking": 1, "i am insecure": 1,
            "i am ugly": 1, "i am fat": 1, "i am skinny": 1, "i am short": 1,
            "i am tall": 1, "i am not good enough": 2, "not good enough": 1,
            "i am not enough": 2, "never enough": 1, "i hate myself": 2,
            "i hate my life": 2, "i hate everything": 2, "i hate my": 1,
            "i hate the way": 1, "i do not like myself": 2, "dislike myself": 1,
            "i can not stand myself": 2, "i disgust myself": 2, "i am disgusted": 1,
            "i am ashamed": 1, "i am embarrassed": 1, "i am guilty": 1, "i feel guilty": 1,
            "i regret": 1, "i regret everything": 2, "i wish i": 1, "i wish i was": 1,
            "i wish i could": 1, "if only i": 1, "what if i": 1, "sometimes i": 1,
            "often i": 1, "lately i": 1, "recently i": 1, "these days i": 1,
        },
    },
    "crisis_escalation": {
        "weight": 1,
        "keywords": {
            "suicide": 3, "suicidal": 3, "kill myself": 3, "end my life": 3,
            "self harm": 3, "self-harm": 3, "cutting": 2, "overdose": 2,
            "want to die": 3, "better off dead": 3, "not worth living": 3,
            "life not worth": 3, "end it all": 3, "can not go on": 2,
            "cannot go on": 2, "give up on life": 3, "no reason to live": 3,
            "mar jana hai": 3, "marne ka mann hai": 3, "mar jau": 3, "mar jaunga": 3,
            "mar jaungi": 3, "zindagi se tang aagaya": 3, "tang aagaya": 2,
            "jeene ka mann nahi": 3, "jee ka mann nahi": 3, "jee nahi rha": 3,
            "jee nahi raha": 3, "jee nahi chahiye": 3, "ab nahi rha jaata": 3,
            "ab nahi raha jaata": 3, "sab khatam kar du": 3, "khatam kar du": 2,
            "bas ab nahi": 3, "bas nahi ho sakta": 3, "ab khatam karo": 3,
            "khatam karo": 2, "rope": 2, "hang myself": 3, "jump off": 2,
            "jump from": 2, "bridge": 1, "building": 1, "tall building": 1,
            "poison": 1, "pills": 1, "blade": 1, "razor": 1, "wrist": 1,
            "wrists": 1, "slit": 1, "slit my": 2, "slit wrist": 2, "bleed out": 2,
            "bleeding out": 2, "no point": 2, "what is the point": 2,
            "what is the point of living": 3, "point of living": 2, "no purpose": 2,
            "no meaning": 2, "meaningless": 1, "empty inside": 2, "hollow": 1,
            "void": 1, "darkness": 1, "dark": 1, "dark place": 1, "abyss": 1,
            "drowning": 1, "suffocating": 1, "suffocate": 1, "can not breathe": 2,
            "cannot breathe": 2, "trapped": 1, "no escape": 2, "no way out": 2,
            "cornered": 1, "backed into a corner": 2, "last resort": 1, "final option": 1,
            "only option": 1, "only way out": 2, "only escape": 1, "only way": 1,
            "goodbye": 1, "bye": 1, "farewell": 1, "last post": 1, "final post": 1,
            "last message": 1, "final message": 1, "note": 1, "suicide note": 3,
            "last words": 2, "final words": 2, "i am done": 2, "i am finished": 2,
            "i am ending": 2, "i am leaving": 1, "i am going": 1, "i am gone": 2,
            "you will not see me": 2, "will not see me again": 2, "no one will miss me": 2,
            "no one will care": 2, "nobody will miss me": 2, "nobody will care": 2,
            "world will be better": 2, "better off without me": 2, "without me": 1,
            "burden to everyone": 2, "burden to my family": 2, "burden to my parents": 2,
            "burden to friends": 2, "everyone will be happier": 2, "happier without me": 2,
            "please remember me": 1, "remember me": 1, "tell my family": 1,
            "tell my parents": 1, "tell my friends": 1, "sorry for everything": 2,
            "i am sorry": 1, "forgive me": 1, "i did not mean": 1, "it is not your fault": 1,
            "not your fault i": 1, "do not blame yourself": 1, "not because of you": 1,
            "i love you all": 1, "love you all": 1, "love you mom": 1, "love you dad": 1,
            "goodbye everyone": 2, "goodbye world": 2, "goodbye forever": 2,
            "forever": 1, "eternal": 1, "peace at last": 1, "finally at peace": 1,
            "finally free": 1, "free at last": 1, "no more pain": 2, "no more suffering": 2,
            "no more": 1, "end of pain": 1, "end of suffering": 1, "release": 1,
            "relief from pain": 1, "relief from suffering": 1, "escape": 1,
            "only way to escape": 2, "escape the pain": 1, "escape this": 1,
        },
    },
}

# Compile regex patterns for efficiency
COMPILED_PATTERNS = {}
for cat, data in TAXONOMY.items():
    if "keywords" in data and data["keywords"]:
        patterns = []
        for word, weight in data["keywords"].items():
            escaped = re.escape(word.lower())
            if len(word) > 3 or ' ' in word:
                patterns.append((re.compile(r'\b' + escaped + r'\b'), weight))
            else:
                patterns.append((re.compile(r'\b' + escaped + r'\b'), weight))
        COMPILED_PATTERNS[cat] = patterns
    else:
        COMPILED_PATTERNS[cat] = []

# Emoji detection for humor
HUMOR_EMOJIS = set([
    "😂", "🤣", "😭", "💀", "☠️", "🤡", "🤓", "🤪", "🥴", "😹", "😆", "😄", "😁",
    "😎", "🤩", "😳", "😵", "😵‍💫", "🤯", "🤮", "🤢", "🤥", "🤭", "🤨", "😐",
    "😑", "😬", "🙄", "😯", "😦", "😧", "😮", "😲", "🥱", "😴", "🤤", "😪",
    "😺", "😸", "😹", "😻", "😼", "😽", "🙀", "😿", "😾", "🤖", "👽", "👾",
    "🎃", "💩", "👻", "👹", "👺", "👿", "😈", "🤠", "🥸", "🤡",
])

# AutoModerator / bot detection
BOT_PATTERNS = re.compile(r'auto\s*moderator|i am a bot|this action was performed automatically|contact the moderators|report it using', re.I)

def classify_comment(text, author, parent_distress, score):
    """Classify a comment into 12 interaction categories with scores."""
    if not isinstance(text, str) or len(text.strip()) < 3:
        return {cat: 0 for cat in TAXONOMY}, {"primary": "neutral_discussion", "secondary": None, "confidence": 1.0}
    
    text_lower = text.lower().strip()
    word_count = len(text_lower.split())
    
    # Bot/AutoMod -> neutral
    if author and author.lower() == "automoderator":
        return {cat: 0 for cat in TAXONOMY}, {"primary": "neutral_discussion", "secondary": None, "confidence": 1.0}
    if BOT_PATTERNS.search(text_lower):
        return {cat: 0 for cat in TAXONOMY}, {"primary": "neutral_discussion", "secondary": None, "confidence": 1.0}
    
    # Very short comments with humor markers
    if word_count <= 2:
        if any(e in text for e in HUMOR_EMOJIS) or text_lower in {"lol", "lmao", "rofl", "haha", "hehe", "xd", "bruh", "rip", "f"}:
            scores = {cat: 0 for cat in TAXONOMY}
            scores["humor_meme_coping"] = 3.0
            return scores, {"primary": "humor_meme_coping", "secondary": None, "confidence": 1.0}
        return {cat: 0 for cat in TAXONOMY}, {"primary": "neutral_discussion", "secondary": None, "confidence": 1.0}
    
    scores = {cat: 0.0 for cat in TAXONOMY}
    
    # Keyword scoring
    for cat, patterns in COMPILED_PATTERNS.items():
        for pattern, weight in patterns:
            matches = pattern.findall(text_lower)
            scores[cat] += len(matches) * weight
    
    # Emoji analysis for humor
    humor_emojis = sum(1 for c in text if c in HUMOR_EMOJIS)
    if humor_emojis >= 2:
        scores["humor_meme_coping"] += humor_emojis * 1.5
    if humor_emojis >= 3:
        scores["humor_meme_coping"] += 3.0
    
    # Question detection for neutral
    if '?' in text and max(scores.values()) < 2:
        scores["neutral_discussion"] += 1.0
    
    # Score-based modifiers
    if score is not None and score < -1:
        scores["toxic_abusive"] += 1.0
        scores["blame_criticism"] += 0.5
    
    if score is not None and score > 10 and scores["advice_guidance"] > 0:
        scores["advice_guidance"] += 1.0
    
    # Very long comments with advice -> boost advice
    if word_count > 80 and scores["advice_guidance"] > 0:
        scores["advice_guidance"] += 2.0
    
    # Very short toxic comments
    if word_count <= 5 and scores["toxic_abusive"] > 0:
        scores["toxic_abusive"] += 1.0
    
    # Personal experience markers
    if text_lower.startswith("i ") or text_lower.startswith("mujhe ") or text_lower.startswith("meri ") or text_lower.startswith("mere "):
        if scores["personal_experience"] > 0 or scores["self_disclosure"] > 0:
            scores["personal_experience"] += 0.5
            scores["self_disclosure"] += 0.5
    
    # Crisis escalation context
    if parent_distress in {"suicidal", "depression_anxiety", "academic_burnout"}:
        if scores["crisis_escalation"] > 0:
            scores["crisis_escalation"] += 0.5
    
    # Determine primary and secondary categories
    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_cats[0][0] if sorted_cats[0][1] > 0 else "neutral_discussion"
    secondary = sorted_cats[1][0] if len(sorted_cats) > 1 and sorted_cats[1][1] > 0 else None
    
    # Confidence: primary score / (max possible or total)
    total_score = sum(scores.values()) + 0.01
    confidence = scores[primary] / total_score if primary in scores else 0.0
    
    return scores, {"primary": primary, "secondary": secondary, "confidence": round(confidence, 3)}


def main():
    print("=" * 60)
    print("COMMENTS INTERACTION TAXONOMY CLASSIFIER")
    print("=" * 60)
    
    df = pd.read_csv(INPUT)
    print(f"\n[1] Loaded {len(df):,} preprocessed comments")
    
    # Classify all comments
    print(f"[2] Classifying into 12 categories...")
    all_scores = []
    all_labels = []
    
    for idx, row in df.iterrows():
        if idx % 20000 == 0:
            print(f"    Progress: {idx:,} / {len(df):,} ({idx/len(df)*100:.1f}%)")
        
        scores, labels = classify_comment(
            row.get('clean_body', ''),
            row.get('author', ''),
            row.get('distress_type', ''),
            row.get('score', 0)
        )
        all_scores.append(scores)
        all_labels.append(labels)
    
    # Add score columns
    for cat in TAXONOMY:
        df[f'score_{cat}'] = [s[cat] for s in all_scores]
    
    df['primary_category'] = [l['primary'] for l in all_labels]
    df['secondary_category'] = [l['secondary'] for l in all_labels]
    df['confidence'] = [l['confidence'] for l in all_labels]
    
    # Save
    out_csv = os.path.join(OUTPUT_DIR, "comments_taxonomy.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[3] Saved to {out_csv}")
    
    # ── Distribution Report ──────────────────────────────────────────
    print(f"\n[4] Generating distribution report...")
    
    dist = df['primary_category'].value_counts()
    total = len(df)
    
    report_lines = [
        "# Comments Interaction Taxonomy Report",
        "",
        "## Overview",
        f"- **Total comments analyzed**: {total:,}",
        f"- **Taxonomy categories**: 12 (multi-label heuristic classifier)",
        f"- **Primary categories**: Support/Empathy, Advice/Guidance, Personal Experience, Encouragement, Information, Neutral, Humor, Dismissive, Toxic, Blame, Self-Disclosure, Crisis Escalation",
        "",
        "## Overall Distribution",
        "| Category | Count | Percentage |",
        "|----------|-------|------------|",
    ]
    for cat in dist.index:
        count = dist[cat]
        pct = count / total * 100
        report_lines.append(f"| {cat} | {count:,} | {pct:.1f}% |")
    
    # Per-distress-type breakdown
    report_lines.extend(["", "## Per-Distress-Type Response Patterns"])
    
    distress_dist = df.groupby('distress_type')['primary_category'].value_counts().unstack(fill_value=0)
    for distress in distress_dist.index:
        report_lines.append(f"\n### {distress} ({distress_dist.loc[distress].sum():,} comments)")
        row = distress_dist.loc[distress]
        d_total = row.sum()
        # Sort by count descending
        for cat in row.sort_values(ascending=False).index:
            count = row[cat]
            pct = count / d_total * 100 if d_total > 0 else 0
            report_lines.append(f"- {cat}: {count:,} ({pct:.1f}%)")
    
    # Key findings
    report_lines.extend(["", "## Key Findings"])
    
    # Most common response
    most_common = dist.index[0]
    report_lines.append(f"1. **Most common response type**: {most_common} ({dist.iloc[0]/total*100:.1f}%)")
    
    # Support vs toxic ratio
    support_count = dist.get('support_empathy', 0) + dist.get('encouragement_motivation', 0)
    toxic_count = dist.get('toxic_abusive', 0) + dist.get('blame_criticism', 0) + dist.get('dismissive_minimizing', 0)
    if toxic_count > 0:
        ratio = support_count / toxic_count
        report_lines.append(f"2. **Support-to-Harm ratio**: {ratio:.2f}:1 ({support_count:,} supportive vs {toxic_count:,} harmful)")
    
    # Crisis posts
    crisis_df = df[df['distress_type'] == 'suicidal']
    if len(crisis_df) > 0:
        support_pct = (crisis_df['primary_category'] == 'support_empathy').sum() / len(crisis_df) * 100
        advice_pct = (crisis_df['primary_category'] == 'advice_guidance').sum() / len(crisis_df) * 100
        crisis_esc_pct = (crisis_df['primary_category'] == 'crisis_escalation').sum() / len(crisis_df) * 100
        self_dis_pct = (crisis_df['primary_category'] == 'self_disclosure').sum() / len(crisis_df) * 100
        report_lines.append(f"3. **Suicidal post responses**: {support_pct:.1f}% Support, {advice_pct:.1f}% Advice, {crisis_esc_pct:.1f}% Crisis Escalation, {self_dis_pct:.1f}% Self-Disclosure")
    
    # Depression/anxiety posts
    dep_df = df[df['distress_type'] == 'depression_anxiety']
    if len(dep_df) > 0:
        exp_pct = (dep_df['primary_category'] == 'personal_experience').sum() / len(dep_df) * 100
        self_dis_pct = (dep_df['primary_category'] == 'self_disclosure').sum() / len(dep_df) * 100
        report_lines.append(f"4. **Depression/anxiety post responses**: {exp_pct:.1f}% Personal Experience, {self_dis_pct:.1f}% Self-Disclosure")
    
    # Meme posts
    meme_df = df[df['distress_type'] == 'meme_distress']
    if len(meme_df) > 0:
        humor_pct = (meme_df['primary_category'] == 'humor_meme_coping').sum() / len(meme_df) * 100
        report_lines.append(f"5. **Meme distress post responses**: {humor_pct:.1f}% Humor/Meme Coping")
    
    # Exam stress posts
    exam_df = df[df['distress_type'] == 'exam_stress']
    if len(exam_df) > 0:
        advice_pct = (exam_df['primary_category'] == 'advice_guidance').sum() / len(exam_df) * 100
        info_pct = (exam_df['primary_category'] == 'information_resources').sum() / len(exam_df) * 100
        report_lines.append(f"6. **Exam stress post responses**: {advice_pct:.1f}% Advice, {info_pct:.1f}% Information/Resources")
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, "taxonomy_report.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"    Saved report to {report_path}")
    
    # Save distribution CSV
    dist_df = pd.DataFrame({'category': dist.index, 'count': dist.values, 'percentage': dist.values / total * 100})
    dist_df.to_csv(os.path.join(OUTPUT_DIR, "taxonomy_distribution.csv"), index=False)
    
    # Save per-distress CSV
    distress_dist.to_csv(os.path.join(OUTPUT_DIR, "taxonomy_per_distress.csv"))
    
    # Print summary
    print(f"\n{'='*60}")
    print("CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    for cat in dist.index:
        count = dist[cat]
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {cat:30} | {count:6,} ({pct:5.1f}%) | {bar}")
    print(f"{'='*60}")
    print(f"Saved all outputs to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
