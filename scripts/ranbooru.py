from io import BytesIO
import html
import re
import random
import requests
import copy
import dataclasses
from urllib.parse import quote_plus, unquote_plus
import modules.scripts as scripts
import gradio as gr
import os
from PIL import Image
import numpy as np
import requests_cache
import json
import tempfile
import threading
import time
from functools import wraps

from modules.processing import (
    process_images,
    StableDiffusionProcessingImg2Img
)

from modules import shared
from modules.ui_components import InputAccordion

extension_root = scripts.basedir()

user_data_dir = os.path.join(extension_root, 'user')
user_credentials_dir = os.path.join(user_data_dir, 'credentials')

os.makedirs(user_credentials_dir, exist_ok=True)

# Initialize credentials manager
class CredentialsManager:
    def __init__(self, extension_root):
        self.extension_root = extension_root
        self.credentials_dir = os.path.join(
            extension_root,
            'user',
            'credentials'
        )

        self.credentials_file = os.path.join(
            self.credentials_dir,
            'credentials.json'
        )

        # Create credentials directory if it doesn't exist
        os.makedirs(self.credentials_dir, exist_ok=True)

        # Initialize credentials file if it doesn't exist
        if not os.path.exists(self.credentials_file):
            self._save_credentials({})

    def _load_credentials(self):
        """Load credentials from the JSON file"""
        try:
            with open(self.credentials_file, 'r') as f:
                return json.load(f)

        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_credentials(self, credentials):
        """Save credentials to the JSON file"""
        with open(self.credentials_file, 'w') as f:
            json.dump(credentials, f, indent=2)

    def save_booru_credentials(
        self,
        booru_name,
        api_key,
        user_id=None
    ):
        """Save API credentials for a specific booru"""

        credentials = self._load_credentials()

        if booru_name not in credentials:
            credentials[booru_name] = {}

        credentials[booru_name]['api_key'] = api_key

        if user_id is not None:
            credentials[booru_name]['user_id'] = user_id

        self._save_credentials(credentials)

    def get_booru_credentials(self, booru_name):
        """Get API credentials for a specific booru"""
        credentials = self._load_credentials()
        return credentials.get(booru_name, {})

    def has_credentials(self, booru_name):
        """Check if credentials exist for a specific booru"""

        credentials = self.get_booru_credentials(booru_name)

        return (
            'api_key' in credentials
            and credentials['api_key'].strip() != ''
        )

    def clear_booru_credentials(self, booru_name):
        """Clear credentials for a specific booru"""

        credentials = self._load_credentials()

        if booru_name in credentials:
            del credentials[booru_name]

        self._save_credentials(credentials)


credentials_manager = CredentialsManager(extension_root)

# Persist user-facing settings only when generating with Ranbooru enabled.
# Gelbooru API credentials are managed separately via the opt-in credentials manager.
RANBOORU_STATE_FILE = os.path.join(user_data_dir, 'last_settings.json')
RANBOORU_STATE_LOCK = threading.RLock()
RANBOORU_STATE_FIELDS = (
    'enabled',
    'tags',
    'booru',
    'remove_bad_tags',
    'max_pages',
    'change_dash',
    'same_prompt',
    'fringe_benefits',
    'remove_tags',
    'use_img2img',
    'denoising',
    'use_last_img',
    'change_background',
    'change_color',
    'shuffle_tags',
    'post_id',
    'limit_tags',
    'max_tags',
    'sorting_order',
    'mature_rating',
    'use_ip',
    'crop_center',
    'use_cache',
    'only_poses',
    'strip_clothing',
    'names_all_boorus',
    'skip_prompt_tags',
    'save_credentials',
)


def _read_ranbooru_state():
    """Read the state file. Caller must hold RANBOORU_STATE_LOCK."""
    try:
        with open(RANBOORU_STATE_FILE, 'r', encoding='utf-8') as state_file:
            data = json.load(state_file)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        print(f'[Ranbooru] Unable to load last settings: {exc}')
        return {}


def load_ranbooru_settings(is_img2img):
    """Return the last generated settings for this WebUI tab."""
    with RANBOORU_STATE_LOCK:
        data = _read_ranbooru_state()
        tabs = data.get('tabs', {})
        if not isinstance(tabs, dict):
            return {}
        tab_settings = tabs.get('img2img' if is_img2img else 'txt2img', {})
        return tab_settings if isinstance(tab_settings, dict) else {}


def save_ranbooru_settings(is_img2img, generation_values):
    """Atomically save a snapshot without interrupting image generation on I/O errors."""
    settings = {
        name: generation_values[name]
        for name in RANBOORU_STATE_FIELDS
        if name in generation_values
    }

    with RANBOORU_STATE_LOCK:
        temp_path = None
        try:
            data = _read_ranbooru_state()
            tabs = data.get('tabs', {})
            if not isinstance(tabs, dict):
                tabs = {}
            tabs['img2img' if is_img2img else 'txt2img'] = settings
            data = {'version': 1, 'tabs': tabs}

            # Temporary file in the same directory allows an atomic rename.
            fd, temp_path = tempfile.mkstemp(
                prefix='.ranbooru_state_', suffix='.tmp', dir=user_data_dir
            )
            with os.fdopen(fd, 'w', encoding='utf-8') as state_file:
                json.dump(data, state_file, indent=2, ensure_ascii=False)
                state_file.write('\n')
            os.replace(temp_path, RANBOORU_STATE_FILE)
            print(f'[Ranbooru] Saved {len(settings)} settings for '
                  f"{'img2img' if is_img2img else 'txt2img'} to {RANBOORU_STATE_FILE}")
        except (OSError, TypeError, ValueError) as exc:
            print(f'[Ranbooru] Unable to save last settings: {exc}')
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


# Most-recently-used booru order shared by txt2img and img2img.
# The old v6/v7 manually sorted booru_order.json is intentionally ignored.
# Only generating with Ranbooru enabled counts as using a booru.
BOORU_RECENT_FILE = os.path.join(user_data_dir, 'booru_recent.json')
BOORU_RECENT_LOCK = threading.RLock()
BOORU_DEFAULT_ORDER = (
    'safebooru', 'rule34', 'danbooru', 'gelbooru', 'konachan',
    'yande.re', 'aibooru', 'xbooru', 'e621'
)


def _read_booru_recent_unlocked():
    """Return valid recent sites, newest first; ignore unknown/duplicate entries."""
    try:
        with open(BOORU_RECENT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        saved = data.get('recent', []) if isinstance(data, dict) else []
    except FileNotFoundError:
        saved = []
    except (OSError, ValueError) as exc:
        print(f'[Ranbooru] Unable to read recent booru list: {exc}')
        saved = []

    if not isinstance(saved, list):
        saved = []
    seen = set()
    recent = []
    for name in saved:
        if isinstance(name, str) and name in BOORU_DEFAULT_ORDER and name not in seen:
            seen.add(name)
            recent.append(name)
    return recent


def load_booru_order():
    """Show recently generated boorus first, then the original default order."""
    with BOORU_RECENT_LOCK:
        recent = _read_booru_recent_unlocked()
    return recent + [name for name in BOORU_DEFAULT_ORDER if name not in recent]


def mark_booru_used(booru):
    """Record the chosen site once per Generate, never on dropdown browsing."""
    if booru not in BOORU_DEFAULT_ORDER:
        return
    with BOORU_RECENT_LOCK:
        recent = _read_booru_recent_unlocked()
        recent = [booru] + [name for name in recent if name != booru]
        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix='.booru_recent_', suffix='.tmp', dir=user_data_dir
            )
            with os.fdopen(fd, 'w', encoding='utf-8') as out:
                json.dump({'version': 1, 'recent': recent}, out, indent=2)
                out.write('\n')
            os.replace(temp_path, BOORU_RECENT_FILE)
        except OSError as exc:
            print(f'[Ranbooru] Unable to save recent booru list: {exc}')
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


def show_ranbooru_progress(message):
    """Use Forge Neo's /internal/progress textinfo -> .progressDiv path."""
    try:
        state = getattr(shared, 'state', None)
        if state is not None:
            state.textinfo = f'Ranbooru: {message}'
    except (AttributeError, RuntimeError):
        # Console logging/generation should still work with alternate WebUI builds.
        pass

def clear_ranbooru_progress():
    """Release our temporary status before Forge Neo starts sampling.

    Only clear textinfo if it still contains Ranbooru's message: another
    extension may have published a newer status during preprocessing.
    """
    try:
        state = getattr(shared, 'state', None)
        if state is not None:
            textinfo = getattr(state, 'textinfo', None)
            if isinstance(textinfo, str) and textinfo.startswith('Ranbooru: '):
                state.textinfo = None
    except (AttributeError, RuntimeError):
        pass


def clear_ranbooru_progress_on_error(callback):
    """Keep the search result visible during sampling, but clear on failure."""
    @wraps(callback)
    def wrapped(*args, **kwargs):
        try:
            return callback(*args, **kwargs)
        except Exception:
            clear_ranbooru_progress()
            raise
    return wrapped


def clear_ranbooru_progress_after_postprocess(callback):
    """Release Ranbooru's status when image generation has finished."""
    @wraps(callback)
    def wrapped(*args, **kwargs):
        try:
            return callback(*args, **kwargs)
        finally:
            clear_ranbooru_progress()
    return wrapped


COLORED_BG = [
    'black_background',
    'aqua_background',
    'white_background',
    'colored_background',
    'gray_background',
    'blue_background',
    'green_background',
    'red_background',
    'brown_background',
    'purple_background',
    'yellow_background',
    'orange_background',
    'pink_background',
    'plain',
    'transparent_background',
    'simple_background',
    'two-tone_background',
    'grey_background'
]

ADD_BG = [
    'outdoors',
    'indoors'
]

BW_BG = [
    'monochrome',
    'greyscale',
    'grayscale'
]

# --- Only Poses mode ---------------------------------------------------------
# The main lists come from Danbooru's tag group wiki pages, pre-generated into
# pose_tag_groups.json by tools/build_pose_tag_groups.py (never fetched at
# runtime). Tags in the 'keep' set (posture, gestures, expressions, framing,
# character count) are never stripped. Colour/creature combinations the wiki
# does not list one by one (white_shirt, cat_tail) are matched structurally:
# a known noun preceded only by known modifier words. Substring wildcards are
# avoided: '*_eyes' would also remove closed_eyes, which this mode must keep.
POSE_COLOR_WORDS = frozenset((
    'aqua', 'black', 'blonde', 'blue', 'brown', 'green', 'grey', 'gray',
    'orange', 'pink', 'purple', 'red', 'silver', 'white', 'yellow', 'gold',
    'golden', 'beige', 'teal', 'cyan', 'magenta', 'violet', 'lavender',
    'light', 'dark', 'pale', 'platinum', 'strawberry', 'ash', 'rainbow',
    'multicolored', 'two-tone', 'gradient', 'streaked', 'split-color',
    'colored', 'tan', 'olive', 'maroon', 'navy', 'crimson',
))

# Fallback for appearance: colour + one of these nouns (red_eyes, blue_skin).
POSE_COLOR_TRAIT_NOUNS = frozenset((
    'hair', 'eyes', 'skin', 'pupils', 'sclera', 'eyelashes', 'eyebrows',
    'streaks',
    # e621 body colours: blue_fur, white_body, green_scales.
    'body', 'fur', 'scales', 'feathers',
))

POSE_CREATURE_WORDS = frozenset((
    'animal', 'cat', 'dog', 'fox', 'wolf', 'rabbit', 'bunny', 'mouse',
    'horse', 'cow', 'sheep', 'goat', 'tiger', 'lion', 'bear', 'raccoon',
    'squirrel', 'monkey', 'pig', 'deer', 'dragon', 'demon', 'devil',
    'angel', 'bat', 'bird', 'feathered', 'fairy', 'insect', 'butterfly',
    'fish', 'shark', 'snake', 'lizard', 'oni', 'kitsune', 'monster',
    'mechanical', 'fake', 'extra', 'multiple', 'small', 'large', 'curled',
    'single', 'eastern', 'western',
))

# Fallback for creature parts: creature/colour + noun (cat_tail, black_horns).
POSE_CREATURE_TRAIT_NOUNS = frozenset((
    'ears', 'tail', 'tails', 'horns', 'horn', 'wings', 'halo', 'girl', 'boy',
))

# e621-only fallback: colour/pattern + body noun (tan_pawpads, striped_tail).
POSE_E621_BODY_NOUNS = frozenset((
    'pawpads', 'claws', 'markings', 'stripes', 'spots', 'scales',
    'countershading', 'inner_ear', 'eyelids', 'mustache', 'facial_hair',
    'beard', 'face', 'head', 'ears', 'tail', 'nose', 'muzzle', 'fur', 'body',
    'hair', 'horns', 'antlers', 'wings', 'feathers', 'hands', 'feet', 'paws',
    'hooves', 'mane', 'beak', 'tongue', 'sclera', 'pupils', 'eyes', 'chest',
))
POSE_E621_BODY_MODIFIERS = POSE_COLOR_WORDS | frozenset((
    'two', 'tone', 'multi', 'striped', 'spotted', 'hand', 'foot', 'tan',
    'tricolor', 'big', 'long', 'short', 'small', 'fluffy', 'scut', 'ghost',
    'dipstick', 'curled', 'prehensile', 'spiked', 'ventral',
))
POSE_E621_BODY_PREFIXES = ('countershade_',)
POSE_E621_BODY_SUFFIXES = ('_tuft', '_tufts', '_markings', '_marking', '_pawpads')

# e621-only clothing fallback: colour/pattern + e621 garment noun
# (denim_bottomwear, pattern_clothing). The bare nouns stay, like on Danbooru.
POSE_E621_CLOTHING_NOUNS = frozenset((
    'topwear', 'bottomwear', 'footwear', 'handwear', 'headwear', 'legwear',
    'armwear', 'swimwear', 'underwear', 'neckwear', 'eyewear', 'clothing',
    'outfit', 'accessory', 'piercing', 'piercings', 'thigh_highs',
    'thigh_socks', 'hotpants', 'bodysuit', 'collar', 'jewelry',
))
POSE_E621_CLOTHING_MODIFIERS = frozenset((
    'denim', 'pattern', 'camo', 'checkered', 'plaid', 'spotted', 'striped',
    'baggy', 'leather', 'latex', 'rubber', 'spiked', 'studded', 'metal',
    'gold', 'golden', 'silver', 'fishnet', 'lace', 'frilly', 'knitted',
    'two-piece', 'one-piece',
))
# States, not garments: open_clothing / torn_topwear stay, like Danbooru's
# open_clothes and torn_clothes.
POSE_E621_CLOTHING_STATES = frozenset((
    'open', 'torn', 'unbuttoned', 'wet', 'transparent', 'shiny', 'loose',
    'oversized', 'tight', 'cropped',
))

POSE_CLOTHING_MODIFIERS = POSE_COLOR_WORDS | frozenset((
    'long', 'short', 'mini', 'micro', 'pencil', 'pleated', 'frilled',
    'striped', 'vertical-striped', 'plaid', 'checkered', 'polka', 'dot',
    'print', 'floral', 'lace', 'leather', 'latex', 'denim', 'fur',
    'see-through', 'sheer', 'sleeveless', 'strapless', 'backless',
    'halterneck', 'off-shoulder', 'detached', 'wide', 'puffy', 'juliet',
    'open', 'unbuttoned', 'torn', 'cropped', 'high-waist', 'side-tie',
    'string', 'competition', 'one-piece', 'school', 'sailor', 'maid',
    'gothic', 'military', 'santa', 'witch', 'baseball', 'peaked', 'top',
    'straw', 'cowboy', 'hair', 'thigh', 'knee', 'ankle', 'elbow',
    'fingerless', 'bridal', 'wedding', 'china', 'chinese', 'japanese',
    'single', 'mismatched', 'hooded', 'neck', 'mob', 'wrist', 'x', 'star',
    'heart', 'gym', 'sports', 'naval', 'nurse', 'police', 'track',
    'turtleneck', 'sweater', 'bow', 'ribbon', 'animal', 'rabbit', 'fake',
    'round', 'high', 'low', 'thick', 'arm', 'leg', 'garter', 'tight',
    'loose', 'oversized', 'collared', 'buttoned', 'shiny', 'wet',
    'transparent', 'frilled', 'winter', 'summer', 'asymmetrical', 'waist',
    'highleg', 'front-tie', 'multiple',
))

# Generic fallbacks for boorus whose posts carry no tag categories.
POSE_QUALIFIER_SUFFIXES = ('_(artist)', '_(cosplay)')

# Booru post fields listing the artist, character and copyright tags.
POSE_CATEGORY_FIELDS = (
    'tag_string_artist', 'tag_string_character', 'tag_string_copyright'
)

POSE_EMPTY_GROUPS = {
    'traits': frozenset(), 'clothing': frozenset(), 'keep': frozenset(),
    'clothing_nouns': frozenset(), 'e621_traits': frozenset(),
    'e621_keep': frozenset(), 'e621_clothing': frozenset(),
}


def load_pose_tag_groups(path):
    """Read the generated tag lists; an unreadable file leaves only the fallback rules."""
    try:
        with open(path, 'r', encoding='utf-8') as groups_file:
            data = json.load(groups_file)
        groups = {
            name: frozenset(
                tag for tag in data.get(name, []) if isinstance(tag, str)
            )
            for name in ('traits', 'clothing', 'keep', 'e621_traits', 'e621_keep',
                         'e621_clothing')
        }
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        print(f'[Ranbooru] Only Poses tag lists unavailable ({exc}); '
              'using built-in rules only.')
        return dict(POSE_EMPTY_GROUPS)
    # Single-word clothing tags (shirt, skirt) become nouns for white_shirt etc.
    groups['clothing_nouns'] = frozenset(
        tag for tag in groups['clothing'] if '_' not in tag
    ) | {'hair_ornament', 'hair_ribbon', 'hair_bow', 'hair_flower',
         'sailor_collar', 'high_heels', 'tank_top', 'crop_top', 'nails'}
    return groups


def load_pose_name_tags(path):
    """Danbooru's most-used character/copyright/artist names, for boorus without tag categories."""
    try:
        with open(path, 'r', encoding='utf-8') as names_file:
            data = json.load(names_file)
        return frozenset(
            tag
            for key in ('copyright', 'character', 'artist')
            for tag in data.get(key, [])
            if isinstance(tag, str)
        )
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        print(f'[Ranbooru] Only Poses name list unavailable ({exc}); '
              'names are only stripped where the booru reports tag categories.')
        return frozenset()


def load_pose_name_qualifiers(path):
    """Series names that mark a tag as a character/series item: name_(series)."""
    try:
        with open(path, 'r', encoding='utf-8') as names_file:
            data = json.load(names_file)
        return frozenset(
            q for q in data.get('qualifiers', []) if isinstance(q, str)
        )
    except (OSError, ValueError, AttributeError, TypeError):
        return frozenset()


POSE_QUALIFIER_PATTERN = re.compile(r'_\(([^()]+)\)')


def has_series_qualifier(norm, qualifiers):
    """jester_(fire_emblem)_(timeskip) -> True when 'fire_emblem' is a known series."""
    if '_(' not in norm:
        return False
    return any(q in qualifiers for q in POSE_QUALIFIER_PATTERN.findall(norm))


# Boorus that share Danbooru's tag names but report no tag categories.
POSE_NAME_LIST_BOORUS = ('gelbooru',)


def pose_names_for(booru, only_poses, names_all_boorus, names):
    """The Danbooru name list applies on Gelbooru, or anywhere when the experimental toggle is on."""
    if only_poses and (booru in POSE_NAME_LIST_BOORUS or names_all_boorus):
        return names
    return frozenset()


def _normalize_pose_tag(tag):
    """Compare tags in booru form: lower case, underscores, unescaped parens."""
    tag = tag.strip().lower().replace('\\(', '(').replace('\\)', ')')
    return '_'.join(tag.split())


def parse_tag_list(text):
    """Split a comma- or newline-separated tag list; trim and drop empties."""
    return [tag.strip() for tag in re.split(r'[,\n]', str(text or '')) if tag.strip()]


def remove_listed_tags(tags, bad_tags):
    """Drop tags named in bad_tags. `*` makes an entry match any tag that
    contains the rest of it (`*hair`). Spaces/underscores and case are
    ignored on both sides."""
    exact = set()
    partial = []
    for bad_tag in bad_tags:
        norm = _normalize_pose_tag(str(bad_tag))
        if '*' in norm:
            norm = norm.replace('*', '').strip('_')
            if norm:
                partial.append(norm)
        elif norm:
            exact.add(norm)

    kept = []
    for tag in tags:
        norm = _normalize_pose_tag(tag)
        if not norm or norm in exact:
            continue
        if any(part in norm for part in partial):
            continue
        kept.append(tag)
    return kept


def prompt_tag_key(tag):
    """Compare a prompt tag by its booru name.

    Attention syntax is dropped, `(blue hair:1.2)` and `[blue_hair]` become
    blue_hair; escaped parentheses are kept, `name \\(series\\)` is
    name_(series); `<lora:...>` and other extra-network tags give None.
    """
    tag = str(tag or '').strip()
    if not tag or tag.startswith('<'):
        return None
    tag = tag.replace('\\(', '\x00').replace('\\)', '\x01')
    tag = re.sub(r'[()\[\]{}]', '', tag)
    tag = re.sub(r':\s*-?[\d.]+\s*$', '', tag)
    tag = tag.replace('\x00', '(').replace('\x01', ')')
    return _normalize_pose_tag(tag) or None


def prompt_tag_keys(prompt):
    """prompt_tag_key of every comma-separated tag of a prompt (or prompts)."""
    prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt]
    keys = set()
    for text in prompts:
        for tag in str(text or '').split(','):
            key = prompt_tag_key(tag)
            if key:
                keys.add(key)
    return keys


def _matches_structure(words, nouns, modifiers):
    """True when the tag is a noun preceded only by (at least one) modifier."""
    for size in (2, 1):
        if len(words) <= size:
            continue
        if '_'.join(words[-size:]) in nouns and all(
            word in modifiers for word in words[:-size]
        ):
            return True
    return False


def is_characteristic_tag(tag, strip_traits=True, strip_clothing=False,
                          excluded=(), groups=None, names=(), qualifiers=(),
                          e621_vocab=False):
    """Whether Only Poses / clothing stripping should remove this tag.

    e621_vocab adds e621's own body vocabulary (tufts, markings, pawpads)
    and its pose/expression keep list; it is set only for e621 posts.
    """
    if groups is None:
        groups = pose_tag_groups
    norm = _normalize_pose_tag(tag)
    if not norm:
        return False

    # The booru itself says this is an artist/character/copyright tag.
    if strip_traits and (norm in excluded or norm.endswith(POSE_QUALIFIER_SUFFIXES)):
        return True
    if norm in groups['keep']:
        return False
    if e621_vocab and norm in groups['e621_keep']:
        return False
    words = norm.split('_')

    if strip_traits:
        # Danbooru name list (character/copyright/artist), when enabled.
        if norm in names or has_series_qualifier(norm, qualifiers):
            return True
        if e621_vocab and (
            norm in groups['e621_traits']
            or norm.startswith(POSE_E621_BODY_PREFIXES)
            or norm.endswith(POSE_E621_BODY_SUFFIXES)
            or _matches_structure(words, POSE_E621_BODY_NOUNS,
                                  POSE_E621_BODY_MODIFIERS)
        ):
            return True
        if norm in groups['traits']:
            return True
        if _matches_structure(words, POSE_COLOR_TRAIT_NOUNS, POSE_COLOR_WORDS):
            return True
        if _matches_structure(words, POSE_CREATURE_TRAIT_NOUNS,
                              POSE_CREATURE_WORDS | POSE_COLOR_WORDS):
            return True

    if strip_clothing:
        if norm in groups['clothing']:
            return True
        if e621_vocab and (
            norm in groups['e621_clothing']
            or _matches_structure(
                words, POSE_E621_CLOTHING_NOUNS | groups['clothing_nouns'],
                (POSE_E621_CLOTHING_MODIFIERS | POSE_CLOTHING_MODIFIERS)
                - POSE_E621_CLOTHING_STATES)
        ):
            return True
        if _matches_structure(words, groups['clothing_nouns'], POSE_CLOTHING_MODIFIERS):
            return True

    return False


def filter_pose_tags(tags, strip_traits=True, strip_clothing=False,
                     excluded=(), groups=None, names=(), qualifiers=(),
                     e621_vocab=False):
    """Return the tags (in order) that survive Only Poses / clothing stripping."""
    if not strip_traits and not strip_clothing:
        return list(tags)
    return [
        tag for tag in tags
        if not is_characteristic_tag(tag, strip_traits, strip_clothing,
                                     excluded, groups, names, qualifiers,
                                     e621_vocab)
    ]


def post_category_tags(post):
    """Artist, character and copyright tags a booru post reports, normalized."""
    found = []
    for field in POSE_CATEGORY_FIELDS:
        value = post.get(field)
        if isinstance(value, str):
            found.extend(value.split())
    categories = post.get('ranbooru_tag_categories')
    if isinstance(categories, dict):
        for values in categories.values():
            if isinstance(values, list):
                found.extend(v for v in values if isinstance(v, str))
    return {_normalize_pose_tag(tag) for tag in found if tag.strip()}


pose_tag_groups = load_pose_tag_groups(
    os.path.join(extension_root, 'pose_tag_groups.json')
)
pose_name_tags = load_pose_name_tags(
    os.path.join(extension_root, 'pose_name_tags.json')
)
pose_name_qualifiers = load_pose_name_qualifiers(
    os.path.join(extension_root, 'pose_name_tags.json')
)


POST_AMOUNT = 100

# Number of images the search returned.
# Booru classes below were modified to update this value
# with the latest search result count.
COUNT = 100

DEBUG = False

RATING_TYPES = {
    "none": {
        "All": "All"
    },

    "full": {
        "All": "All",
        "Safe": "safe",
        "Questionable": "questionable",
        "Explicit": "explicit"
    },

    "single": {
        "All": "All",
        "Safe": "g",
        "Sensitive": "s",
        "Questionable": "q",
        "Explicit": "e"
    }
}

RATINGS = {
    "e621": RATING_TYPES['full'],
    "danbooru": RATING_TYPES['single'],
    "aibooru": RATING_TYPES['full'],
    "yande.re": RATING_TYPES['full'],
    "konachan": RATING_TYPES['full'],
    "safebooru": RATING_TYPES['none'],
    "rule34": RATING_TYPES['full'],
    "xbooru": RATING_TYPES['full'],
    "gelbooru": RATING_TYPES['single']
}


# Generation settings copied from the txt2img run into Ranbooru's img2img pass.
IMG2IMG_COPIED_SETTINGS = (
    'prompt', 'negative_prompt', 'styles', 'seed', 'subseed',
    'subseed_strength', 'seed_resize_from_h', 'seed_resize_from_w',
    'sampler_name', 'scheduler', 'batch_size', 'n_iter', 'cfg_scale',
    'distilled_cfg_scale', 'restore_faces', 'tiling', 'override_settings',
    'override_settings_restore_afterwards',
)


def img2img_settings_from(p, img2img_class):
    """Settings of p that img2img_class accepts as constructor arguments."""
    accepted = None
    if dataclasses.is_dataclass(img2img_class):
        accepted = {f.name for f in dataclasses.fields(img2img_class) if f.init}
    return {
        name: getattr(p, name)
        for name in IMG2IMG_COPIED_SETTINGS
        if hasattr(p, name) and (accepted is None or name in accepted)
    }


def per_image_values(value, total):
    """A list with one entry per image: a single value repeated, or a list
    cycled/trimmed to `total`."""
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if not values:
        values = ['']
    return [values[i % len(values)] for i in range(total)]


def per_image_seeds(p, total, kind='seed'):
    """The seed (or subseed) of every image of the txt2img run, which Forge
    fills in all_seeds / all_subseeds. None when unknown: then each pass
    gets a random one."""
    seeds = list(getattr(p, f'all_{kind}s', None) or [])
    if len(seeds) >= total:
        return seeds[:total]
    seed = getattr(p, kind, -1)
    seed = seed[0] if isinstance(seed, (list, tuple)) and seed else seed
    try:
        base = int(seeds[0] if seeds else seed)
    except (TypeError, ValueError):
        return None
    if base == -1:
        return None
    return [base + i for i in range(total)]


def source_infotext(infotexts, first_image_index, num):
    """Infotext of the num-th generated image (a grid, if any, comes first)."""
    if not infotexts:
        return ''
    index = (first_image_index or 0) + num
    return infotexts[min(index, len(infotexts) - 1)]


def send_image_to_controlnet(p, image, weight):
    """Put a booru image into the first unit of Forge Neo's built-in ControlNet.

    Neo's ControlNet reads its units from p.script_args when it runs
    process(), after every before_process hook, so the first unit is replaced
    there. The user picks the model/preprocessor in ControlNet Unit 1; this
    enables the unit and supplies the image and weight.
    """
    runner = getattr(p, 'scripts', None)
    scripts_list = getattr(runner, 'alwayson_scripts', None) or []
    controlnet = next(
        (script for script in scripts_list
         if str(script.title()).strip().lower() == 'controlnet'),
        None
    )
    if controlnet is None:
        raise RuntimeError(
            "[Ranbooru] Send to ControlNet needs Forge Neo's built-in "
            "ControlNet, which was not found."
        )

    args = list(p.script_args or ())
    start, end = controlnet.args_from, controlnet.args_to
    if start is None or end is None or end <= start or len(args) <= start:
        raise RuntimeError(
            '[Ranbooru] Send to ControlNet: no ControlNet unit was found in '
            'the generation arguments.'
        )

    pixels = {'image': np.asarray(image.convert('RGB')), 'mask': None}
    unit = args[start]
    if isinstance(unit, dict):
        unit = dict(unit)
        unit.update(enabled=True, weight=weight, image=pixels,
                    use_preview_as_input=False)
        model = unit.get('model', 'None')
    else:
        changes = dict(enabled=True, weight=weight, image=pixels,
                       use_preview_as_input=False)
        if dataclasses.is_dataclass(unit):
            unit = dataclasses.replace(unit, **changes)
        else:
            unit = copy.copy(unit)
            for key, value in changes.items():
                setattr(unit, key, value)
        model = getattr(unit, 'model', 'None')

    if str(model or 'None') == 'None':
        print(
            '[Ranbooru] Send to ControlNet: pick a model in ControlNet '
            'Unit 1, otherwise the image is ignored.'
        )

    args[start] = unit
    p.script_args = type(p.script_args)(args) if isinstance(
        p.script_args, (list, tuple)) else tuple(args)


def get_available_ratings(booru):
    # gr.update: the per-component .update() methods were removed in Gradio 4.
    return gr.update(
        choices=list(RATINGS.get(booru, RATINGS['safebooru']).keys()),
        value="All"
    )


def show_fringe_benefits(booru):
    return gr.update(visible=(booru == 'gelbooru'))


def check_exception(booru, parameters):
    tags = parameters.get('tags')

    # Danbooru's public search limit is normally two tags. Previously Ranbooru
    # used a slot for its automatic -animated filter, leaving only one for users.
    # The Danbooru adapter now filters animated posts locally instead.
    if booru == 'danbooru' and tags:
        user_tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
        if len(user_tags) > 2:
            raise ValueError(
                '[Ranbooru] Danbooru allows up to two search tags for '
                'anonymous/basic accounts. Remove excess tags, or use '
                'an account with a higher search-tag limit.'
            )


# Booru sites that take an API key plus a user ID (or username) for login.
CREDENTIAL_BOORUS = {
    'gelbooru': {
        'title': 'Gelbooru API Credentials',
        'key_hint': 'Enter your Gelbooru API key',
        'user_label': 'User ID',
        'user_hint': 'Enter your Gelbooru user ID',
    },
    'rule34': {
        'title': 'Rule34 API Credentials',
        'key_hint': 'Enter your Rule34 API key (My Account > Options on rule34.xxx)',
        'user_label': 'User ID',
        'user_hint': 'Enter your Rule34 user ID',
    },
    'e621': {
        'title': 'e621 Login (optional)',
        'key_hint': 'Enter your e621 API key (Account > Manage API Access)',
        'user_label': 'Username',
        'user_hint': 'Enter your e621 username',
    },
}


def resolve_booru_credentials(booru, api_key, user_id, save_credentials,
                              manager):
    """Return (api_key, user_id) for the selected booru, or (None, None).

    Credentials saved for the selected booru win over the textboxes, so a key
    left in the (hidden) fields for one site is never sent to another.
    """
    if booru not in CREDENTIAL_BOORUS:
        return None, None

    saved = manager.get_booru_credentials(booru)
    saved_key = str(saved.get('api_key', '') or '').strip()
    saved_user = str(saved.get('user_id', '') or '').strip()
    if saved_key and saved_user:
        return saved_key, saved_user

    key = str(api_key or '').strip()
    user = str(user_id or '').strip()
    if key and user:
        if save_credentials:
            manager.save_booru_credentials(booru, key, user)
        return key, user

    return None, None


# --- Booru HTTP and the Use cache option --------------------------------------
# Booru API requests go through booru_http. With Use cache on they use a
# Ranbooru-only cache (user/ranbooru_cache.sqlite, one hour); everything else
# in the WebUI, and Ranbooru's own image downloads, use plain requests.
RANBOORU_CACHE_FILE = os.path.join(user_data_dir, 'ranbooru_cache')
# Credentials are left out of what is stored, whatever requests-cache defaults
# the installed version has.
CACHE_IGNORED_PARAMETERS = (
    'api_key', 'user_id', 'login', 'password_hash', 'access_token',
    'Authorization', 'X-API-Key',
)
CACHE_REFUSED_TEXT = (
    'missing authentication', '"success":false', '"success": false',
    'too deep', 'search error', 'api limited',
)


def cacheable_reply(response):
    """Cache only successful API replies, never errors or refusals, so a
    login added later is not answered from an hour-old refusal."""
    try:
        if response.status_code != 200:
            return False
        text = (response.text or '')[:4096].lstrip().lower()
    except Exception:
        return False
    if not text.startswith(('[', '{', '<?xml', '<posts')):
        return False
    return not any(marker in text for marker in CACHE_REFUSED_TEXT)


class BooruHTTP:
    def __init__(self):
        self.session = None

    def set_cache(self, enabled):
        if enabled and self.session is None:
            try:
                self.session = requests_cache.CachedSession(
                    RANBOORU_CACHE_FILE,
                    backend='sqlite',
                    expire_after=3600,
                    ignored_parameters=CACHE_IGNORED_PARAMETERS,
                    # Gelbooru's Fringe Benefits cookie changes the results.
                    match_headers=['Cookie'],
                    filter_fn=cacheable_reply,
                )
            except Exception as exc:
                print(f'[Ranbooru] Use cache unavailable ({exc}); continuing without it.')
                self.session = None
        elif not enabled and self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None

    def get(self, url, cache=True, **kwargs):
        if cache and self.session is not None:
            return self.session.get(url, **kwargs)
        return requests.get(url, **kwargs)


booru_http = BooruHTTP()


def remove_legacy_global_cache():
    """Older Ranbooru installed a process-wide cache; a "Reload UI" keeps it
    patched into requests. Remove it, but only if it is Ranbooru's."""
    try:
        if not requests_cache.patcher.is_installed():
            return
        cache = requests_cache.get_cache()
        name = str(getattr(cache, 'cache_name', '') or getattr(cache, 'db_path', ''))
        if 'ranbooru_cache' in name:
            requests_cache.uninstall_cache()
    except Exception:
        pass


remove_legacy_global_cache()


def redact_secrets(text, secrets=()):
    text = str(text)
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), '***')
    return text


POSTS_COUNT_PATTERN = re.compile(r'<posts\b[^>]*\bcount="(\d+)"')
XML_REASON_PATTERN = re.compile(
    r'reason="([^"]*)"|<error>([^<]*)</error>', re.IGNORECASE
)


def fetch_booru_json(url, site, headers=None, auth=None, secrets=()):
    """GET a booru JSON endpoint and return the decoded body.

    Every failure becomes a RuntimeError that names the site and the reason,
    with credentials redacted. An empty body is an empty result.
    """
    try:
        response = booru_http.get(
            url, headers=headers, auth=auth, timeout=20
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f'[Ranbooru] Unable to contact {site}: '
            f'{redact_secrets(exc, secrets)}'
        ) from exc

    status = getattr(response, 'status_code', 200)
    text = (getattr(response, 'text', '') or '').strip()

    if 'Missing authentication' in text:
        raise RuntimeError(
            f'[Ranbooru] {site} requires an API key and user ID. Enter them '
            f'in the {site} credentials fields of the Ranbooru panel.'
        )
    if status in (401, 403):
        raise RuntimeError(
            f'[Ranbooru] {site} rejected the request (HTTP {status}). '
            f'Check the {site} API key and user ID.'
        )

    match = XML_REASON_PATTERN.search(text) if text.startswith('<') else None
    reason = (match.group(1) or match.group(2)).strip() if match else ''
    if status >= 400:
        detail = f': {redact_secrets(reason, secrets)}' if reason else ''
        if status == 429:
            detail += '. Rate limit reached; retry later'
        raise RuntimeError(
            f'[Ranbooru] {site} API error (HTTP {status}){detail}.'
        )
    if not text:
        return []
    if text.startswith('<'):
        raise RuntimeError(
            f'[Ranbooru] {site} API error: '
            f'{redact_secrets(reason or "unexpected XML/HTML response", secrets)}.'
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f'[Ranbooru] {site} returned a non-JSON response (HTTP {status}).'
        ) from exc

    if isinstance(data, dict) and data.get('success') is False:
        message = data.get('message') or data.get('reason') or 'request rejected'
        raise RuntimeError(
            f'[Ranbooru] {site} API error: '
            f'{redact_secrets(message, secrets)}.'
        )
    if isinstance(data, str):
        raise RuntimeError(
            f'[Ranbooru] {site} API error: {redact_secrets(data, secrets)}'
        )
    return data


def fetch_xml_count(url, headers=None):
    """Return the result count from an XML `<posts count=...>` reply, or None.

    Used only to pick a page in range, so any failure is soft.
    """
    try:
        response = booru_http.get(url, headers=headers, timeout=20)
        match = POSTS_COUNT_PATTERN.search(getattr(response, 'text', '') or '')
    except (requests.exceptions.RequestException, AttributeError, TypeError):
        return None
    return int(match.group(1)) if match else None


def pages_for_count(count, max_pages, page_limit=None):
    """Number of pages worth sampling: Max Pages, capped by results and site."""
    pages = max(1, int(max_pages or 1))
    if count is not None:
        pages = min(pages, max(1, -(-int(count) // POST_AMOUNT)))
    if page_limit:
        pages = min(pages, page_limit)
    return pages


def parse_post_id(site, post_id):
    post_id = str(post_id or '').strip()
    if not post_id.isdigit():
        raise ValueError(
            f'[Ranbooru] {site} post IDs are numbers; got {post_id!r}.'
        )
    return post_id


class Booru:
    def __init__(self, booru, booru_url):
        self.booru = booru
        self.booru_url = booru_url
        self.last_url = booru_url

        self.headers = {
            'user-agent': 'my-app/0.0.1'
        }

    def get_data(self, add_tags, max_pages=10, id=''):
        pass

    def get_post(self, add_tags, max_pages=10, id=''):
        pass


class Gelbooru(Booru):
    """Gelbooru API adapter with bounded retries and validated JSON responses."""

    RETRYABLE_STATUSES = (429, 500, 502, 503, 504)
    RETRY_DELAYS = (1, 2)

    class HTMLResponseError(RuntimeError):
        """A Gelbooru page returned HTML instead of JSON."""

    class TooDeepError(HTMLResponseError):
        """Gelbooru refused a page past its paging limit ("Too deep!")."""

    # Gelbooru answers pages past its paging limit with a plain "Too deep!
    # Pull it back some." (HTTP 200). In November 2026 the limit was about
    # 1,000 posts: page (pid) 10 served, 11 refused at limit=100 (measured by
    # the sd-webui-ranbooru-D fork; a user log showed pid 80 refused). Random
    # posts therefore come from page 0 with sort:random, which Gelbooru draws
    # from the whole search; pages are only walked, capped here, when the
    # search has its own sort: order.
    MAX_PAGES = 10


    def __init__(self, fringe_benefits, api_key=None, user_id=None):
        super().__init__(
            'gelbooru',
            (
                'https://gelbooru.com/index.php'
                '?page=dapi&s=post&q=index'
                f'&json=1&limit={POST_AMOUNT}'
            )
        )
        self.fringeBenefits = fringe_benefits
        self.api_key = api_key
        self.user_id = user_id
        self.headers = {'User-Agent': 'Ranbooru-Forge-Neo/1.0'}

    def _fetch_json(self, url, page_number=None, cache=True):
        """Fetch and validate a post page, without exposing credentials in logs.

        Auth parameters are deliberately separate from self.last_url, and no
        exception contains a request URL (requests would include the API key).
        """
        auth = (
            {'api_key': self.api_key, 'user_id': self.user_id}
            if self.api_key and self.user_id else None
        )
        cookies = {'fringeBenefits': 'yup'} if self.fringeBenefits else None

        for attempt in range(3):
            try:
                response = booru_http.get(
                    url,
                    cache=cache,
                    params=auth,
                    cookies=cookies,
                    headers=self.headers,
                    timeout=20,
                )
            except requests.exceptions.RequestException:
                if attempt < 2:
                    time.sleep(self.RETRY_DELAYS[attempt])
                    continue
                raise RuntimeError(
                    '[Ranbooru] Gelbooru network request failed after '
                    '3 attempts. Check your connection and try again.'
                ) from None

            status = response.status_code
            # Log only bounded diagnostics, never URLs, bodies, or credentials.
            content_type = response.headers.get('Content-Type', '').lower()
            sample = (response.text or '').lstrip()[:32].lower()
            response_kind = (
                'HTML' if 'html' in content_type or sample.startswith(('<html', '<!doctype'))
                else 'JSON' if 'json' in content_type
                else 'other'
            )
            print(
                f'[Ranbooru] Gelbooru page={page_number} '
                f'HTTP={status} response={response_kind} '
                f'attempt={attempt + 1}/3'
            )
            if status in (401, 403):
                if not auth:
                    raise RuntimeError(
                        f'[Ranbooru] Gelbooru denied tag-search API access '
                        f'(HTTP {status}). No API key/user ID pair was sent. '
                        'Add BOTH values in the Gelbooru settings and enable '
                        '"Save credentials" if you want them kept after restart.'
                    )
                raise RuntimeError(
                    f'[Ranbooru] Gelbooru denied API access (HTTP {status}) '
                    'even though an API key and user ID were supplied. '
                    'Verify the pair in your Gelbooru account options.'
                )
            if status in self.RETRYABLE_STATUSES:
                if attempt < 2:
                    time.sleep(self.RETRY_DELAYS[attempt])
                    continue
                raise RuntimeError(
                    f'[Ranbooru] Gelbooru API temporarily unavailable '
                    f'(HTTP {status}) after 3 attempts. Try again later.'
                )
            if status >= 400:
                raise RuntimeError(
                    f'[Ranbooru] Gelbooru API rejected the request '
                    f'(HTTP {status}). Check your search tags and API access.'
                )

            try:
                data = response.json()
            except ValueError:
                # HTML login/challenge/error pages are not transient JSON errors.
                content_type = response.headers.get('Content-Type', '').lower()
                prefix = (response.text or '').lstrip()[:120].lower()
                if 'too deep' in prefix:
                    raise self.TooDeepError(
                        f'[Ranbooru] Gelbooru refused page {page_number}: it '
                        'only pages about 1,000 posts into a search.'
                    ) from None
                if 'html' in content_type or prefix.startswith(('<!doctype', '<html')):
                    if not auth:
                        raise self.HTMLResponseError(
                            '[Ranbooru] Gelbooru returned an HTML page instead of JSON '
                            f'(HTTP {status}). No API key/user ID pair was sent. '
                            'Gelbooru may allow post-ID lookups but require '
                            'authentication for tag searches. Enter BOTH '
                            'values under Gelbooru credentials; select '
                            '"Save credentials" to retain them after restart.'
                        ) from None
                    raise self.HTMLResponseError(
                        '[Ranbooru] Gelbooru returned an HTML page instead of JSON '
                        f'(HTTP {status}) even though an API key and user ID '
                        'were supplied. A site challenge, search restriction, '
                        'or temporary issue may be blocking tag searches. '
                        'Try Max pages = 1 and a simple tag, then verify '
                        'your credentials and Gelbooru access.'
                    ) from None
                if attempt < 2:
                    time.sleep(self.RETRY_DELAYS[attempt])
                    continue
                raise RuntimeError(
                    '[Ranbooru] Gelbooru returned empty or invalid JSON after '
                    '3 attempts. The API may be temporarily unavailable.'
                ) from None

            if isinstance(data, dict) and str(data.get('success', '')).lower() == 'false':
                raise RuntimeError(
                    '[Ranbooru] Gelbooru reported a failed API search. '
                    'Its search service may be down, or the query may be '
                    'restricted. Try again later or check your tags.'
                )

            # Gelbooru normally returns @attributes/count and a list of posts.
            # Empty list responses are also accepted as a no-results search.
            if data == []:
                data = {'@attributes': {'count': 0}, 'post': []}
            if not isinstance(data, dict):
                raise RuntimeError(
                    '[Ranbooru] Gelbooru returned an unexpected JSON format '
                    'instead of a post-list object.'
                )

            attributes = data.get('@attributes')
            if not isinstance(attributes, dict):
                raise RuntimeError(
                    '[Ranbooru] Gelbooru response has no valid post count. '
                    'The API response format may have changed.'
                )
            try:
                count = int(attributes.get('count'))
                if count < 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise RuntimeError(
                    '[Ranbooru] Gelbooru returned an invalid post count.'
                ) from None

            posts = data.get('post', [])
            if isinstance(posts, dict):
                posts = [posts]
            if not isinstance(posts, list):
                raise RuntimeError(
                    '[Ranbooru] Gelbooru returned an invalid posts collection.'
                )
            data['post'] = [post for post in posts if isinstance(post, dict)]
            return data, count

        raise RuntimeError('[Ranbooru] Gelbooru search failed after retries.')

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        if id:
            add_tags = ''
        requested_pages = max(1, int(max_pages))

        # A typed sort:random gives a new draw from the same URL each time.
        cache_pages = 'sort:random' not in unquote_plus(add_tags).lower()

        def fetch_page(page_number):
            # Credentials are supplied separately by _fetch_json.
            # Never print URLs or response bodies (they may include secrets).
            url = f'{self.booru_url}&pid={page_number}{id}{add_tags}'
            self.last_url = url
            return self._fetch_json(
                url, page_number=page_number, cache=cache_pages
            )

        # Direct post-ID lookup: one request only, no need for a count probe.
        if id:
            data, COUNT = fetch_page(0)
            return data

        # Random posts: one request, page 0 with sort:random, which Gelbooru
        # draws from the whole search. Searches with their own sort: order
        # walk pages instead (capped at Gelbooru's paging limit).
        if 'sort:' not in unquote_plus(add_tags).lower():
            random_tags = (
                f'{add_tags}+sort:random' if '&tags=' in add_tags
                else f'{add_tags}&tags=sort:random'
            )
            try:
                # Same URL every time: never answer it from the cache.
                url = f'{self.booru_url}&pid=0{random_tags}'
                self.last_url = url
                data, COUNT = self._fetch_json(url, page_number=0, cache=False)
            except self.HTMLResponseError as exc:
                print(
                    f'[Ranbooru] Gelbooru sort:random was refused ({exc}); '
                    'walking pages instead.'
                )
            else:
                if data['post'] or COUNT == 0:
                    sample = ', '.join(
                        str(post.get('id')) for post in data['post'][:5]
                    )
                    print(
                        f'[Ranbooru] Gelbooru sort:random: matching_posts='
                        f'{COUNT}, {len(data["post"])} posts (ids {sample}...)'
                    )
                    return data
                print(
                    '[Ranbooru] Gelbooru sort:random returned no posts; '
                    'walking pages instead.'
                )

        # Page 0 tells us how many pages are really available for this query.
        # Do not select a random page until we have validated that response.
        first_page, COUNT = fetch_page(0)
        available_pages = max(1, (COUNT + POST_AMOUNT - 1) // POST_AMOUNT)
        eligible_pages = min(requested_pages, available_pages, self.MAX_PAGES)
        selected_page = random.randrange(eligible_pages)
        print(
            f'[Ranbooru] Gelbooru matching_posts={COUNT} '
            f'available_pages={available_pages} '
            f'max_pages={requested_pages} selected_page={selected_page}'
        )

        if selected_page == 0 or COUNT == 0:
            data = first_page
        else:
            try:
                data, _ = fetch_page(selected_page)
                if not data['post']:
                    print(
                        f'[Ranbooru] Gelbooru page={selected_page} returned '
                        'no posts; falling back to page=0.'
                    )
                    data = first_page
            except self.HTMLResponseError:
                # Reuse the already-validated page 0 as a fallback rather than
                # making a third API request and increasing quota pressure.
                print(
                    f'[Ranbooru] Gelbooru page={selected_page} returned HTML; '
                    'falling back to validated page=0 results.'
                )
                data = first_page

        shown = min(eligible_pages * POST_AMOUNT, COUNT)
        print(f' Processing up to {shown} out of {COUNT} results.')
        return data

    def get_post(self, add_tags, max_pages=10, id=''):
        return self.get_data(add_tags, max_pages, '&id=' + str(id))


class GelbooruBeta(Booru):
    """Gelbooru 0.2 dapi sites (XBooru, Safebooru, Rule34)."""

    # The dapi refuses offsets past pid 2000 at 100 posts per page
    # ("Search error: API limited due to abuse.").
    MAX_PID = 2000

    def __init__(self, booru, base_url, image_host):
        super().__init__(
            booru,
            f'{base_url}?page=dapi&s=post&q=index&json=1&limit={POST_AMOUNT}'
        )
        self.base_url = base_url
        self.image_host = image_host

    def _auth_params(self):
        return ''

    def _secrets(self):
        return ()

    def _get(self, url):
        self.last_url = redact_secrets(url, self._secrets())
        print(f'{self.booru} URL: {self.last_url}')
        return fetch_booru_json(
            url, self.booru, self.headers, secrets=self._secrets()
        )

    def _normalize(self, data):
        if data is None:
            return []
        if not isinstance(data, list):
            raise RuntimeError(
                f'[Ranbooru] {self.booru} returned an unexpected '
                f'{type(data).__name__} instead of a post list.'
            )
        posts = []
        for post in data:
            if not isinstance(post, dict):
                continue
            file_url = str(post.get('file_url') or '')
            if not file_url.startswith('http') or file_url.endswith('/'):
                # A missing post ID still returns one row with a null image
                # and a file_url ending in "/".
                if not (post.get('image') and post.get('directory')):
                    continue
                file_url = (
                    f"{self.image_host}/{post['directory']}/{post['image']}"
                )
            post['file_url'] = file_url
            post['score'] = post.get('score') or 0
            post['tags'] = post.get('tags') or ''
            posts.append(post)
        return posts

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        auth = self._auth_params()
        count = fetch_xml_count(
            f'{self.base_url}?page=dapi&s=post&q=index&limit=0'
            f'{add_tags}{auth}',
            self.headers
        )
        if count == 0:
            COUNT = 0
            return {'post': []}

        pages = pages_for_count(count, max_pages, self.MAX_PID + 1)
        pid = random.randint(0, pages - 1)
        posts = self._normalize(
            self._get(f'{self.booru_url}&pid={pid}{add_tags}{auth}')
        )
        COUNT = count if count is not None else len(posts)
        print(f' Processing page {pid + 1} of {pages} ({COUNT} results).')
        return {'post': posts}

    def get_post(self, add_tags, max_pages=10, id=''):
        global COUNT

        post_id = parse_post_id(self.booru, id)
        posts = self._normalize(
            self._get(f'{self.booru_url}&id={post_id}{self._auth_params()}')
        )
        if not posts:
            raise ValueError(
                f'[Ranbooru] {self.booru} post {post_id} was not found.'
            )
        COUNT = len(posts)
        return {'post': posts}


class XBooru(GelbooruBeta):
    def __init__(self):
        super().__init__(
            'xbooru',
            'https://xbooru.com/index.php',
            'https://xbooru.com/images'
        )


class Safebooru(GelbooruBeta):
    def __init__(self):
        super().__init__(
            'safebooru',
            'https://safebooru.org/index.php',
            'https://safebooru.org/images'
        )


class Rule34(GelbooruBeta):
    def __init__(self, username=None, api_key=None):
        super().__init__(
            'rule34',
            'https://api.rule34.xxx/index.php',
            'https://api-cdn.rule34.xxx/images'
        )
        self.username = (username or '').strip()
        self.api_key = (api_key or '').strip()
        self.headers['user-agent'] = 'Ranbooru/1.2'

    def _auth_params(self):
        if self.api_key and self.username:
            return (
                f'&api_key={quote_plus(self.api_key)}'
                f'&user_id={quote_plus(self.username)}'
            )
        return ''

    def _secrets(self):
        return tuple(
            secret for secret in (self.api_key, quote_plus(self.api_key))
            if secret
        )


class Moebooru(Booru):
    """Moebooru sites (Konachan, Yande.re). Pages are 1-based."""

    def __init__(self, booru, base_url):
        super().__init__(booru, f'{base_url}/post.json?limit={POST_AMOUNT}')
        self.base_url = base_url

    def _get(self, url):
        self.last_url = url
        print(f'{self.booru} URL: {url}')
        data = fetch_booru_json(url, self.booru, self.headers)
        if not isinstance(data, list):
            raise RuntimeError(
                f'[Ranbooru] {self.booru} returned an unexpected '
                f'{type(data).__name__} instead of a post list.'
            )
        posts = []
        for post in data:
            if not isinstance(post, dict) or not post.get('file_url'):
                continue
            post['score'] = post.get('score') or 0
            post['tags'] = post.get('tags') or ''
            posts.append(post)
        return posts

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        count = fetch_xml_count(
            f'{self.base_url}/post.xml?limit=1{add_tags}', self.headers
        )
        if count == 0:
            COUNT = 0
            return {'post': []}

        pages = pages_for_count(count, max_pages)
        page = random.randint(1, pages)
        posts = self._get(f'{self.booru_url}&page={page}{add_tags}')
        COUNT = count if count is not None else len(posts)
        print(f' Processing page {page} of {pages} ({COUNT} results).')
        return {'post': posts}

    def get_post(self, add_tags, max_pages=10, id=''):
        global COUNT

        post_id = parse_post_id(self.booru, id)
        posts = self._get(f'{self.booru_url}&tags=id:{post_id}')
        if not posts:
            raise ValueError(
                f'[Ranbooru] {self.booru} post {post_id} was not found.'
            )
        COUNT = len(posts)
        return {'post': posts}


class Konachan(Moebooru):
    def __init__(self):
        super().__init__('konachan', 'https://konachan.com')


class Yandere(Moebooru):
    def __init__(self):
        super().__init__('yande.re', 'https://yande.re')


class AIBooru(Booru):
    # Danbooru-based: pages are 1-based and capped at 1000 (HTTP 410 beyond).
    MAX_PAGE = 1000

    def __init__(self):
        super().__init__(
            'AIBooru',
            f'https://aibooru.online/posts.json?limit={POST_AMOUNT}'
        )

    def _count(self, add_tags):
        try:
            response = booru_http.get(
                'https://aibooru.online/counts/posts.json?'
                + add_tags.lstrip('&'),
                headers=self.headers,
                timeout=20
            )
            count = response.json()['counts']['posts']
        except (requests.exceptions.RequestException, ValueError,
                KeyError, TypeError, AttributeError):
            return None
        return count if isinstance(count, int) else None

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        count = self._count(add_tags)
        if count == 0:
            COUNT = 0
            return {'post': []}

        pages = pages_for_count(count, max_pages, self.MAX_PAGE)
        page = random.randint(1, pages)
        url = f'{self.booru_url}&page={page}{add_tags}'
        self.last_url = url
        print(f'AIBooru URL: {url}')
        data = fetch_booru_json(url, 'AIBooru', self.headers)
        if not isinstance(data, list):
            raise RuntimeError(
                f'[Ranbooru] AIBooru returned an unexpected '
                f'{type(data).__name__} instead of a post list.'
            )
        posts = []
        for post in data:
            # Danbooru-based sites omit file_url on posts hidden from guests.
            if not isinstance(post, dict) or not post.get('file_url'):
                continue
            post['tags'] = post.get('tag_string') or ''
            post['score'] = post.get('score') or 0
            posts.append(post)
        COUNT = count if count is not None else len(posts)
        print(f' Processing page {page} of {pages} ({COUNT} results).')
        return {'post': posts}

    def get_post(self, add_tags, max_pages=10, id=''):
        global COUNT

        post_id = parse_post_id('AIBooru', id)
        url = f'https://aibooru.online/posts/{post_id}.json'
        self.last_url = url
        try:
            post = fetch_booru_json(url, 'AIBooru', self.headers)
        except RuntimeError as exc:
            if '(HTTP 404)' in str(exc):
                raise ValueError(
                    f'[Ranbooru] AIBooru post {post_id} was not found.'
                ) from None
            raise
        if not isinstance(post, dict) or not post.get('file_url'):
            raise ValueError(
                f'[Ranbooru] AIBooru post {post_id} was not found or has no '
                'image available.'
            )
        # Keep tag_string_artist/character/copyright for Only poses.
        post['tags'] = post.get('tag_string') or ''
        post['score'] = post.get('score') or 0
        COUNT = 1
        return {'post': [post]}


class Danbooru(Booru):
    def __init__(self):
        super().__init__(
            'danbooru',
            (
                'https://danbooru.donmai.us/posts.json'
                f'?limit={POST_AMOUNT}'
            )
        )
        # Danbooru asks API clients to identify themselves in User-Agent.
        self.headers['user-agent'] = 'Ranbooru-Forge-Neo/1.0 (tag prompt extension)'

    @staticmethod
    def _read_api_json(response, expected_type):
        """Reject API error objects instead of iterating their string keys.

        Success from /posts.json is a list; /posts/{id}.json is a dict.
        A failed request can return a dict such as {"success":false,
        "message":"..."} even when the caller expected a list.
        """
        status = getattr(response, 'status_code', 200)
        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f'[Ranbooru] Danbooru returned a non-JSON response '
                f'(HTTP {status}). Check network access or API availability.'
            ) from exc

        failed = status >= 400 or (
            isinstance(result, dict) and result.get('success') is False
        )
        if failed:
            details = []
            if isinstance(result, dict):
                for key in ('message', 'reason', 'error'):
                    value = result.get(key)
                    if isinstance(value, str) and value.strip():
                        if value.strip() not in details:
                            details.append(value.strip())
            message = ' — '.join(details) or 'request rejected'
            if status == 429:
                message += '. Rate limit reached; retry later.'
            elif status in (401, 403):
                message += '. Check Danbooru access, account permissions, or tag limits.'
            elif status == 422:
                message += '. Check your search tags and Danbooru tag limits.'
            raise RuntimeError(
                f'[Ranbooru] Danbooru API error (HTTP {status}): {message}'
            )

        if not isinstance(result, expected_type):
            if isinstance(result, dict):
                description = next(
                    (result[key] for key in ('message', 'reason', 'error')
                     if isinstance(result.get(key), str) and result[key].strip()),
                    'unexpected JSON object instead of posts'
                )
            else:
                description = f'unexpected {type(result).__name__} response'
            raise RuntimeError(
                f'[Ranbooru] Danbooru API returned invalid data '
                f'(HTTP {status}): {description}'
            )
        return result

    @staticmethod
    def _normalize_posts(posts):
        """Keep only valid post records; preserve the extension's 'tags' field."""
        usable = []
        for post in posts:
            if not isinstance(post, dict):
                print('[Ranbooru] Skipping malformed Danbooru post.')
                continue
            tag_string = post.get('tag_string')
            if not isinstance(tag_string, str):
                print('[Ranbooru] Skipping Danbooru post without tag_string.')
                continue
            # Replaces the old -animated search term without taking a tag slot.
            if 'animated' in tag_string.split() or post.get('is_animated') is True:
                continue
            post['tags'] = tag_string
            usable.append(post)
        return usable

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        if id:
            add_tags = ''

        # Danbooru pages are ONE-based. Page 0 is not a valid random page.
        last_page = max(1, int(max_pages))
        chosen_page = random.randint(1, last_page)
        pages = [chosen_page]
        if chosen_page != 1:
            pages.append(1)

        for page in pages:
            url = f'{self.booru_url}&page={page}{id}{add_tags}'
            self.last_url = url
            try:
                response = booru_http.get(url, headers=self.headers, timeout=20)
            except requests.RequestException as exc:
                raise RuntimeError(
                    f'[Ranbooru] Unable to contact Danbooru: {exc}'
                ) from exc

            raw_posts = self._read_api_json(response, list)
            posts = self._normalize_posts(raw_posts)
            COUNT = len(posts)
            if posts:
                print(f'[Ranbooru] Danbooru: {COUNT} usable posts on page {page}.')
                return {'post': posts}
            if page != 1:
                print('[Ranbooru] Danbooru page had no usable posts; '
                      'retrying page 1.')

        COUNT = 0
        return {'post': []}

    def get_post(self, add_tags, max_pages=10, id=''):
        url = f'https://danbooru.donmai.us/posts/{id}.json'
        self.last_url = url
        try:
            response = booru_http.get(url, headers=self.headers, timeout=20)
        except requests.RequestException as exc:
            raise RuntimeError(
                f'[Ranbooru] Unable to contact Danbooru: {exc}'
            ) from exc

        post = self._read_api_json(response, dict)
        if not isinstance(post.get('tag_string'), str):
            raise RuntimeError(
                '[Ranbooru] Danbooru returned a post without tag_string.'
            )
        post['tags'] = post['tag_string']
        return {'post': [post]}

class e621(Booru):
    # e621 rejects page 0 and anything beyond page 750 (HTTP 410).
    MAX_PAGE = 750
    TAG_CATEGORIES = ('general', 'artist', 'copyright', 'character', 'species')

    def __init__(self, username=None, api_key=None):
        super().__init__(
            'e621',
            f'https://e621.net/posts.json?limit={POST_AMOUNT}'
        )
        # e621 asks API clients for a descriptive User-Agent.
        self.headers['user-agent'] = 'Ranbooru-Forge-Neo/1.0 (tag prompt extension)'
        # Optional login: HTTP Basic auth with username and API key, never
        # part of the URL.
        username = (username or '').strip()
        api_key = (api_key or '').strip()
        self.auth = (username, api_key) if username and api_key else None

    @staticmethod
    def _read_api_json(response):
        """Return the post list; raise a readable error for e621 error objects."""
        status = getattr(response, 'status_code', 200)
        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f'[Ranbooru] e621 returned a non-JSON response (HTTP {status}).'
            ) from exc

        if isinstance(result, dict) and (
            status >= 400 or result.get('success') is False
        ):
            message = result.get('message') or result.get('reason') or 'request rejected'
            if status == 429:
                message += '. Rate limit reached; retry later.'
            elif status == 401:
                message += '. Check the e621 username and API key.'
            raise RuntimeError(f'[Ranbooru] e621 API error (HTTP {status}): {message}')
        if status >= 400:
            raise RuntimeError(f'[Ranbooru] e621 API error (HTTP {status}).')

        # The current API returns a bare list; older versions wrapped it.
        if isinstance(result, dict) and isinstance(result.get('posts'), list):
            result = result['posts']
        elif isinstance(result, dict) and isinstance(result.get('post'), dict):
            result = [result['post']]
        elif isinstance(result, dict) and 'id' in result:
            # /posts/{id}.json returns a bare post object.
            result = [result]
        if not isinstance(result, list):
            raise RuntimeError(
                f'[Ranbooru] e621 API returned invalid data (HTTP {status}).'
            )
        return result

    @staticmethod
    def _file_url(post):
        files = post.get('files') or post.get('file') or {}
        original = files.get('original', files) if isinstance(files, dict) else {}
        file_url = original.get('url') if isinstance(original, dict) else None
        return file_url if isinstance(file_url, str) and file_url else None

    @classmethod
    def _skip_reason(cls, post):
        """Why a post can't be used, or None when it can."""
        if not isinstance(post, dict):
            return 'malformed post data'
        flags = post.get('flags')
        if isinstance(flags, dict) and flags.get('deleted'):
            return 'the post was deleted'
        # Anonymous requests get no image URL for some posts; skip those so
        # img2img never falls back to an unrelated picture.
        if cls._file_url(post) is None:
            return 'e621 gives no image URL for it without logging in'
        tags = post.get('tags')
        if isinstance(tags, dict):
            tags = [t for values in tags.values() if isinstance(values, list) for t in values]
        elif isinstance(tags, str):
            tags = tags.split()
        if not isinstance(tags, list):
            return 'it has no tags'
        if 'animated' in tags:
            return 'it is animated'
        return None

    @classmethod
    def _normalize_post(cls, post):
        """Map an e621 post onto the fields before_process reads, or None to skip it."""
        if cls._skip_reason(post) is not None:
            return None
        file_url = cls._file_url(post)

        tags = post.get('tags')
        if isinstance(tags, dict):
            # Older API: tags grouped by category.
            post['ranbooru_tag_categories'] = {
                category: tags.get(category, [])
                for category in ('artist', 'copyright', 'character', 'species')
                if isinstance(tags.get(category), list)
            }
            tags = [
                tag
                for category in cls.TAG_CATEGORIES
                for tag in tags.get(category, [])
                if isinstance(tag, str)
            ]
        elif isinstance(tags, str):
            tags = tags.split()
        if not isinstance(tags, list):
            return None
        tags = [tag for tag in tags if isinstance(tag, str) and tag]

        stats = post.get('stats') if isinstance(post.get('stats'), dict) else post
        score = stats.get('score', 0)
        if isinstance(score, dict):
            score = score.get('total', 0)
        if type(score) is not int:
            score = 0

        post['tags'] = ' '.join(tags)
        post['score'] = score
        post['file_url'] = file_url
        return post

    def _request(self, url):
        self.last_url = url
        try:
            response = booru_http.get(url, headers=self.headers, auth=self.auth, timeout=20)
        except requests.RequestException as exc:
            raise RuntimeError(f'[Ranbooru] Unable to contact e621: {exc}') from exc
        return self._read_api_json(response)

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        if id:
            add_tags = ''

        # One request to a random ONE-based page; before_process retries
        # page 1 itself when a random page comes back empty.
        last_page = max(1, min(int(max_pages), self.MAX_PAGE))
        page = random.randint(1, last_page)
        raw_posts = self._request(f'{self.booru_url}&page={page}{id}{add_tags}')
        posts = [p for p in map(self._normalize_post, raw_posts) if p is not None]
        COUNT = len(posts)
        print(f'[Ranbooru] e621: {COUNT} usable posts on page {page}.')
        return {'post': posts}

    def get_post(self, add_tags, max_pages=10, id=''):
        global COUNT

        post_id = str(id).strip()
        if not post_id.isdigit():
            raise ValueError(
                f'[Ranbooru] e621 post IDs are numbers; got {post_id!r}.'
            )
        try:
            raw_posts = self._request(f'https://e621.net/posts/{post_id}.json')
        except RuntimeError as exc:
            if 'HTTP 404' in str(exc):
                raise RuntimeError(
                    f'[Ranbooru] e621 post {post_id} was not found.'
                ) from exc
            raise
        if not raw_posts:
            raise RuntimeError(f'[Ranbooru] e621 post {post_id} was not found.')
        reason = self._skip_reason(raw_posts[0])
        if reason:
            raise RuntimeError(
                f'[Ranbooru] e621 post {post_id} cannot be used: {reason}.'
            )
        COUNT = 1
        return {'post': [self._normalize_post(raw_posts[0])]}

    # Posts no longer carry tag categories, so look them up: meta, lore and
    # invalid tags never belonged in a prompt (the old adapter skipped those
    # categories), and Only Poses strips the name categories.
    TAGS_URL = 'https://e621.net/tags.json?limit=320&search[name]='
    NAME_CATEGORIES = {1: 'artist', 3: 'copyright', 4: 'character', 5: 'species'}
    DROP_CATEGORIES = {6: 'invalid', 7: 'meta', 8: 'lore'}
    LOOKUP_CHUNK = 300
    _category_cache = {}

    def name_tags(self, tags):
        """Artist/copyright/character/species tags among `tags`."""
        categories = self.tag_categories(tags)
        return {t for t, c in categories.items() if c in self.NAME_CATEGORIES}

    def tag_categories(self, tags):
        """Map each of `tags` to its e621 category id (None when unknown).

        Results are cached per tag; on any error the remaining tags are simply
        treated as unknown, so generation never fails because of this lookup.
        """
        from urllib.parse import quote
        cache = self._category_cache
        missing = sorted(t for t in set(tags) if t and t not in cache)
        for start in range(0, len(missing), self.LOOKUP_CHUNK):
            chunk = missing[start:start + self.LOOKUP_CHUNK]
            url = self.TAGS_URL + quote(','.join(chunk), safe=',')
            try:
                response = booru_http.get(url, headers=self.headers, auth=self.auth, timeout=20)
                rows = response.json() if response.status_code == 200 else None
            except (requests.RequestException, ValueError) as exc:
                rows = None
                print(f'[Ranbooru] e621 tag category lookup failed: {exc}')
            if not isinstance(rows, list):
                print('[Ranbooru] e621 tag categories unavailable; '
                      'names may remain in the prompt.')
                break
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get('name'), str):
                    cache[row['name']] = row.get('category')
            for tag in chunk:
                cache.setdefault(tag, None)
        return {t: cache.get(t) for t in set(tags)}


def resize_image(img, width, height, cropping=True):
    """
    Resize image to specified width and height

    Args:
        img (PIL.Image): the image
        width (int): width in pixels
        height (int): height in pixels
        cropping (bool): whether to crop

    Returns:
        PIL.Image: resized image
    """

    if cropping:

        # resize the picture and center crop it
        x, y = img.size

        if x < y:

            # scale to width keeping aspect ratio
            wpercent = width / float(img.size[0])

            hsize = int(
                float(img.size[1]) * float(wpercent)
            )

            img_new = img.resize((width, hsize))

            if img_new.size[1] < height:

                # scale to height keeping aspect ratio
                hpercent = height / float(img.size[1])

                wsize = int(
                    float(img.size[0]) * float(hpercent)
                )

                img_new = img.resize((wsize, height))

        else:
            ypercent = height / float(img.size[1])

            wsize = int(
                float(img.size[0]) * float(ypercent)
            )

            img_new = img.resize((wsize, height))

            if img_new.size[0] < width:
                xpercent = width / float(img.size[0])

                hsize = int(
                    float(img.size[1]) * float(xpercent)
                )

                img_new = img.resize((width, hsize))

        # crop center
        x, y = img_new.size

        left = (x - width) / 2
        top = (y - height) / 2
        right = (x + width) / 2
        bottom = (y + height) / 2

        img = img_new.crop(
            (left, top, right, bottom)
        )

    else:
        img = img.resize((width, height))

    return img


def limit_prompt_tags(prompt, limit_tags, mode):
    """Limits the amount of tags in the prompt."""

    clean_prompt = [tag.strip() for tag in prompt.split(',') if tag.strip()]

    if mode == 'Limit':
        clean_prompt = clean_prompt[:int(len(clean_prompt) * limit_tags)]
    elif mode == 'Max':
        clean_prompt = clean_prompt[:limit_tags]

    return ', '.join(clean_prompt)

def normalize_prompt(prompt):
    """Normalize prompt formatting to use ', ' between tags."""

    if not prompt:
        return prompt

    return ', '.join(
        tag.strip()
        for tag in prompt.split(',')
        if tag.strip()
    )


class Script(scripts.Script):

    last_img = []
    real_steps = 0
    version = "1.2"
    original_prompt = ''


    def title(self):
        return "Ranbooru"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def booru_credential_updates(self, booru, cleared=False):
        """Gradio updates for the credentials group of the selected booru.

        Outputs: group, heading, api_key, user_id, status, clear button.
        The textboxes are always emptied: saved credentials are read from disk
        at generation time, so no key stays in the page for another site.
        """
        spec = CREDENTIAL_BOORUS.get(booru)
        if spec is None:
            return (
                gr.update(visible=False),
                gr.update(),
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=""),
                gr.update(visible=False)
            )

        saved = (
            {} if cleared
            else credentials_manager.get_booru_credentials(booru)
        )
        api_key = str(saved.get('api_key', '') or '').strip()
        user_id = str(saved.get('user_id', '') or '').strip()
        has_saved = bool(api_key and user_id)
        status_text = (
            f"✓ Credentials loaded from: "
            f"{credentials_manager.credentials_file}"
            if has_saved else ""
        )

        return (
            gr.update(visible=True),
            gr.update(value=f"### {spec['title']}"),
            gr.update(
                visible=not has_saved,
                value="",
                placeholder=spec['key_hint']
            ),
            gr.update(
                visible=not has_saved,
                value="",
                label=spec['user_label'],
                placeholder=spec['user_hint']
            ),
            gr.update(visible=has_saved, value=status_text),
            gr.update(visible=has_saved)
        )

    def load_booru_credentials(self, booru):
        """Show the saved credentials state for the selected booru."""
        return self.booru_credential_updates(booru)

    def clear_booru_credentials(self, booru):
        """Clear saved credentials for the selected booru."""
        if booru in CREDENTIAL_BOORUS:
            credentials_manager.clear_booru_credentials(booru)
        return self.booru_credential_updates(booru, cleared=True)


    def ui(self, is_img2img):
        self._ranbooru_is_img2img = is_img2img
        saved_state = load_ranbooru_settings(is_img2img)
        if saved_state:
            tab_name = 'img2img' if is_img2img else 'txt2img'
            print(f'[Ranbooru] Restoring {len(saved_state)} settings for {tab_name}')

        def saved_value(name, default):
            """Prefer a saved control value, falling back if the state is malformed."""
            value = saved_state.get(name, default)
            if type(default) is bool:
                return value if type(value) is bool else default
            if isinstance(default, str):
                return value if isinstance(value, str) else default
            if isinstance(default, (int, float)):
                return value if type(value) in (int, float) else default
            return default

        def saved_choice(name, default, choices):
            value = saved_value(name, default)
            return value if value in choices else default

        def saved_slider(name, default, minimum, maximum):
            value = saved_value(name, default)
            return max(minimum, min(maximum, value))

        selected_booru = saved_choice('booru', 'safebooru', RATINGS)
        available_ratings = list(RATINGS[selected_booru])

        # Initial credential state for the selected booru (if it takes one)
        credential_spec = CREDENTIAL_BOORUS.get(
            selected_booru, CREDENTIAL_BOORUS['gelbooru']
        )
        saved_creds = (
            credentials_manager.get_booru_credentials(selected_booru)
            if selected_booru in CREDENTIAL_BOORUS else {}
        )
        has_saved = bool(
            str(saved_creds.get('api_key', '') or '').strip()
            and str(saved_creds.get('user_id', '') or '').strip()
        )
        if not has_saved:
            saved_creds = {}

        initial_api_key_visible = not has_saved
        initial_user_id_visible = not has_saved
        initial_status_visible = has_saved

        initial_status_value = (
            f"✓ Credentials loaded from: "
            f"{credentials_manager.credentials_file}"
            if has_saved else ""
        )

        initial_clear_visible = has_saved

        with InputAccordion(
            saved_value('enabled', False),
            label="Ranbooru",
            elem_id=self.elem_id("ra_enable")
        ) as enabled:

            # Native Gradio dropdown; no drag handles or sorting buttons.
            # Browser reloads fetch the current MRU order from booru_recent.json.
            booru = gr.Dropdown(
                load_booru_order(),
                label="Booru",
                value=selected_booru,
                elem_id=self.elem_id('ra_booru'),
            )

            max_pages = gr.Slider(
                label="Max Pages",
                minimum=1,
                maximum=100,
                value=saved_slider('max_pages', 100, 1, 100),
                step=1
            )

            gr.Markdown("""## Post""")

            post_id = gr.Textbox(
                value=saved_value('post_id', ""),
                lines=1,
                label="Post ID"
            )

            gr.Markdown("""## Tags""")

            tags = gr.Textbox(
                value=saved_value('tags', ""),
                lines=1,
                label="Tags to Search (Pre)"
            )

            remove_tags = gr.Textbox(
                value=saved_value('remove_tags', ""),
                lines=1,
                label="Tags to Remove (Post)"
            )

            mature_rating = gr.Radio(
                available_ratings,
                label="Mature Rating",
                value=saved_choice('mature_rating', "All", available_ratings)
            )

            remove_bad_tags = gr.Checkbox(
                label="Remove bad tags",
                value=saved_value('remove_bad_tags', True)
            )

            shuffle_tags = gr.Checkbox(
                label="Shuffle tags",
                value=saved_value('shuffle_tags', True)
            )

            only_poses = gr.Checkbox(
                label="Only poses (strip appearance, character, copyright & artist tags)",
                info="Names come from booru tag categories on Danbooru/AIBooru and from Danbooru's top-name list on Gelbooru",
                value=saved_value('only_poses', False)
            )

            strip_clothing = gr.Checkbox(
                label="Strip clothing & accessories",
                value=saved_value('strip_clothing', False)
            )

            names_all_boorus = gr.Checkbox(
                label="Experimental: strip Danbooru name list on every booru",
                info="With Only poses, also removes Danbooru's ~35k most-used character, copyright and artist names on boorus other than Gelbooru",
                value=saved_value('names_all_boorus', False)
            )

            skip_prompt_tags = gr.Checkbox(
                label="Skip tags already in prompt",
                info="Leave out fetched tags you already typed, e.g. blue_hair when the prompt has (blue hair:1.2)",
                value=saved_value('skip_prompt_tags', False)
            )

            change_dash = gr.Checkbox(
                label='Convert "_" to spaces',
                value=saved_value('change_dash', False)
            )

            same_prompt = gr.Checkbox(
                label="Use same prompt for all images",
                value=saved_value('same_prompt', False)
            )

            fringe_benefits = gr.Checkbox(
                label="Fringe Benefits",
                value=saved_value('fringe_benefits', True),
                visible=(selected_booru == 'gelbooru')
            )

            # API credentials for boorus that take a key and user ID
            with gr.Group(
                visible=(selected_booru in CREDENTIAL_BOORUS)
            ) as credentials_group:

                credentials_heading = gr.Markdown(
                    f"### {credential_spec['title']}"
                )

                api_key = gr.Textbox(
                    lines=1,
                    label="API Key",
                    placeholder=credential_spec['key_hint'],
                    type="password",
                    visible=initial_api_key_visible,
                    value=""
                )

                user_id = gr.Textbox(
                    lines=1,
                    label=credential_spec['user_label'],
                    placeholder=credential_spec['user_hint'],
                    visible=initial_user_id_visible,
                    value=""
                )

                save_credentials = gr.Checkbox(
                    label="Save credentials",
                    value=saved_value('save_credentials', False),
                    visible=True
                )

                credentials_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=initial_status_visible,
                    value=initial_status_value
                )

                clear_credentials_btn = gr.Button(
                    "Clear saved credentials",
                    visible=initial_clear_visible
                )

            limit_tags = gr.Slider(
                value=saved_slider('limit_tags', 1.0, 0.05, 1),
                label="Limit tags",
                minimum=0.05,
                maximum=1.0,
                step=0.05
            )

            max_tags = gr.Slider(
                value=saved_slider('max_tags', 100, 1, 100),
                label="Max tags",
                minimum=1,
                maximum=100,
                step=1
            )

            change_background = gr.Radio(
                [
                    "Don't Change",
                    "Add Background",
                    "Remove Background",
                    "Remove All"
                ],

                label="Change Background",
                value=saved_choice('change_background', "Don't Change", ["Don't Change", 'Add Background', 'Remove Background', 'Remove All'])
            )

            change_color = gr.Radio(
                [
                    "Don't Change",
                    "Colored",
                    "Limited Palette",
                    "Monochrome"
                ],

                label="Change Color",
                value=saved_choice('change_color', "Don't Change", ["Don't Change", 'Colored', 'Limited Palette', 'Monochrome'])
            )

            sorting_order = gr.Radio(
                [
                    "Random",
                    "High Score",
                    "Low Score"
                ],

                label="Sorting Order",
                value=saved_choice('sorting_order', 'Random', ['Random', 'High Score', 'Low Score'])
            )

            booru.input(
                get_available_ratings,
                booru,
                mature_rating
            )

            # display fringe benefits checkbox if gelbooru is selected

            booru.input(
                show_fringe_benefits,
                booru,
                fringe_benefits
            )

            gr.Markdown("""\n---\n""")
            with gr.Group():

                with gr.Accordion("Img2Img", open=False):

                    use_img2img = gr.Checkbox(
                        label="Use img2img",
                        value=saved_value('use_img2img', False)
                    )

                    use_ip = gr.Checkbox(
                        label="Send to Controlnet",
                        value=saved_value('use_ip', False)
                    )

                    denoising = gr.Slider(
                        value=saved_slider('denoising', 0.75, 0.05, 1),
                        label="Denoising",
                        minimum=0.05,
                        maximum=1.0,
                        step=0.05
                    )

                    use_last_img = gr.Checkbox(
                        label="Use last image as img2img",
                        value=saved_value('use_last_img', False)
                    )

                    crop_center = gr.Checkbox(
                        label="Crop Center",
                        value=saved_value('crop_center', False)
                    )

            use_cache = gr.Checkbox(
                label="Use cache",
                info="Reuse booru API replies for an hour (faster, but repeats the same pages)",
                value=saved_value('use_cache', True)
            )

            credential_outputs = [
                credentials_group,
                credentials_heading,
                api_key,
                user_id,
                credentials_status,
                clear_credentials_btn
            ]

            # .change (not .input) so the fields follow the booru however it
            # is set, and never keep one site's key while another is selected.
            booru.change(
                fn=self.load_booru_credentials,
                inputs=[booru],
                outputs=credential_outputs
            )

            clear_credentials_btn.click(
                fn=self.clear_booru_credentials,
                inputs=[booru],
                outputs=credential_outputs
            )

        # Forge Neo applies ui-config.json AFTER building extension controls.
        # Without this, old default values overwrite the last-generated state.
        # The Ranbooru-specific JSON is the source of truth for these controls.
        controls = [
            enabled,
            tags,
            booru,
            remove_bad_tags,
            max_pages,
            change_dash,
            same_prompt,
            fringe_benefits,
            remove_tags,
            use_img2img,
            denoising,
            use_last_img,
            change_background,
            change_color,
            shuffle_tags,
            post_id,
            limit_tags,
            max_tags,
            sorting_order,
            mature_rating,
            use_ip,
            crop_center,
            use_cache,
            only_poses,
            strip_clothing,
            names_all_boorus,
            skip_prompt_tags,
            api_key,
            user_id,
            save_credentials,
            credentials_status,
            clear_credentials_btn
        ]

        # Suppress Forge Neo's UiLoadsave override for EVERY Ranbooru control,
        # including the hidden checkbox AND visible accordion belonging to each
        # InputAccordion. This preserves saved settings after a full WebUI restart.
        for control in controls:
            control.do_not_save_to_config = True
            accordion = getattr(control, 'accordion', None)
            if accordion is not None:
                accordion.do_not_save_to_config = True

        # The UI is constructed only once when the Python WebUI starts.
        # On browser refresh/new tab, Gradio reuses that old server-side
        # component configuration, so loading JSON in ui() alone is not enough.
        # Register on the current txt2img/img2img Blocks: Gradio carries this
        # load event over when Forge Neo renders the nested interface.
        # Values are fetched from disk for EACH browser page load.
        from gradio.context import Context

        ui_locals = locals()
        state_controls = {
            name: ui_locals[name]
            for name in RANBOORU_STATE_FIELDS
        }
        state_names = tuple(state_controls)
        load_outputs = [state_controls[name] for name in state_names]
        load_outputs.append(enabled.accordion)
        load_outputs.extend(credential_outputs)

        def restore_browser_page():
            state = load_ranbooru_settings(is_img2img)
            if not state:
                updates = [gr.update() for _ in load_outputs]
                # The server-side UI stays alive when only the browser reloads.
                # Restore the current shared order even without a generated state.
                updates[state_names.index('booru')] = gr.update(choices=load_booru_order())
                return updates

            # The booru determines available rating options and visibility.
            # Refresh both of those together, rather than allowing a delayed
            # booru.change event to reset the restored rating to "All".
            current_booru = state.get('booru', selected_booru)
            if current_booru not in RATINGS:
                current_booru = selected_booru
            rating_choices = list(RATINGS[current_booru])
            results = []

            for name in state_names:
                widget = state_controls[name]
                if name not in state and name not in (
                    'booru', 'mature_rating', 'fringe_benefits'
                ):
                    results.append(gr.update())
                    continue

                value = state.get(name, getattr(widget, 'value', None))

                if name == 'booru':
                    results.append(gr.update(choices=load_booru_order(), value=current_booru))
                    continue

                if name == 'mature_rating':
                    value = value if value in rating_choices else 'All'
                    results.append(gr.update(choices=rating_choices, value=value))
                    continue

                if isinstance(widget, gr.Checkbox):
                    if type(value) is not bool:
                        value = bool(getattr(widget, 'value', False))
                elif isinstance(widget, gr.Slider):
                    if type(value) not in (int, float):
                        value = getattr(widget, 'value', widget.minimum)
                    value = max(widget.minimum, min(widget.maximum, value))
                elif isinstance(widget, (gr.Radio, gr.Dropdown)):
                    choices = getattr(widget, 'choices', [])
                    valid = [
                        item[1] if isinstance(item, (tuple, list)) and len(item) >= 2 else item
                        for item in choices
                    ]
                    if value not in valid:
                        value = getattr(widget, 'value', None)
                elif isinstance(widget, gr.Textbox):
                    if not isinstance(value, str):
                        value = getattr(widget, 'value', '') or ''

                if name == 'fringe_benefits':
                    results.append(gr.update(
                        value=value, visible=(current_booru == 'gelbooru')
                    ))
                else:
                    results.append(gr.update(value=value))

            enabled_on = state.get('enabled', enabled.value)
            if type(enabled_on) is not bool:
                enabled_on = bool(enabled.value)
            results.append(gr.update(open=enabled_on))
            # The credentials panel follows the restored booru.
            results.extend(self.booru_credential_updates(current_booru))
            print(f'[Ranbooru] Restored {len(state)} settings for '
                  f"{'img2img' if is_img2img else 'txt2img'} on browser load")
            return results

        root_blocks = Context.root_block
        if root_blocks is not None:
            root_blocks.load(
                fn=restore_browser_page,
                inputs=[],
                outputs=load_outputs,
                queue=False,
                show_progress=False,
                api_name=False,
            )
        else:
            print('[Ranbooru] Warning: Gradio Blocks context not found; '
                  'browser-load restore is unavailable.')

        return controls

    def check_orientation(self, img):
        """Check if image is portrait, landscape or square"""

        x, y = img.size

        if x / y > 1.2:
            return [768, 512]

        elif y / x > 1.2:
            return [512, 768]

        else:
            return [768, 768]

    @clear_ranbooru_progress_on_error
    def before_process(
        self,
        p,
        enabled,
        tags,
        booru,
        remove_bad_tags,
        max_pages,
        change_dash,
        same_prompt,
        fringe_benefits,
        remove_tags,
        use_img2img,
        denoising,
        use_last_img,
        change_background,
        change_color,
        shuffle_tags,
        post_id,
        limit_tags,
        max_tags,
        sorting_order,
        mature_rating,
        use_ip,
        crop_center,
        use_cache,
        only_poses,
        strip_clothing,
        names_all_boorus,
        skip_prompt_tags,
        api_key,
        user_id,
        save_credentials,
        credentials_status,
        clear_credentials_btn
    ):

        # Snapshot the actual submitted values, not partially edited browser state.
        # Saving before API queries also preserves inputs when the query fails.
        if enabled:
            save_ranbooru_settings(
                getattr(self, '_ranbooru_is_img2img', isinstance(p, StableDiffusionProcessingImg2Img)),
                locals()
            )

        # Use cache applies to Ranbooru's own booru API requests only.
        booru_http.set_cache(use_cache)

        if enabled:
            # MRU is updated only by a genuine Ranbooru generation attempt;
            # selecting a different site in the dropdown does not change order.
            mark_booru_used(booru)
            show_ranbooru_progress(f'Searching {booru}...')

            # Credentials of the selected booru only (saved ones win).
            booru_api_key, booru_user_id = resolve_booru_credentials(
                booru, api_key, user_id, save_credentials, credentials_manager
            )

            # Initialize APIs
            booru_apis = {
                'gelbooru': Gelbooru(
                    fringe_benefits,
                    booru_api_key if booru == 'gelbooru' else None,
                    booru_user_id if booru == 'gelbooru' else None
                ),

                'rule34': Rule34(
                    username=booru_user_id if booru == 'rule34' else None,
                    api_key=booru_api_key if booru == 'rule34' else None
                ),

                'safebooru': Safebooru(),
                'danbooru': Danbooru(),
                'konachan': Konachan(),
                'yande.re': Yandere(),
                'aibooru': AIBooru(),
                'xbooru': XBooru(),
                'e621': e621(
                    username=booru_user_id if booru == 'e621' else None,
                    api_key=booru_api_key if booru == 'e621' else None
                ),
            }

            self.original_prompt = p.prompt

            # Check compatibility
            check_exception(
                booru,
                {
                    'tags': tags,
                    'post_id': post_id
                }
            )

            # Manage Bad Tags
            bad_tags = []

            if remove_bad_tags:

                bad_tags = [
                    'mixed-language_text',
                    'watermark',
                    'text',
                    'english_text',
                    'speech_bubble',
                    'signature',
                    'artist_name',
                    'censored',
                    'bar_censor',
                    'translation',
                    'twitter_username',
                    "twitter_logo",
                    'patreon_username',
                    'commentary_request',
                    'tagme',
                    'commentary',
                    'character_name',
                    'mosaic_censoring',
                    'instagram_username',
                    'text_focus',
                    'english_commentary',
                    'comic',
                    'translation_request',
                    'fake_text',
                    'translated',
                    'paid_reward_available',
                    'thought_bubble',
                    'multiple_views',
                    'silent_comic',
                    'out-of-frame_censoring',
                    'symbol-only_commentary',
                    '3koma',
                    '2koma',
                    'character_watermark',
                    'spoken_question_mark',
                    'japanese_text',
                    'spanish_text',
                    'language_text',
                    'fanbox_username',
                    'commission',
                    'original',
                    'ai_generated',
                    'stable_diffusion',
                    'tagme_(artist)',
                    'text_bubble',
                    'qr_code',
                    'chinese_commentary',
                    'korean_text',
                    'partial_commentary',
                    'chinese_text',
                    'copyright_request',
                    'heart_censor',
                    'censored_nipples',
                    'page_number',
                    'scan',
                    'fake_magazine_cover',
                    'korean_commentary'
                ]

            bad_tags.extend(parse_tag_list(remove_tags))

            # Manage Backgrounds
            background_options = {
                'Add Background': (
                    'detailed_background,'
                    + random.choice(
                        [
                            "outdoors",
                            "indoors"
                        ]
                    ),

                    COLORED_BG
                ),

                'Remove Background': (
                    'plain_background,'
                    'simple_background,'
                    + random.choice(COLORED_BG),

                    ADD_BG
                ),

                'Remove All': (
                    '',
                    COLORED_BG + ADD_BG
                )
            }

            if change_background in background_options:

                prompt_addition, tags_to_remove = (
                    background_options[
                        change_background
                    ]
                )

                bad_tags.extend(tags_to_remove)

                p.prompt = (
                    f'{p.prompt},{prompt_addition}'
                    if p.prompt
                    else prompt_addition
                )

            # Manage Colors
            color_options = {
                'Colored': BW_BG,
                'Limited Palette': '(limited_palette:1.3)',
                'Monochrome': ','.join(BW_BG)
            }

            if change_color in color_options:

                color_option = color_options[
                    change_color
                ]

                if isinstance(color_option, list):
                    bad_tags.extend(color_option)

                else:
                    p.prompt = (
                        f'{p.prompt},{color_option}'
                        if p.prompt
                        else color_option
                    )

            if booru == 'danbooru':
                # Don't spend a scarce tag slot on -animated: the Danbooru
                # adapter filters it locally after retrieving the posts.
                from urllib.parse import quote_plus
                search_terms = [
                    tag.strip() for tag in tags.split(',') if tag.strip()
                ] if tags else []
                if mature_rating != 'All':
                    search_terms.append(
                        f'rating:{RATINGS[booru][mature_rating]}'
                    )
                add_tags = '&tags=' + quote_plus(' '.join(search_terms))
            else:
                # Preserve the existing request syntax for every other booru.
                add_tags = '&tags=-animated'
                if tags:
                    add_tags += f'+{tags.replace(",", "+")}'
                if mature_rating != 'All':
                    add_tags += f'+rating:{RATINGS[booru][mature_rating]}'

            # Getting Data
            random_post = {'preview_url': ''}

            prompts = []
            last_img = []
            preview_urls = []

            api_url = booru_apis.get(
                booru,
                Gelbooru(fringe_benefits)
            )

            print(f'Using {booru}')
            if booru in CREDENTIAL_BOORUS:
                auth_present = bool(booru_api_key and booru_user_id)
                print(
                    f'[Ranbooru] {booru} API credentials: '
                    + ('key and user ID supplied' if auth_present
                       else 'MISSING/INCOMPLETE (search is anonymous)')
                )

            # Manage Post ID
            if post_id:
                data = api_url.get_post(
                    add_tags,
                    max_pages,
                    post_id
                )

            else:
                data = api_url.get_data(
                    add_tags,
                    max_pages
                )

            # A randomly selected page may be beyond the last available page.
            # Retry the first page once before declaring the search empty.
            # Keep this in before_process so it works across all booru APIs.
            if not isinstance(data, dict):
                data = {'post': []}

            if not data.get('post') and not post_id and max_pages > 1:
                print('[Ranbooru] Empty API page; retrying the first page.')
                retry_data = api_url.get_data(add_tags, 1)
                if isinstance(retry_data, dict):
                    data = retry_data

            # Clean malformed posts and replace null scores.
            clean_posts = []
            raw_posts = data.get('post', [])
            if not isinstance(raw_posts, list):
                raw_posts = []

            for post in raw_posts:

                if not isinstance(post, dict):
                    print(
                        f"Skipping invalid post object: "
                        f"{post}"
                    )
                    continue

                post['score'] = post.get('score', 0)

                if 'tags' not in post:
                    post['tags'] = ''

                clean_posts.append(post)

            data['post'] = clean_posts

            # Artist/character/copyright tags reported by the booru, gathered
            # from the whole page so mixed prompts are covered too.
            pose_excluded_tags = set()
            if only_poses:
                for post in clean_posts:
                    pose_excluded_tags.update(post_category_tags(post))
            pose_names = pose_names_for(
                booru, only_poses, names_all_boorus, pose_name_tags
            )
            pose_qualifiers = pose_names_for(
                booru, only_poses, names_all_boorus, pose_name_qualifiers
            )

            if not clean_posts:
                raise ValueError(
                    '[Ranbooru] No posts were returned by the booru API for '
                    'these tags and rating. Check the tags, rating, post ID, '
                    'or API credentials.'
                )

            # COUNT is the site-reported result count where supported; for
            # other adapters it is the number of posts on the fetched page.
            # A post-ID lookup should always report the one requested post.
            results_count = len(clean_posts) if post_id else COUNT
            if type(results_count) is not int or results_count < len(clean_posts):
                results_count = len(clean_posts)
            show_ranbooru_progress(f'Processing {results_count} results')

            # Sort based on sorting_order
            if sorting_order == 'High Score':

                data['post'] = sorted(
                    data['post'],
                    key=lambda k: k.get('score', 0),
                    reverse=True
                )

            elif sorting_order == 'Low Score':

                data['post'] = sorted(
                    data['post'],
                    key=lambda k: k.get('score', 0)
                )

            if post_id:

                print(f'Using post ID: {post_id}')

                random_numbers = [
                    0
                    for _ in range(
                        0,
                        p.batch_size * p.n_iter
                    )
                ]

            else:
                # Select from the ACTUAL returned page, not the API's global
                # match count (which may be 201 even if this page has 1 post).
                random_numbers = self.random_number(
                    sorting_order,
                    p.batch_size * p.n_iter,
                    len(data['post'])
                )

            for random_number in random_numbers:

                if same_prompt:

                    random_post = data['post'][
                        random_numbers[0]
                    ]

                else:

                    random_post = data['post'][random_number]

                clean_tags = (
                    random_post['tags']
                    .replace('(', r'\(')
                    .replace(')', r'\)')
                )

                tag_list = [
                    t
                    for t in clean_tags.split(' ')
                    if t.strip()
                ]

                if shuffle_tags:
                    temp_tags = random.sample(
                        tag_list,
                        len(tag_list)
                    )

                else:
                    temp_tags = tag_list

                prompts.append(', '.join(tag.strip() for tag in temp_tags if tag.strip()))

                preview_urls.append(
                    random_post.get(
                        'file_url',
                        'https://pic.re/image'
                    )
                )

                # Debug picture
                if DEBUG:
                    print(random_post)

            # Get Images
            if use_img2img:

                image_urls = (
                    [random_post['file_url']]
                    if use_last_img
                    else preview_urls
                )

                for img in image_urls:

                    try:
                        response = requests.get(
                            img,
                            headers=api_url.headers,
                            timeout=20
                        )

                        response.raise_for_status()

                        image = Image.open(
                            BytesIO(response.content)
                        ).convert("RGB")

                        last_img.append(image)

                    except Exception as e:
                        print(
                            f"Failed to load image "
                            f"{img}: {e}"
                        )

                if not last_img:
                    raise RuntimeError(
                        f'[Ranbooru] Use img2img: none of the {len(image_urls)} '
                        f'booru image(s) could be downloaded (see the console '
                        f'for each error). Try again or turn off Use img2img.'
                    )

            # e621 posts carry no categories; look up the prompt's tags.
            e621_dropped_tags = set()
            if isinstance(api_url, e621):
                e621_categories = api_url.tag_categories({
                    _normalize_pose_tag(tag)
                    for prompt in prompts
                    for tag in html.unescape(prompt).split(',')
                    if tag.strip()
                })
                e621_dropped_tags = {
                    tag for tag, category in e621_categories.items()
                    if category in e621.DROP_CATEGORIES
                }
                if only_poses:
                    pose_excluded_tags.update(
                        tag for tag, category in e621_categories.items()
                        if category in e621.NAME_CATEGORIES
                    )

            new_prompts = []

            # Tags the user already typed (Skip tags already in prompt).
            typed_tag_keys = (
                prompt_tag_keys(self.original_prompt)
                if skip_prompt_tags else set()
            )

            # Cleaning Tags
            for prompt in prompts:

                prompt_tags = remove_listed_tags(
                    html.unescape(prompt).split(','),
                    bad_tags
                )

                if typed_tag_keys:
                    prompt_tags = [
                        tag for tag in prompt_tags
                        if prompt_tag_key(tag) not in typed_tag_keys
                    ]

                if e621_dropped_tags:
                    prompt_tags = [
                        tag for tag in prompt_tags
                        if _normalize_pose_tag(tag) not in e621_dropped_tags
                    ]

                prompt_tags = filter_pose_tags(
                    prompt_tags,
                    strip_traits=only_poses,
                    strip_clothing=strip_clothing,
                    excluded=pose_excluded_tags,
                    names=pose_names,
                    qualifiers=pose_qualifiers,
                    e621_vocab=isinstance(api_url, e621)
                )

                new_prompt = ', '.join(tag.strip() for tag in prompt_tags if tag.strip())

                if change_dash:
                    new_prompt = new_prompt.replace(
                        "_",
                        " "
                    )

                new_prompts.append(new_prompt)

            prompts = new_prompts

            if len(prompts) == 1:

                print('Processing Single Prompt')

                p.prompt = (
                    f"{p.prompt},{prompts[-1]}"
                    if p.prompt
                    else prompts[-1]
                )

            else:

                print('Processing Multiple Prompts')

                p.negative_prompt = [
                    p.negative_prompt
                    for _ in range(
                        0,
                        p.batch_size * p.n_iter
                    )
                ]

                p.prompt = (
                    prompts
                    if not p.prompt
                    else [
                        f"{p.prompt},{prompt}"
                        for prompt in prompts
                    ]
                )

            if use_img2img:

                if (
                    len(last_img)
                    < p.batch_size * p.n_iter
                ):

                    last_img = [
                        last_img[0]
                        for _ in range(
                            0,
                            p.batch_size * p.n_iter
                        )
                    ]
            if limit_tags < 1:

                if isinstance(p.prompt, list):

                    p.prompt = [
                        limit_prompt_tags(
                            pr,
                            limit_tags,
                            'Limit'
                        )
                        for pr in p.prompt
                    ]

                else:

                    p.prompt = limit_prompt_tags(
                        p.prompt,
                        limit_tags,
                        'Limit'
                    )

            if max_tags > 0:

                if isinstance(p.prompt, list):

                    p.prompt = [
                        limit_prompt_tags(
                            pr,
                            max_tags,
                            'Max'
                        )
                        for pr in p.prompt
                    ]

                else:

                    p.prompt = limit_prompt_tags(
                        p.prompt,
                        max_tags,
                        'Max'
                    )

            # Normalize prompt formatting
            if isinstance(p.prompt, list):
                p.prompt = [normalize_prompt(x) for x in p.prompt]
            else:
                p.prompt = normalize_prompt(p.prompt)

            if isinstance(p.negative_prompt, list):
                p.negative_prompt = [normalize_prompt(x) for x in p.negative_prompt]
            elif isinstance(p.negative_prompt, str):
                p.negative_prompt = normalize_prompt(p.negative_prompt)

            if use_img2img:

                if not use_ip:

                    self.real_steps = p.steps
                    p.steps = 1

                self.last_img = last_img

                if use_ip:
                    send_image_to_controlnet(p, last_img[0], denoising)

    @clear_ranbooru_progress_after_postprocess
    def postprocess(
        self,
        p,
        processed,
        enabled,
        tags,
        booru,
        remove_bad_tags,
        max_pages,
        change_dash,
        same_prompt,
        fringe_benefits,
        remove_tags,
        use_img2img,
        denoising,
        use_last_img,
        change_background,
        change_color,
        shuffle_tags,
        post_id,
        limit_tags,
        max_tags,
        sorting_order,
        mature_rating,
        use_ip,
        crop_center,
        use_cache,
        only_poses,
        strip_clothing,
        names_all_boorus,
        skip_prompt_tags,
        api_key,
        user_id,
        save_credentials,
        credentials_status,
        clear_credentials_btn
    ):

        if (
            use_img2img
            and not use_ip
            and enabled
        ):

            print('Using pictures')
            if crop_center:

                width, height = (
                    p.width,
                    p.height
                )

                self.last_img = [
                    resize_image(
                        img,
                        width,
                        height,
                        cropping=True
                    )
                    for img in self.last_img
                ]

            else:
                width, height = self.check_orientation(
                    self.last_img[0]
                )

            # Second pass: real img2img runs on the booru images. Copy the
            # generation settings that exist on this WebUI build (styles,
            # variation seed, distilled CFG, clip skip and other overrides...),
            # but not scripts, which would run Ranbooru again.
            # Forge Neo takes at most batch_size init images per run, so run
            # one img2img pass per batch, each with its own images, prompts
            # and seeds.
            batch_size = max(1, int(p.batch_size))
            n_iter = max(1, int(p.n_iter))
            total = batch_size * n_iter
            prompts = per_image_values(p.prompt, total)
            negative_prompts = per_image_values(p.negative_prompt, total)
            seeds = per_image_seeds(p, total)
            subseeds = per_image_seeds(p, total, 'subseed')
            init_images = per_image_values(self.last_img, total)

            base_settings = img2img_settings_from(
                p, StableDiffusionProcessingImg2Img
            )
            images, infotexts, first_image = [], [], 0
            for batch in range(n_iter):
                state = getattr(shared, 'state', None)
                if state is not None and (
                    getattr(state, 'interrupted', False)
                    or getattr(state, 'stopping_generation', False)
                ):
                    break

                part = slice(batch * batch_size, (batch + 1) * batch_size)
                settings = dict(base_settings)
                settings.update(
                    sd_model=shared.sd_model,
                    outpath_samples=(
                        shared.opts.outdir_samples
                        or shared.opts.outdir_img2img_samples
                    ),
                    outpath_grids=(
                        shared.opts.outdir_grids
                        or shared.opts.outdir_img2img_grids
                    ),
                    prompt=prompts[part],
                    negative_prompt=negative_prompts[part],
                    seed=seeds[part] if seeds else -1,
                    batch_size=batch_size,
                    n_iter=1,
                    steps=self.real_steps,
                    width=width,
                    height=height,
                    init_images=init_images[part],
                    denoising_strength=denoising,
                )
                if subseeds and 'subseed' in settings:
                    settings['subseed'] = subseeds[part]
                proc = process_images(
                    StableDiffusionProcessingImg2Img(**settings)
                )

                if n_iter == 1:
                    # One run: keep Forge's own grid, if it made one.
                    images = list(proc.images)
                    infotexts = list(proc.infotexts)
                    first_image = proc.index_of_first_image
                else:
                    # Several runs: their images only, no per-run grids.
                    start = proc.index_of_first_image or 0
                    images.extend(proc.images[start:])
                    infotexts.extend(proc.infotexts[start:])

            processed.images = images
            processed.infotexts = infotexts
            processed.index_of_first_image = first_image

            # Show the source booru image(s) after the results, each with the
            # infotext of the image it produced.
            sources = self.last_img[:1] if use_last_img else self.last_img
            for num, img in enumerate(sources[:len(images) - first_image]):
                processed.images.append(img)
                processed.infotexts.append(
                    source_infotext(infotexts, first_image, num)
                )

    def random_number(self, sorting_order, size, available_count):
        """Choose valid indices from the posts actually returned by the API.

        The API's total match count is NOT the length of the current page.
        Some searches return a partially filled final page, e.g. 201 total
        matches but just one post on the final page.
        """
        count = int(available_count)
        if count <= 0:
            raise ValueError('[Ranbooru] Cannot select from an empty post list.')
        if size <= 0:
            return []

        if sorting_order in ('High Score', 'Low Score'):
            weights = np.arange(count, 0, -1, dtype=float)
            weights /= weights.sum()
            return np.random.choice(
                np.arange(count), size=size, p=weights,
                replace=size > count
            ).tolist()

        if size > count:
            return random.choices(range(count), k=size)
        return random.sample(range(count), size)

