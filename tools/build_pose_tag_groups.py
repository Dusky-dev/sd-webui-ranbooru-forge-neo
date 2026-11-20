"""Regenerate the Only Poses tag lists from Danbooru.

Run manually when the lists need refreshing:

    python tools/build_pose_tag_groups.py          # both files
    python tools/build_pose_tag_groups.py groups   # pose_tag_groups.json only
    python tools/build_pose_tag_groups.py names    # pose_name_tags.json only
    python tools/build_pose_tag_groups.py e621     # e621_* keys only

pose_tag_groups.json comes from the tag group wiki pages; pose_name_tags.json
holds Danbooru's most-used character, copyright and artist names, used on
boorus whose posts carry no tag categories.

The extension never fetches these pages at runtime; it only reads the JSON.
Sections are chosen by heading so that pose/expression sections of mixed pages
(e.g. "Gazes" on the eyes page) are kept, not stripped.
"""
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

WIKI_URL = 'https://danbooru.donmai.us/wiki_pages/{}.json'
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'pose_tag_groups.json'
NAMES_OUTPUT = ROOT / 'pose_name_tags.json'
TAGS_URL = ('https://danbooru.donmai.us/tags.json?search[category]={category}'
            '&search[order]=count&limit=1000&page={page}&only=name')
# category id, name, how many of the most-used tags to take. Measured on
# recent Danbooru posts: 5k copyrights cover ~97% of copyright tags, 20k
# characters ~83%, 10k artists only ~23% (the artist tail is very long).
NAME_CATEGORIES = ((3, 'copyright', 5000), (4, 'character', 20000),
                   (1, 'artist', 10000))
# Short artist names are often ordinary words on other boorus (arf, eco).
MIN_ARTIST_LENGTH = 5
# Pose names that Danbooru files as copyrights but are pose tags in practice.
NAME_KEEP = {'saturday_night_fever'}

# name_(series) rule: a tag whose parenthesised qualifier is a series is a
# character or series item (jester_(fire_emblem)). Qualifiers are copyright
# names plus abbreviations used by at least MIN_QUALIFIER_USES characters
# (kancolle, fate, nikke), minus costume/role words such as (1st_costume).
MIN_QUALIFIER_USES = 8
QUALIFIER = re.compile(r'_\(([^()]+)\)')
QUALIFIER_IGNORE = {
    'casual', 'female', 'male', 'vtuber', 'character', 'summer', 'cookie',
    'new_year', 'young', 'school_uniform', 'timeskip', 'stand', 'human',
    'rider', 'lancer', 'assassin', 'caster', 'saber', 'archer', 'berserker',
    'ruler', 'avenger', 'dress', 'maid', 'halloween', 'cat', 'dog', 'adult',
    'racing', 'pajamas', 'summer_loungewear', 'armed', 'normal', 'tales',
    'league', 'child', 'old', 'cosplay', 'artist', 'real_life', 'original',
    'animal', 'object', 'food', 'weapon', 'shape', 'medium', 'pattern',
    'meme', 'symbol', 'company', 'band', 'game', 'series', 'anime', 'manga',
    'novel', 'film', 'movie', 'comic', 'creature', 'hair', 'clothes',
}
QUALIFIER_IGNORE_WORDS = ('costume', 'ascension', 'swimsuit', 'outfit', 'uniform')
ALL = None  # every section except the SKIPPED ones

SKIPPED = {'see also', 'related tag groups', 'about', 'external links', 'unrelated'}

# group -> sections to take (ALL = every section not in SKIPPED)
TRAIT_GROUPS = {
    'hair_color': ALL,
    'hair_styles': ALL,
    'eyes_tags': [
        'iris', 'individual colors of the iris', 'multiple colors of the iris',
        'form of the iris', 'pupils', 'colors of the pupils', 'form of the pupils',
        'sclera', 'more appearance', 'animal or inhuman eyes',
        'series specific eyes', 'number of eyes',
    ],
    'skin_color': ['natural colors', 'unnatural colors', 'alternate materials'],
    'breasts_tags': ['breast sizes', 'ranges of breast sizes'],
    'ears_tags': ['animal ears', 'other ears', 'number of ears'],
    'wings': [
        'number of wings', 'size', 'types', 'insect wings', 'elemental wings',
        'artificial wings', 'location (body)', 'misc. attributes',
        'media-specific', 'color variations', 'colors (solid)',
    ],
}

CLOTHING_GROUPS = {
    'attire': [
        'headwear and headgear', 'shirts and topwear', 'pants and bottomwear',
        'legs and feet', 'shoes and footwear', 'uniforms and costumes',
        'swimsuits and bodysuits', 'traditional clothing',
        'jewelry and accessories', 'head and face', 'neck and shoulders',
        'limbs', 'torso and misc',
    ],
    'headwear': ['types', 'colours', 'hats', 'with visor', 'with brim',
                 'with ear flaps', 'brimless', 'misc', 'helmets', 'crowns',
                 'other', 'non-specific styles', 'accessories'],
    'legwear': ['main tags / legwear by height', 'additional detail tags',
                'specific color', 'multiple colors', 'pattern', 'bands', 'style'],
    'handwear': ['gloves', 'glove prints', 'glove accessories', 'glove trims',
                 'asymmetrical gloves', 'glove materials', 'glove colors',
                 'other handwear', 'hand accessories', 'rings'],
    'accessories': ALL,
    'eyewear': ['main', 'eyewear', 'frame colors', 'lens colors', 'types',
                'accessories', 'glasses', 'sunglasses'],
    'makeup': ALL,
    'sleeves': ['colors', 'patterns', 'length', 'trim', 'type',
                'sleeveless clothing', 'clothing with abnormal sleeves'],
    'neck_and_neckwear': [
        'detached neckwear', 'chokers and standalone collars', 'other standalone',
        'shirt and dress collar designs', 'accessories', 'ties',
        'misc. neckwear', 'collared clothes', 'neckwear types', 'ascots',
        'bowties', 'chokers', 'neckerchiefs', 'necklaces', 'neckties', 'scarves',
    ],
}

# Whole groups that are never stripped; they win over any overlap.
KEEP_GROUPS = {
    'posture': ALL, 'gestures': ALL, 'hands': ALL, 'sexual_positions': ALL,
    'focus_tags': ALL, 'image_composition': ALL, 'character_count': ALL,
    'face_tags': ALL,
    'eyes_tags': ['stylistic eyes, for emotion or comedy', 'emotions and expressions',
                  'one or two eyes closed', 'gazes', 'misc / actions'],
    'headwear': ['actions / states'], 'legwear': ['action'],
    'handwear': ['glove interactions'], 'eyewear': ['actions'],
    'sleeves': ['actions'], 'wings': ['actions'],
    'neck_and_neckwear': ['actions / posture', 'neckwear actions'],
}

# Bare body-part words: too generic to strip on their own.
NEVER_STRIP = {'hair', 'eyes', 'eye', 'ears', 'ear', 'skin', 'breasts',
               'hands', 'face', 'head', 'neck', 'tag_groups'}

# Common characteristics the tag group pages do not list (checked against
# Danbooru's 1000 most-used general tags).
EXTRA_TRAITS = {
    'tail', 'multiple_tails', 'ponytail', 'braid', 'hair_bun', 'horns',
    'wings', 'halo', 'fang', 'fangs', 'skin_fang', 'sharp_teeth', 'claws',
    'fins', 'mole', 'mole_under_eye', 'mole_under_mouth', 'mole_on_breast',
    'beauty_mark', 'animal_ear_fluff', 'eyelashes', 'colored_eyelashes',
    'thick_eyebrows', 'colored_sclera', 'bright_pupils', 'glowing_eyes',
    'tsurime', 'tareme', 'muscular', 'muscular_male', 'muscular_female',
    'abs', 'pectorals', 'large_pectorals', 'thick_thighs', 'wide_hips',
    'curvy', 'toned', 'plump', 'petite', 'elf', 'scar', 'scar_on_face',
    'scar_across_eye', 'facial_hair', 'beard', 'mustache', 'stubble',
    'facial_mark', 'whisker_markings', 'freckles', 'tattoo', 'furry',
    'furry_female', 'furry_male', 'kemonomimi_mode', 'monster_girl',
    'short_eyebrows', 'colored_extremities',
}
EXTRA_CLOTHING = {
    'jewelry', 'underwear', 'panties', 'bra', 'lingerie', 'thong',
    'sleeveless', 'strapless', 'off_shoulder', 'highleg', 'headgear',
    'zettai_ryouiki', 'cleavage_cutout', 'clothing_cutout', 'fishnets',
    'nail_polish', 'makeup', 'lipstick', 'eyeshadow', 'skirt_set',
    'jersey', 'baseball_jersey', 'bead_bracelet', 'biker_clothes',
}
# Held props and body parts the attire pages happen to list.
NOT_CLOTHING = {'umbrella', 'hand_fan', 'fingernails', 'long_fingernails',
                'toenails', 'book_on_head', 'bucket_on_head', 'food_on_head',
                'object_on_head', 'traffic_cone_on_head'}


def is_action(tag):
    """Actions listed on appearance/attire pages (holding_compact, hair_in_own_mouth)."""
    return tag.startswith(('holding_', 'adjusting_')) or '_in_own_' in tag

LINK = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]')
HEADING = re.compile(r'^h\d(?:#[\w-]+)?\.\s*(.+)$')


def normalize(title):
    title = title.strip().lower()
    return '_'.join(title.split())


def heading_name(raw):
    """'[[Helmet]]s' -> 'helmets', '[[scarf|Scarves]]' -> 'scarves'."""
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]*)\]\]', r'\2', raw)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    return ' '.join(text.strip().lower().split())


E621_WIKI_URL = 'https://e621.net/wiki_pages/{}.json'
E621_USER_AGENT = 'Ranbooru-Forge-Neo/1.0 (tag prompt extension)'

# e621 uses its own vocabulary (tufts, markings, pawpads), so it gets its own
# lists, applied only to e621 posts. Bare body parts (tongue, feet, belly) are
# never stripped: they often carry the pose.
E621_TRAIT_GROUPS = {
    'anatomy': [
        'per properties', 'color type description', 'skin color patterns',
        'hair', 'head and neck', 'ears', 'eyes', 'nose', 'mouth', 'lips',
        'teeth', 'torso', 'breasts', 'alternatives', 'arms', 'wings', 'shape',
        'feet', 'tail', 'buttocks', 'other',
    ],
    'weight': ['body weight', 'body parts'],
    'markings': ['colors', 'by location', 'by body composition', 'other',
                 'scars', 'tattoos'],
}
E621_KEEP_GROUPS = {'pose': ALL, 'facial_expressions': ALL, 'touching': ALL}
E621_NEVER_STRIP = {
    'body', 'fur', 'scales', 'feathers', 'skin', 'hair', 'head', 'eyes',
    'ears', 'nose', 'mouth', 'lips', 'tongue', 'teeth', 'torso', 'belly',
    'navel', 'collarbone', 'breasts', 'nipples', 'arms', 'fingers', 'legs',
    'feet', 'toes', 'butt', 'human', 'insect', 'reptile', 'dragon',
    'dolphin', 'amphibian', 'crustacean', 'bare_chest', 'skindentation',
    'wet_body', 'glistening_body', 'tears', 'eyebrows',
}
# Common e621 body tags the group pages miss (from a live leak check).
E621_EXTRA_TRAITS = {
    'tuft', 'head_tuft', 'cheek_tuft', 'facial_tuft', 'neck_tuft',
    'chest_tuft', 'arm_tuft', 'shoulder_tuft', 'leg_tuft', 'tail_tuft',
    'elbow_tufts', 'inner_ear_fluff', 'ear_tuft', 'fluffy', 'fluffy_tail',
    'fluffy_ears', 'dipstick_tail', 'dipstick_ears', 'snout', 'muzzle',
    'black_nose', 'pink_nose', 'brown_nose', 'red_nose', 'whiskers',
    'pawpads', 'pink_pawpads', 'black_pawpads', 'markings', 'tail_markings',
    'countershading', 'stripes', 'spots', 'athletic', 'athletic_male',
    'athletic_female', 'athletic_anthro', 'muscular_anthro', 'muscular_male',
    'muscular_female', 'curvy_figure', 'voluptuous', 'bottom_heavy',
    'pear-shaped_figure', 'hourglass_figure', 'overweight_anthro',
    'huge_thighs', 'thick_thighs', 'wide_hips', 'big_belly', 'biceps',
    'abs', 'pecs', 'big_pecs', '2_horns', '3_fingers', '4_fingers',
    '5_fingers', '3_toes', '4_toes', '5_toes', 'mane', 'hooves', 'beak',
    'tail_feathers', 'head_crest', 'frill_(anatomy)', 'fin', 'fins',
    'hair_over_eye',
}
E621_EXTRA_TRAITS |= {
    f'{tone}_{part}'
    for tone in ('two_tone', 'multicolored', 'multi_tone', 'gradient', 'tricolor')
    for part in ('body', 'fur', 'scales', 'feathers', 'skin', 'hair', 'tail', 'ears')
}

E621_LINK = re.compile(r'\[\[([^\]|#][^\]|]*)(?:\|[^\]]*)?\]\]')
E621_HEADING = re.compile(r'^h\d\.\s*(.+)$')


def e621_heading_name(raw):
    """'[#anchor] [[Hair]]:' / '[b]Primary:[/b] [[#top|^]]' -> plain heading."""
    text = re.sub(r'\[\[#[^\]]*\]\]', '', raw)
    text = re.sub(r'\[#[^\]]*\]', '', text)
    text = re.sub(r'\[/?b\]', '', text)
    text = heading_name(text)
    return text.strip().rstrip(':^').strip()


def e621_sections(body):
    current, tags = '', []
    for line in body.splitlines():
        match = E621_HEADING.match(line.strip())
        if match:
            yield current, tags
            current, tags = e621_heading_name(match.group(1)), []
        for title in E621_LINK.findall(line):
            if ':' in title:
                continue
            tags.append(normalize(title))
    yield current, tags


def build_e621():
    bodies = {}
    for group in sorted(set(E621_TRAIT_GROUPS) | set(E621_KEEP_GROUPS)):
        bodies[group] = fetch_json(E621_WIKI_URL.format(quote(f'tag_group:{group}')),
                                   E621_USER_AGENT)['body']
        time.sleep(0.6)

    def pick(config):
        found = set()
        for group, wanted in config.items():
            seen = set()
            for heading, tags in e621_sections(bodies[group]):
                seen.add(heading)
                if heading in SKIPPED:
                    continue
                if wanted is ALL or heading in wanted:
                    found.update(tags)
            if wanted is not ALL:
                for missing in sorted(set(wanted) - seen):
                    print(f'warning: e621 {group}: no section {missing!r}', file=sys.stderr)
        return found

    keep = (pick(E621_KEEP_GROUPS) - E621_EXTRA_TRAITS) | E621_NEVER_STRIP
    traits = pick(E621_TRAIT_GROUPS) | E621_EXTRA_TRAITS
    overlap = sorted(traits & keep)
    if overlap:
        print('e621 traits kept because a keep group lists them: '
              + ', '.join(overlap), file=sys.stderr)

    data = json.loads(OUTPUT.read_text(encoding='utf-8'))
    data['e621_source'] = 'https://e621.net/wiki_pages/tag_groups'
    data['e621_traits'] = sorted(t for t in traits - keep if not is_action(t))
    data['e621_keep'] = sorted(keep)
    OUTPUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n',
                      encoding='utf-8')
    print(f"wrote e621 lists to {OUTPUT}: {len(data['e621_traits'])} trait, "
          f"{len(data['e621_keep'])} keep tags")


def fetch_json(url, user_agent='ranbooru-tag-group-builder'):
    request = Request(url, headers={'User-Agent': user_agent})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_body(group):
    return fetch_json(WIKI_URL.format(quote(f'tag_group:{group}')))['body']


def load_names():
    """Names from pose_name_tags.json, so the keep set can exclude them."""
    try:
        data = json.loads(NAMES_OUTPUT.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return set()
    return {tag for key in ('copyright', 'character', 'artist')
            for tag in data.get(key, [])}


def build_names():
    data = {'source': 'https://danbooru.donmai.us/tags', 'generated': time.strftime('%Y-%m-%d')}
    for category, name, count in NAME_CATEGORIES:
        tags = []
        for page in range(1, count // 1000 + 1):
            tags.extend(row['name'] for row in fetch_json(
                TAGS_URL.format(category=category, page=page)))
            time.sleep(0.5)
        if name == 'artist':
            tags = [tag for tag in tags if len(tag) >= MIN_ARTIST_LENGTH]
        data[name] = sorted(set(tags) - NAME_KEEP)
    uses = {}
    for tag in data['character']:
        for qualifier in QUALIFIER.findall(tag):
            uses[qualifier] = uses.get(qualifier, 0) + 1
    qualifiers = set(data['copyright']) | {
        q for q, count in uses.items() if count >= MIN_QUALIFIER_USES
    }
    data['qualifiers'] = sorted(
        q for q in qualifiers - QUALIFIER_IGNORE
        if not any(word in q for word in QUALIFIER_IGNORE_WORDS)
    )
    NAMES_OUTPUT.write_text(json.dumps(data, indent=0, ensure_ascii=False) + '\n',
                            encoding='utf-8')
    print(f'wrote {NAMES_OUTPUT}: ' + ', '.join(
        f'{len(data[name])} {name}' for _, name, _ in NAME_CATEGORIES)
        + f", {len(data['qualifiers'])} series qualifiers")


def sections(body):
    """Yield (heading, tags) pairs in page order."""
    current, tags = '', []
    for line in body.splitlines():
        match = HEADING.match(line.strip())
        if match:
            yield current, tags
            current, tags = heading_name(match.group(1)), []
        for title in LINK.findall(line):
            if ':' in title or title.lower().startswith('list of'):
                continue
            tags.append(normalize(title))
    yield current, tags


def collect(groups, bodies, seen_sections):
    found = set()
    for group, wanted in groups.items():
        for heading, tags in sections(bodies[group]):
            seen_sections.setdefault(group, set()).add(heading)
            if heading in SKIPPED:
                continue
            if wanted is ALL or heading in wanted:
                found.update(tags)
    return found


def build_groups():
    names = sorted(set(TRAIT_GROUPS) | set(CLOTHING_GROUPS) | set(KEEP_GROUPS))
    bodies = {}
    for name in names:
        bodies[name] = fetch_body(name)
        time.sleep(0.5)

    seen = {}
    traits = collect(TRAIT_GROUPS, bodies, seen) | EXTRA_TRAITS
    clothing = (collect(CLOTHING_GROUPS, bodies, seen) | EXTRA_CLOTHING) - NOT_CLOTHING
    # Curated extras are deliberate, so they override the keep groups.
    # Wiki pages use character names as examples; those are not pose tags.
    keep = ((collect(KEEP_GROUPS, bodies, seen) - EXTRA_TRAITS - EXTRA_CLOTHING
             - (load_names() - NAME_KEEP))
            | NEVER_STRIP)

    # Report section names that were asked for but not found (wiki edits).
    for config in (TRAIT_GROUPS, CLOTHING_GROUPS, KEEP_GROUPS):
        for group, wanted in config.items():
            if wanted is not ALL:
                for missing in sorted(set(wanted) - seen[group]):
                    print(f'warning: {group}: no section {missing!r}', file=sys.stderr)

    for label, tags in (('traits', traits), ('clothing', clothing)):
        overlap = sorted(tags & keep)
        if overlap:
            print(f'{label} tags kept because a keep group lists them: '
                  + ', '.join(overlap), file=sys.stderr)

    try:
        previous = json.loads(OUTPUT.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        previous = {}
    data = {
        'source': 'https://danbooru.donmai.us/wiki_pages/tag_groups',
        'generated': time.strftime('%Y-%m-%d'),
        'traits': sorted(t for t in traits - keep if not is_action(t)),
        'clothing': sorted(t for t in clothing - keep - traits if not is_action(t)),
        'keep': sorted(keep),
    }
    # The e621 lists are rebuilt separately (target 'e621'); keep them.
    data.update({k: v for k, v in previous.items() if k.startswith('e621_')})
    OUTPUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n',
                      encoding='utf-8')
    print(f"wrote {OUTPUT}: {len(data['traits'])} trait, "
          f"{len(data['clothing'])} clothing, {len(data['keep'])} keep tags")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if target not in ('all', 'groups', 'names', 'e621'):
        sys.exit('usage: build_pose_tag_groups.py [all|groups|names|e621]')
    # Names first: the keep set is built without them.
    if target in ('all', 'names'):
        build_names()
    if target in ('all', 'groups'):
        build_groups()
    if target in ('all', 'e621'):
        build_e621()


if __name__ == '__main__':
    main()
