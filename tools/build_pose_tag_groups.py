"""Regenerate pose_tag_groups.json from Danbooru's tag group wiki pages.

Run manually when the lists need refreshing:

    python tools/build_pose_tag_groups.py

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
OUTPUT = Path(__file__).resolve().parents[1] / 'pose_tag_groups.json'
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


def fetch_body(group):
    request = Request(WIKI_URL.format(quote(f'tag_group:{group}')),
                      headers={'User-Agent': 'ranbooru-tag-group-builder'})
    with urlopen(request, timeout=30) as response:
        return json.load(response)['body']


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


def main():
    names = sorted(set(TRAIT_GROUPS) | set(CLOTHING_GROUPS) | set(KEEP_GROUPS))
    bodies = {}
    for name in names:
        bodies[name] = fetch_body(name)
        time.sleep(0.5)

    seen = {}
    traits = collect(TRAIT_GROUPS, bodies, seen) | EXTRA_TRAITS
    clothing = (collect(CLOTHING_GROUPS, bodies, seen) | EXTRA_CLOTHING) - NOT_CLOTHING
    # Curated extras are deliberate, so they override the keep groups.
    keep = ((collect(KEEP_GROUPS, bodies, seen) - EXTRA_TRAITS - EXTRA_CLOTHING)
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

    data = {
        'source': 'https://danbooru.donmai.us/wiki_pages/tag_groups',
        'generated': time.strftime('%Y-%m-%d'),
        'traits': sorted(t for t in traits - keep if not is_action(t)),
        'clothing': sorted(t for t in clothing - keep - traits if not is_action(t)),
        'keep': sorted(keep),
    }
    OUTPUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n',
                      encoding='utf-8')
    print(f"wrote {OUTPUT}: {len(data['traits'])} trait, "
          f"{len(data['clothing'])} clothing, {len(data['keep'])} keep tags")


if __name__ == '__main__':
    main()
