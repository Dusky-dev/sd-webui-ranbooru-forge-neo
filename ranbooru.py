from io import BytesIO
import html
import random
import requests
import modules.scripts as scripts
import gradio as gr
import os
from PIL import Image
import numpy as np
import importlib
import requests_cache
import json
import tempfile
import threading

from modules.processing import (
    process_images,
    StableDiffusionProcessingImg2Img
)

from modules import shared
from modules import deepbooru
from modules.ui_components import InputAccordion

extension_root = scripts.basedir()

user_data_dir = os.path.join(extension_root, 'user')
user_search_dir = os.path.join(user_data_dir, 'search')
user_remove_dir = os.path.join(user_data_dir, 'remove')
user_credentials_dir = os.path.join(user_data_dir, 'credentials')

os.makedirs(user_search_dir, exist_ok=True)
os.makedirs(user_remove_dir, exist_ok=True)
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

# Persist user-facing settings only when generating with Ranbooru or LoRAnado enabled.
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
    'mix_prompt',
    'mix_amount',
    'chaos_mode',
    'negative_mode',
    'chaos_amount',
    'limit_tags',
    'max_tags',
    'sorting_order',
    'mature_rating',
    'lora_folder',
    'lora_amount',
    'lora_min',
    'lora_max',
    'lora_enabled',
    'lora_custom_weights',
    'lora_lock_prev',
    'use_ip',
    'use_search_txt',
    'use_remove_txt',
    'choose_search_txt',
    'choose_remove_txt',
    'crop_center',
    'use_deepbooru',
    'type_deepbooru',
    'use_same_seed',
    'use_cache',
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



if not os.path.isfile(
    os.path.join(user_search_dir, 'tags_search.txt')
):
    with open(
        os.path.join(user_search_dir, 'tags_search.txt'),
        'w'
    ):
        pass

if not os.path.isfile(
    os.path.join(user_remove_dir, 'tags_remove.txt')
):
    with open(
        os.path.join(user_remove_dir, 'tags_remove.txt'),
        'w'
    ):
        pass

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


def get_available_ratings(booru):
    mature_ratings = gr.Radio.update(
        choices=RATINGS[booru].keys(),
        value="All"
    )

    return mature_ratings


def show_fringe_benefits(booru):
    if booru == 'gelbooru':
        return gr.Checkbox.update(visible=True)

    else:
        return gr.Checkbox.update(visible=False)


def check_exception(booru, parameters):
    post_id = parameters.get('post_id')
    tags = parameters.get('tags')

    if booru == 'konachan' and post_id:
        raise Exception("Konachan does not support post IDs")

    if booru == 'yande.re' and post_id:
        raise Exception("Yande.re does not support post IDs")

    if booru == 'e621' and post_id:
        raise Exception("e621 does not support post IDs")

    if booru == 'danbooru' and tags and len(tags.split(',')) > 1:
        raise Exception(
            "Danbooru does not support multiple tags. "
            "You can have only one tag."
        )


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
    def __init__(
        self,
        fringe_benefits,
        api_key=None,
        user_id=None
    ):
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

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        loop_msg = True

        # run loop at most twice
        for loop in range(2):

            if id:
                add_tags = ''

            # Build the API URL with credentials if available
            api_params = (
                f"&pid={random.randint(0, max_pages - 1)}"
                f"{id}{add_tags}"
            )

            if self.api_key and self.user_id:
                api_params += (
                    f"&api_key={self.api_key}"
                    f"&user_id={self.user_id}"
                )

            url = f"{self.booru_url}{api_params}"

            self.last_url = url

            # The randint function is an alias to randrange(a, b+1),
            # so 'max_pages' should be passed as 'max_pages-1'
            if self.fringeBenefits:
                res = requests.get(
                    url,
                    cookies={'fringeBenefits': 'yup'}
                )

            else:
                res = requests.get(url)

            data = res.json()

            COUNT = data['@attributes']['count']

            if COUNT <= max_pages * POST_AMOUNT:
                max_pages = COUNT // POST_AMOUNT + 1

                # If max_pages is bigger than available pages,
                # loop the function with updated max_pages
                # based on the value of COUNT

                while loop_msg:
                    print(f" Processing {COUNT} results.")
                    loop_msg = False
                    continue

            else:
                print(
                    f" Processing "
                    f"{max_pages * POST_AMOUNT} "
                    f"out of {COUNT} results."
                )

                break

        return data

    def get_post(self, add_tags, max_pages=10, id=''):
        return self.get_data(
            add_tags,
            max_pages,
            "&id=" + id
        )
class XBooru(Booru):
    def __init__(self):
        super().__init__(
            'xbooru',
            (
                'https://xbooru.com/index.php'
                '?page=dapi&s=post&q=index'
                f'&json=1&limit={POST_AMOUNT}'
            )
        )

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        loop_msg = True

        # run loop at most twice
        for loop in range(2):

            if id:
                add_tags = ''

            url = (
                f"{self.booru_url}"
                f"&pid={random.randint(0, max_pages - 1)}"
                f"{id}{add_tags}"
            )

            print(url)

            self.last_url = url

            res = requests.get(url)
            data = res.json()

            COUNT = 0

            for post in data:
                post['file_url'] = (
                    f"https://xbooru.com/images/"
                    f"{post['directory']}/{post['image']}"
                )

                COUNT += 1

            if COUNT <= max_pages * POST_AMOUNT:
                max_pages = COUNT // POST_AMOUNT + 1

                # If max_pages is bigger than available pages,
                # loop the function with updated max_pages
                # based on the value of COUNT
                while loop_msg:
                    print(f" Processing {COUNT} results.")
                    loop_msg = False
                    continue

            else:
                print(
                    f" Processing "
                    f"{max_pages * POST_AMOUNT} "
                    f"out of {COUNT} results."
                )

                break

        return {'post': data}

    def get_post(self, add_tags, max_pages=10, id=''):
        return self.get_data(
            add_tags,
            max_pages,
            "&id=" + id
        )


class Rule34(Booru):
    def __init__(self, username=None, api_key=None):
        super().__init__(
            'rule34',
            (
                'https://api.rule34.xxx/index.php'
                '?page=dapi&s=post&q=index'
                f'&json=1&limit={POST_AMOUNT}'
            )
        )

        self.username = username
        self.api_key = api_key

        self.headers = {
            "User-Agent": "Ranbooru/1.2"
        }

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        loop_msg = True

        for loop in range(2):

            if id:
                add_tags = ''

            api_params = (
                f"&pid={random.randint(0, max_pages - 1)}"
                f"{id}{add_tags}"
            )

            # Add Rule34 credentials if available
            if self.api_key and self.username:
                api_params += (
                    f"&api_key={self.api_key}"
                    f"&user_id={self.username}"
                )

            url = f"{self.booru_url}{api_params}"

            self.last_url = url

            print(f"Rule34 URL: {url}")

            try:
                res = requests.get(
                    url,
                    headers=self.headers,
                    timeout=20
                )

                res.raise_for_status()

            except requests.exceptions.RequestException as e:
                print(f"Rule34 request failed: {e}")
                return {'post': []}

            response_text = res.text.strip()

            if "Missing authentication" in response_text:
                print("Rule34 authentication required")
                return {'post': []}

            try:
                data = res.json()

            except Exception as e:
                print(f"Rule34 returned invalid JSON: {e}")
                print(response_text[:500])
                return {'post': []}

            if isinstance(data, str):
                print(f"Rule34 returned string response: {data}")
                return {'post': []}

            if data is None:
                print("Rule34 returned null response")
                return {'post': []}

            if not isinstance(data, list):
                print(
                    f"Unexpected Rule34 response type: "
                    f"{type(data)}"
                )

                print(data)
                return {'post': []}

            clean_posts = []

            for post in data:

                if not isinstance(post, dict):
                    continue

                post['score'] = post.get('score', 0)

                if 'tags' not in post:
                    post['tags'] = ''

                if 'file_url' not in post:
                    continue

                clean_posts.append(post)

            data = clean_posts

            COUNT = len(data)

            if COUNT == 0:
                max_pages = 2

                while loop_msg:
                    print(f"Processing {COUNT} results.")
                    loop_msg = False
                    continue

            else:
                print(f"Found {COUNT} valid results")
                break

        return {'post': data}

    def get_post(self, add_tags, max_pages=10, id=''):
        return self.get_data(
            add_tags,
            max_pages,
            "&id=" + id
        )


class Safebooru(Booru):
    def __init__(self):
        super().__init__(
            'safebooru',
            (
                'https://safebooru.org/index.php'
                '?page=dapi&s=post&q=index'
                f'&json=1&limit={POST_AMOUNT}'
            )
        )

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        loop_msg = True

        # run loop at most twice
        for loop in range(2):

            if id:
                add_tags = ''

            url = (
                f"{self.booru_url}"
                f"&pid={random.randint(0, max_pages - 1)}"
                f"{id}{add_tags}"
            )

            self.last_url = url

            res = requests.get(url)
            data = res.json()

            COUNT = 0

            for post in data:
                post['file_url'] = (
                    f"https://safebooru.org/images/"
                    f"{post['directory']}/{post['image']}"
                )

                COUNT += 1

            if COUNT <= max_pages * POST_AMOUNT:
                max_pages = COUNT // POST_AMOUNT + 1

                # If max_pages is bigger than available pages,
                # loop the function with updated max_pages
                # based on the value of COUNT
                while loop_msg:
                    print(f" Processing {COUNT} results.")
                    loop_msg = False
                    continue

            else:
                print(
                    f" Processing "
                    f"{max_pages * POST_AMOUNT} "
                    f"out of {COUNT} results."
                )

                break

        return {'post': data}

    def get_post(self, add_tags, max_pages=10, id=''):
        return self.get_data(
            add_tags,
            max_pages,
            "&id=" + id
        )
class Konachan(Booru):
    def __init__(self):
        super().__init__(
            'konachan',
            f'https://konachan.com/post.json?limit={POST_AMOUNT}'
        )

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        loop_msg = True

        # run loop at most twice
        for loop in range(2):

            if id:
                add_tags = ''

            url = (
                f"{self.booru_url}"
                f"&page={random.randint(0, max_pages - 1)}"
                f"{id}{add_tags}"
            )

            self.last_url = url

            res = requests.get(url)
            data = res.json()

            COUNT = len(data)

            if COUNT == 0:
                max_pages = 2

                # Konachan does not have a way to know
                # the amount of results available in the search,
                # so we need to run the function again
                # with a fixed amount of pages

                while loop_msg:
                    print(f" Processing {COUNT} results.")
                    loop_msg = False
                    continue

            else:
                print("Found enough results")
                break

        return {'post': data}

    def get_post(self, add_tags, max_pages=10, id=''):
        raise Exception("Konachan does not support post IDs")


class Yandere(Booru):
    def __init__(self):
        super().__init__(
            'yande.re',
            f'https://yande.re/post.json?limit={POST_AMOUNT}'
        )

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        loop_msg = True

        # run loop at most twice
        for loop in range(2):

            if id:
                add_tags = ''

            url = (
                f"{self.booru_url}"
                f"&page={random.randint(0, max_pages - 1)}"
                f"{id}{add_tags}"
            )

            self.last_url = url

            res = requests.get(url)
            data = res.json()

            COUNT = len(data)

            if COUNT == 0:
                max_pages = 2

                # Yandere does not have a way to know
                # the amount of results available in the search,
                # so we need to run the function again
                # with a fixed amount of pages

                while loop_msg:
                    print(f" Processing {COUNT} results.")
                    loop_msg = False
                    continue

            else:
                print("Found enough results")
                break

        return {'post': data}

    def get_post(self, add_tags, max_pages=10, id=''):
        raise Exception("Yande.re does not support post IDs")


class AIBooru(Booru):
    def __init__(self):
        super().__init__(
            'AIBooru',
            f'https://aibooru.online/posts.json?limit={POST_AMOUNT}'
        )

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        loop_msg = True

        # run loop at most twice
        for loop in range(2):

            if id:
                add_tags = ''

            url = (
                f"{self.booru_url}"
                f"&page={random.randint(0, max_pages - 1)}"
                f"{id}{add_tags}"
            )

            self.last_url = url

            res = requests.get(url)
            data = res.json()

            for post in data:
                post['tags'] = post['tag_string']

            COUNT = len(data)

            if COUNT == 0:
                max_pages = 2

                # AIBooru does not have a way to know
                # the amount of results available in the search,
                # so we need to run the function again
                # with a fixed amount of pages

                while loop_msg:
                    print(f" Processing {COUNT} results.")
                    loop_msg = False
                    continue

            else:
                print("Found enough results")
                break

        return {'post': data}

    def get_post(self, add_tags, max_pages=10, id=''):
        raise Exception("AIBooru does not support post IDs")


class Danbooru(Booru):
    def __init__(self):
        super().__init__(
            'danbooru',
            (
                'https://danbooru.donmai.us/posts.json'
                f'?limit={POST_AMOUNT}'
            )
        )

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        loop_msg = True

        # run loop at most twice
        for loop in range(2):

            if id:
                add_tags = ''

            url = (
                f"{self.booru_url}"
                f"&page={random.randint(0, max_pages - 1)}"
                f"{id}{add_tags}"
            )

            self.last_url = url

            res = requests.get(
                url,
                headers=self.headers
            )

            data = res.json()

            for post in data:
                post['tags'] = post['tag_string']

            COUNT = len(data)

            if COUNT == 0:
                max_pages = 2

                # Danbooru does not have a way to know
                # the amount of results available in the search,
                # so we need to run the function again
                # with a fixed amount of pages

                while loop_msg:
                    print(f" Processing {COUNT} results.")
                    loop_msg = False
                    continue

            else:
                print("Found enough results")
                break

        return {'post': data}

    def get_post(self, add_tags, max_pages=10, id=''):
        url = f"https://danbooru.donmai.us/posts/{id}.json"

        self.last_url = url

        res = requests.get(
            url,
            headers=self.headers
        )

        data = res.json()

        data['tags'] = data['tag_string']
        data = {'post': [data]}

        return data


class e621(Booru):
    def __init__(self):
        super().__init__(
            'danbooru',
            f'https://e621.net/posts.json?limit={POST_AMOUNT}'
        )

    def get_data(self, add_tags, max_pages=10, id=''):
        global COUNT

        loop_msg = True

        # run loop at most twice
        for loop in range(2):

            if id:
                add_tags = ''

            url = (
                f"{self.booru_url}"
                f"&page={random.randint(0, max_pages - 1)}"
                f"{id}{add_tags}"
            )

            self.last_url = url

            res = requests.get(
                url,
                headers=self.headers
            )

            data = res.json()['posts']

            COUNT = len(data)

            for post in data:
                temp_tags = []

                sublevels = [
                    'general',
                    'artist',
                    'copyright',
                    'character',
                    'species'
                ]

                for sublevel in sublevels:
                    temp_tags.extend(post['tags'][sublevel])

                post['tags'] = ' '.join(temp_tags)
                post['score'] = post['score']['total']

            if COUNT <= max_pages * POST_AMOUNT:
                max_pages = COUNT // POST_AMOUNT + 1

                # If max_pages is bigger than available pages,
                # loop the function with updated max_pages
                # based on the value of COUNT

                while loop_msg:
                    print(f" Processing {COUNT} results.")
                    loop_msg = False
                    continue

            else:
                print(
                    f" Processing "
                    f"{max_pages * POST_AMOUNT} "
                    f"out of {COUNT} results."
                )

                break

        return {'post': data}

    def get_post(self, add_tags, max_pages=10, id=''):
        return self.get_data(
            add_tags,
            max_pages,
            "&id=" + id
        )


def generate_chaos(pos_tags, neg_tags, chaos_amount):
    """Generates chaos in the prompt by adding random tags from the prompt to the positive and negative prompts"""

    chaos_list = [tag for tag in pos_tags.split(',') + neg_tags.split(',') if tag.strip() != '']
    chaos_list = list(set(chaos_list))
    random.shuffle(chaos_list)

    len_list = round(len(chaos_list) * chaos_amount)

    pos_list = chaos_list[len_list:]
    neg_list = chaos_list[:len_list]

    random.shuffle(neg_list)

    pos_prompt = ', '.join(pos_list)
    neg_prompt = ', '.join(neg_list)

    return pos_prompt, neg_prompt


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
def modify_prompt(prompt, tagged_prompt, type_deepbooru):
    """
    Modifies the prompt based on the type_deepbooru selected

    Args:
        prompt (str): the prompt
        tagged_prompt (str): the prompt tagged by deepbooru
        type_deepbooru (str): the type of modification

    Returns:
        str: the modified prompt
    """

    if type_deepbooru == 'Add Before':
        return tagged_prompt + ',' + prompt

    elif type_deepbooru == 'Add After':
        return prompt + ',' + tagged_prompt

    elif type_deepbooru == 'Replace':
        return tagged_prompt

    return prompt


def remove_repeated_tags(prompt):
    """Removes repeated tags while keeping order"""

    prompt = prompt.split(',')

    new_prompt = []

    for tag in prompt:
        tag = tag.strip()
        if tag and tag not in new_prompt:
            new_prompt.append(tag)

    return ', '.join(new_prompt)


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

    previous_loras = ''
    last_img = []
    real_steps = 0
    version = "1.2"
    original_prompt = ''

    def get_files(self, path):
        files = []

        for file in os.listdir(path):
            if file.endswith('.txt'):
                files.append(file)

        return files

    def hide_object(self, obj, booru):
        print(f'hide_object: {obj}, {booru.value}')

        if (
            booru.value == 'konachan'
            or booru.value == 'yande.re'
        ):
            obj.interactive = False

        else:
            obj.interactive = True

    def title(self):
        return "Ranbooru"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def show_gelbooru_api_fields(self, booru):
        return gr.update(
            visible=(booru == 'gelbooru')
        )

    def load_gelbooru_credentials(self, booru):
        """Load saved credentials for Gelbooru"""

        if booru != 'gelbooru':
            return (
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=""),
                gr.update(visible=False)
            )

        credentials = (
            credentials_manager.get_booru_credentials(
                'gelbooru'
            )
        )

        api_key = credentials.get('api_key', '')
        user_id = credentials.get('user_id', '')

        has_saved_creds = (
            api_key.strip() != ''
            and user_id.strip() != ''
        )

        if has_saved_creds:

            status_text = (
                f"✓ Credentials loaded from: "
                f"{credentials_manager.credentials_file}"
            )

            return (
                gr.update(
                    visible=False,
                    value=api_key
                ),

                gr.update(
                    visible=False,
                    value=user_id
                ),

                gr.update(
                    visible=True,
                    value=status_text
                ),

                gr.update(visible=True)
            )

        return (
            gr.update(
                visible=True,
                value=""
            ),

            gr.update(
                visible=True,
                value=""
            ),

            gr.update(
                visible=False,
                value=""
            ),

            gr.update(visible=False)
        )

    def clear_gelbooru_credentials(self):
        """Clear saved Gelbooru credentials"""

        credentials_manager.clear_booru_credentials(
            'gelbooru'
        )

        return (
            gr.update(
                visible=True,
                value=""
            ),  # api_key field

            gr.update(
                visible=True,
                value=""
            ),  # user_id field

            gr.update(
                visible=False,
                value=""
            ),  # credentials_status

            gr.update(visible=False)
        )

    def save_gelbooru_credentials(
        self,
        booru,
        api_key,
        user_id,
        save_credentials
    ):
        """Save Gelbooru credentials if checkbox is checked"""

        if (
            booru == 'gelbooru'
            and save_credentials
            and api_key.strip()
            and user_id.strip()
        ):

            credentials_manager.save_booru_credentials(
                'gelbooru',
                api_key.strip(),
                user_id.strip()
            )

            return "✓ Credentials saved"

        elif (
            booru == 'gelbooru'
            and not save_credentials
        ):
            return "Credentials will not be saved"

        return ""

    def refresh_ser(self):
        return gr.update(
            choices=self.get_files(user_search_dir)
        )

    def refresh_rem(self):
        return gr.update(
            choices=self.get_files(user_remove_dir)
        )

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

        # Determine initial Gelbooru credential visibility
        has_saved = credentials_manager.has_credentials(
            'gelbooru'
        )

        saved_creds = (
            credentials_manager.get_booru_credentials(
                'gelbooru'
            )
            if has_saved else {}
        )

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

            booru = gr.Dropdown(
                [
                    "safebooru",
                    "rule34",
                    "danbooru",
                    "gelbooru",
                    "konachan",
                    'yande.re',
                    'aibooru',
                    'xbooru',
                    'e621'
                ],

                label="Booru",
                value=selected_booru
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

            # API credentials for Gelbooru
            with gr.Group(visible=(selected_booru == 'gelbooru')) as gelbooru_credentials_group:

                gr.Markdown(
                    "### Gelbooru API Credentials"
                )

                api_key = gr.Textbox(
                    lines=1,
                    label="API Key",
                    placeholder=(
                        "Enter your Gelbooru API key"
                    ),
                    type="password",
                    visible=initial_api_key_visible,
                    value=saved_creds.get(
                        'api_key',
                        ''
                    )
                )

                user_id = gr.Textbox(
                    lines=1,
                    label="User ID",
                    placeholder=(
                        "Enter your Gelbooru user ID"
                    ),
                    visible=initial_user_id_visible,
                    value=saved_creds.get(
                        'user_id',
                        ''
                    )
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

            # show/hide gelbooru credentials group

            booru.input(
                self.show_gelbooru_api_fields,
                inputs=[booru],
                outputs=[gelbooru_credentials_group]
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

                    use_deepbooru = gr.Checkbox(
                        label="Use Deepbooru",
                        value=saved_value('use_deepbooru', False)
                    )

                    type_deepbooru = gr.Radio(
                        [
                            "Add Before",
                            "Add After",
                            "Replace"
                        ],

                        label="Deepbooru Tags Position",
                        value=saved_choice('type_deepbooru', 'Add Before', ['Add Before', 'Add After', 'Replace'])
                    )

            with gr.Group():

                with gr.Accordion("File", open=False):

                    use_search_txt = gr.Checkbox(
                        label="Use tags_search.txt",
                        value=saved_value('use_search_txt', False)
                    )

                    choose_search_txt = gr.Dropdown(
                        self.get_files(user_search_dir),
                        label="Choose tags_search.txt",
                        value=saved_choice('choose_search_txt', "", self.get_files(user_search_dir))
                    )

                    search_refresh_btn = gr.Button(
                        "Refresh"
                    )

                    use_remove_txt = gr.Checkbox(
                        label="Use tags_remove.txt",
                        value=saved_value('use_remove_txt', False)
                    )

                    choose_remove_txt = gr.Dropdown(
                        self.get_files(user_remove_dir),
                        label="Choose tags_remove.txt",
                        value=saved_choice('choose_remove_txt', "", self.get_files(user_remove_dir))
                    )

                    remove_refresh_btn = gr.Button(
                        "Refresh"
                    )

            with gr.Group():

                with gr.Accordion("Extra", open=False):

                    with gr.Box():

                        mix_prompt = gr.Checkbox(
                            label="Mix prompts",
                            value=saved_value('mix_prompt', False)
                        )

                        mix_amount = gr.Slider(
                            value=saved_slider('mix_amount', 2, 2, 10),
                            label="Mix amount",
                            minimum=2,
                            maximum=10,
                            step=1
                        )

                    with gr.Box():

                        chaos_mode = gr.Radio(
                            [
                                "None",
                                "Chaos",
                                "Less Chaos"
                            ],

                            label="Chaos Mode",
                            value=saved_choice('chaos_mode', 'None', ['None', 'Chaos', 'Less Chaos'])
                        )

                        chaos_amount = gr.Slider(
                            value=saved_slider('chaos_amount', 0.5, 0.1, 1),
                            label="Chaos Amount %",
                            minimum=0.1,
                            maximum=1,
                            step=0.05
                        )

                    with gr.Box():

                        negative_mode = gr.Radio(
                            [
                                "None",
                                "Negative"
                            ],

                            label="Negative Mode",
                            value=saved_choice('negative_mode', 'None', ['None', 'Negative'])
                        )

                    use_same_seed = gr.Checkbox(
                        label="Use same seed for all pictures",
                        value=saved_value('use_same_seed', False)
                    )

                    with gr.Box():

                        use_cache = gr.Checkbox(
                            label="Use cache",
                            value=saved_value('use_cache', True)
                        )

            with InputAccordion(
                saved_value('lora_enabled', False),
                label="LoRAnado",
                elem_id=self.elem_id("lo_enable")
            ) as lora_enabled:

                with gr.Box():

                    lora_lock_prev = gr.Checkbox(
                        label="Lock previous LoRAs",
                        value=saved_value('lora_lock_prev', False)
                    )

                    lora_folder = gr.Textbox(
                        value=saved_value('lora_folder', ""),
                        lines=1,
                        label="LoRAs Subfolder"
                    )

                    lora_amount = gr.Slider(
                        value=saved_slider('lora_amount', 1, 1, 10),
                        label="LoRAs Amount",
                        minimum=1,
                        maximum=10,
                        step=1
                    )

                with gr.Box():

                    lora_min = gr.Slider(
                        value=saved_slider('lora_min', -1.0, -1, 1),
                        label="Min LoRAs Weight",
                        minimum=-1.0,
                        maximum=1,
                        step=0.1
                    )

                    lora_max = gr.Slider(
                        value=saved_slider('lora_max', 1.0, -1, 1),
                        label="Max LoRAs Weight",
                        minimum=-1.0,
                        maximum=1.0,
                        step=0.1
                    )

                    lora_custom_weights = gr.Textbox(
                        value=saved_value('lora_custom_weights', ""),
                        lines=1,
                        label="LoRAs Custom Weights"
                    )

            search_refresh_btn.click(
                fn=self.refresh_ser,
                inputs=[],
                outputs=[choose_search_txt]
            )

            remove_refresh_btn.click(
                fn=self.refresh_rem,
                inputs=[],
                outputs=[choose_remove_txt]
            )

            # Event handler for loading credentials
            # when booru changes
            booru.input(
                fn=self.load_gelbooru_credentials,
                inputs=[booru],

                outputs=[
                    api_key,
                    user_id,
                    credentials_status,
                    clear_credentials_btn
                ]
            )

            # Event handler for clearing credentials
            clear_credentials_btn.click(
                fn=self.clear_gelbooru_credentials,
                inputs=[],

                outputs=[
                    api_key,
                    user_id,
                    credentials_status,
                    clear_credentials_btn
                ]
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
            mix_prompt,
            mix_amount,
            chaos_mode,
            negative_mode,
            chaos_amount,
            limit_tags,
            max_tags,
            sorting_order,
            mature_rating,
            lora_folder,
            lora_amount,
            lora_min,
            lora_max,
            lora_enabled,
            lora_custom_weights,
            lora_lock_prev,
            use_ip,
            use_search_txt,
            use_remove_txt,
            choose_search_txt,
            choose_remove_txt,
            search_refresh_btn,
            remove_refresh_btn,
            crop_center,
            use_deepbooru,
            type_deepbooru,
            use_same_seed,
            use_cache,
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
        load_outputs.extend([
            gelbooru_credentials_group,
            enabled.accordion,
            lora_enabled.accordion,
        ])

        def restore_browser_page():
            state = load_ranbooru_settings(is_img2img)
            if not state:
                return [gr.update() for _ in load_outputs]

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
                    'mature_rating', 'fringe_benefits',
                    'choose_search_txt', 'choose_remove_txt'
                ):
                    results.append(gr.update())
                    continue

                value = state.get(name, getattr(widget, 'value', None))

                if name == 'mature_rating':
                    value = value if value in rating_choices else 'All'
                    results.append(gr.update(choices=rating_choices, value=value))
                    continue

                if name in ('choose_search_txt', 'choose_remove_txt'):
                    path = user_search_dir if name == 'choose_search_txt' else user_remove_dir
                    choices = self.get_files(path)
                    results.append(gr.update(
                        choices=choices,
                        value=value if value in choices else None,
                    ))
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
            loranado_on = state.get('lora_enabled', lora_enabled.value)
            if type(loranado_on) is not bool:
                loranado_on = bool(lora_enabled.value)
            results.extend([
                gr.update(visible=(current_booru == 'gelbooru')),
                gr.update(open=enabled_on),
                gr.update(open=loranado_on),
            ])
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

    def loranado(
        self,
        lora_enabled,
        lora_folder,
        lora_amount,
        lora_min,
        lora_max,
        lora_custom_weights,
        p,
        lora_lock_prev
    ):

        lora_prompt = ''

        if lora_enabled:

            if lora_lock_prev:
                lora_prompt = self.previous_loras

            else:
                loras = []

                loras = os.listdir(f'{lora_folder}')

                # get only .safetensors files
                loras = [
                    lora.replace('.safetensors', '')
                    for lora in loras
                    if lora.endswith('.safetensors')
                ]

                for l in range(0, lora_amount):

                    lora_weight = 0

                    if lora_custom_weights != '':
                        lora_weight = float(
                            lora_custom_weights.split(',')[l]
                        )

                    while lora_weight == 0:
                        lora_weight = round(
                            random.uniform(
                                lora_min,
                                lora_max
                            ),
                            1
                        )

                    lora_prompt += (
                        f'<lora:{random.choice(loras)}'
                        f':{lora_weight}>'
                    )

                self.previous_loras = lora_prompt

            if lora_prompt:

                if isinstance(p.prompt, list):

                    for num, pr in enumerate(p.prompt):
                        p.prompt[num] = (
                            f'{lora_prompt} {pr}'
                        )

                else:
                    p.prompt = (
                        f'{lora_prompt} {p.prompt}'
                    )

        return p
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
        mix_prompt,
        mix_amount,
        chaos_mode,
        negative_mode,
        chaos_amount,
        limit_tags,
        max_tags,
        sorting_order,
        mature_rating,
        lora_folder,
        lora_amount,
        lora_min,
        lora_max,
        lora_enabled,
        lora_custom_weights,
        lora_lock_prev,
        use_ip,
        use_search_txt,
        use_remove_txt,
        choose_search_txt,
        choose_remove_txt,
        search_refresh_btn,
        remove_refresh_btn,
        crop_center,
        use_deepbooru,
        type_deepbooru,
        use_same_seed,
        use_cache,
        api_key,
        user_id,
        save_credentials,
        credentials_status,
        clear_credentials_btn
    ):

        # Snapshot the actual submitted values, not partially edited browser state.
        # Saving before API queries also preserves inputs when the query fails.
        if enabled or lora_enabled:
            save_ranbooru_settings(
                getattr(self, '_ranbooru_is_img2img', isinstance(p, StableDiffusionProcessingImg2Img)),
                locals()
            )

        # Manage Cache
        if (
            use_cache
            and not requests_cache.patcher.is_installed()
        ):
            requests_cache.install_cache(
                'ranbooru_cache',
                backend='sqlite',
                expire_after=3600
            )

        elif (
            not use_cache
            and requests_cache.patcher.is_installed()
        ):
            requests_cache.uninstall_cache()

        if enabled:

            # Handle Gelbooru credentials
            gelbooru_api_key = None
            gelbooru_user_id = None

            if booru == 'gelbooru':

                # Use provided credentials
                # or load from saved credentials
                if (
                    api_key.strip()
                    and user_id.strip()
                ):

                    gelbooru_api_key = api_key.strip()
                    gelbooru_user_id = user_id.strip()

                    # Save credentials if checkbox is checked
                    if save_credentials:
                        credentials_manager.save_booru_credentials(
                            'gelbooru',
                            gelbooru_api_key,
                            gelbooru_user_id
                        )

                else:
                    # Try to load saved credentials
                    saved_credentials = (
                        credentials_manager
                        .get_booru_credentials('gelbooru')
                    )

                    gelbooru_api_key = (
                        saved_credentials.get(
                            'api_key',
                            ''
                        )
                    )

                    gelbooru_user_id = (
                        saved_credentials.get(
                            'user_id',
                            ''
                        )
                    )

            # Initialize APIs
            booru_apis = {
                'gelbooru': Gelbooru(
                    fringe_benefits,
                    gelbooru_api_key,
                    gelbooru_user_id
                ),

                'rule34': Rule34(
                    username=(
                        credentials_manager
                        .get_booru_credentials('rule34')
                        .get('user_id', '')
                    ),

                    api_key=(
                        credentials_manager
                        .get_booru_credentials('rule34')
                        .get('api_key', '')
                    )
                ),

                'safebooru': Safebooru(),
                'danbooru': Danbooru(),
                'konachan': Konachan(),
                'yande.re': Yandere(),
                'aibooru': AIBooru(),
                'xbooru': XBooru(),
                'e621': e621(),
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

            if ',' in remove_tags:
                bad_tags.extend(
                    remove_tags.split(',')
                )

            else:
                bad_tags.append(remove_tags)

            if use_remove_txt:
                bad_tags.extend(
                    open(
                        os.path.join(
                            user_remove_dir,
                            choose_remove_txt
                        ),
                        'r'
                    ).read().split(',')
                )

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

            if use_search_txt:

                search_tags = open(
                    os.path.join(
                        user_search_dir,
                        choose_search_txt
                    ),
                    'r'
                ).read()

                search_tags_r = search_tags.replace(
                    " ",
                    ""
                )

                split_tags = (
                    search_tags_r.splitlines()
                )

                filtered_tags = [
                    line
                    for line in split_tags
                    if line.strip()
                ]

                rand_selected = random.randint(
                    0,
                    len(filtered_tags) - 1
                )

                selected_tags = (
                    filtered_tags[rand_selected]
                )

                tags = (
                    f'{tags},{selected_tags}'
                    if tags
                    else selected_tags
                )

            add_tags = '&tags=-animated'

            if tags:
                add_tags += (
                    f'+{tags.replace(",", "+")}'
                )

            if mature_rating != 'All':
                add_tags += (
                    f'+rating:'
                    f'{RATINGS[booru][mature_rating]}'
                )

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

            if not clean_posts:
                raise ValueError(
                    '[Ranbooru] No posts were returned by the booru API for '
                    'these tags and rating. Check the tags, rating, post ID, '
                    'or API credentials.'
                )

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

                    if mix_prompt:

                        temp_tags = []
                        max_tags = 0

                        for _ in range(0, mix_amount):

                            if not post_id:
                                random_mix_number = 0

                                if not post_id:
                                    random_mix_number = (
                                        self.random_number(
                                            sorting_order,
                                            1,
                                            len(data['post'])
                                        )[0]
                                    )

                                temp_tags.extend(
                                    data['post'][
                                        random_mix_number
                                    ]['tags'].split(' ')
                                )

                                max_tags = max(
                                    max_tags,

                                    len(
                                        data['post'][
                                            random_mix_number
                                        ]['tags'].split(' ')
                                    )
                                )

                        # distinct temp_tags
                        temp_tags = list(set(temp_tags))

                        # Copy to avoid changing the source post when mixing.
                        random_post = dict(data['post'][
                            random_number
                        ])

                        max_tags = min(
                            max(len(temp_tags), 20),
                            max_tags
                        )

                        random_post['tags'] = ' '.join(
                            random.sample(
                                temp_tags,
                                max_tags
                            )
                        )

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
            if use_img2img or use_deepbooru:

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

            new_prompts = []

            # Cleaning Tags
            for prompt in prompts:

                prompt_tags = [
                    tag
                    for tag in html.unescape(
                        prompt
                    ).split(',')
                    if tag.strip() not in bad_tags
                ]

                for bad_tag in bad_tags:

                    if '*' in bad_tag:

                        prompt_tags = [
                            tag
                            for tag in prompt_tags
                            if bad_tag.replace('*', '')
                            not in tag
                        ]

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

                if chaos_mode in [
                    'Chaos',
                    'Less Chaos'
                ]:

                    negative_prompt = ''

                    if chaos_mode == 'Less Chaos':
                        negative_prompt = (
                            p.negative_prompt
                        )

                    p.prompt, negative_prompt = (
                        generate_chaos(
                            p.prompt,
                            negative_prompt,
                            chaos_amount
                        )
                    )

                    p.negative_prompt = (
                        f"{p.negative_prompt},"
                        f"{negative_prompt}"
                        if p.negative_prompt
                        else negative_prompt
                    )

            else:

                print('Processing Multiple Prompts')

                negative_prompts = []
                new_prompts = []

                if chaos_mode == 'Chaos':

                    for prompt in prompts:

                        tmp_prompt, negative_prompt = (
                            generate_chaos(
                                prompt,
                                p.negative_prompt,
                                chaos_amount
                            )
                        )

                        new_prompts.append(tmp_prompt)
                        negative_prompts.append(
                            negative_prompt
                        )

                    prompts = new_prompts
                    p.negative_prompt = negative_prompts

                elif chaos_mode == 'Less Chaos':

                    for prompt in prompts:

                        tmp_prompt, negative_prompt = (
                            generate_chaos(
                                prompt,
                                '',
                                chaos_amount
                            )
                        )

                        new_prompts.append(tmp_prompt)

                        negative_prompts.append(
                            negative_prompt
                        )

                    prompts = new_prompts

                    p.negative_prompt = [
                        p.negative_prompt
                        + ','
                        + negative_prompt

                        for negative_prompt
                        in negative_prompts
                    ]

                else:

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
            if negative_mode == 'Negative':

                # remove tags from p.prompt
                # using tags from the original prompt
                orig_list = self.original_prompt.split(',')

                if isinstance(p.prompt, list):

                    new_positive_prompts = []
                    new_negative_prompts = []

                    for pr, npp in zip(
                        p.prompt,
                        p.negative_prompt
                    ):

                        clean_prompt = pr.split(',')

                        clean_prompt = [
                            tag
                            for tag in clean_prompt
                            if tag not in orig_list
                        ]

                        clean_prompt = ','.join(
                            clean_prompt
                        )

                        new_positive_prompts.append(
                            self.original_prompt
                        )

                        new_negative_prompts.append(
                            f'{npp},{clean_prompt}'
                        )

                    p.prompt = new_positive_prompts
                    p.negative_prompt = (
                        new_negative_prompts
                    )

                else:

                    clean_prompt = p.prompt.split(',')

                    clean_prompt = [
                        tag
                        for tag in clean_prompt
                        if tag not in orig_list
                    ]

                    clean_prompt = ', '.join(tag.strip() for tag in clean_prompt if tag.strip())

                    p.negative_prompt = (
                        f'{p.negative_prompt},'
                        f'{clean_prompt}'
                    )

                    p.prompt = self.original_prompt

            if (
                negative_mode == 'Negative'
                or chaos_mode in [
                    'Chaos',
                    'Less Chaos'
                ]
            ):

                # NEGATIVE PROMPT FIX
                neg_prompt_tokens = []

                for pr in p.negative_prompt:

                    neg_prompt_tokens.append(
                        model_hijack
                        .get_prompt_lengths(pr)[1]
                    )

                if len(set(neg_prompt_tokens)) != 1:

                    print('Padding negative prompts')

                    max_tokens = max(
                        neg_prompt_tokens
                    )

                    for num, neg in enumerate(
                        neg_prompt_tokens
                    ):

                        while neg < max_tokens:

                            p.negative_prompt[num] += (
                                random.choice(
                                    p.negative_prompt[
                                        num
                                    ].split(',')
                                )
                            )

                            # p.negative_prompt[num] += '_'

                            neg = (
                                model_hijack
                                .get_prompt_lengths(
                                    p.negative_prompt[
                                        num
                                    ]
                                )[1]
                            )

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

            if use_same_seed:

                p.seed = (
                    random.randint(
                        0,
                        2 ** 32 - 1
                    )
                    if p.seed == -1
                    else p.seed
                )

                p.seed = [p.seed] * p.batch_size

            # LORANADO
            p = self.loranado(
                lora_enabled,
                lora_folder,
                lora_amount,
                lora_min,
                lora_max,
                lora_custom_weights,
                p,
                lora_lock_prev
            )

            if (
                use_deepbooru
                and not use_img2img
            ):

                self.last_img = last_img

                tagged_prompts = self.use_autotagger(
                    'deepbooru'
                )

                if isinstance(p.prompt, list):

                    p.prompt = [
                        modify_prompt(
                            pr,
                            tagged_prompts[num],
                            type_deepbooru
                        )

                        for num, pr
                        in enumerate(p.prompt)
                    ]

                    p.prompt = [
                        remove_repeated_tags(pr)
                        for pr in p.prompt
                    ]

                else:

                    p.prompt = modify_prompt(
                        p.prompt,
                        tagged_prompts,
                        type_deepbooru
                    )

                    p.prompt = remove_repeated_tags(
                        p.prompt
                    )

            if use_img2img:

                if not use_ip:

                    self.real_steps = p.steps
                    p.steps = 1

                self.last_img = last_img

                if use_ip:

                    controlNetModule = (
                        importlib.import_module(
                            'extensions.'
                            'sd-webui-controlnet.'
                            'scripts.external_code',
                            'external_code'
                        )
                    )

                    controlNetList = (
                        controlNetModule
                        .get_all_units_in_processing(p)
                    )

                    copied_network = (
                        controlNetList[0]
                        .__dict__
                        .copy()
                    )

                    copied_network['enabled'] = True

                    copied_network['weight'] = (
                        denoising
                    )

                    array_img = np.array(last_img[0])

                    copied_network['image']['image'] = (
                        array_img
                    )

                    copied_networks = (
                        [copied_network]
                        + controlNetList[1:]
                    )

                    controlNetModule.update_cn_script_in_processing(
                        p,
                        copied_networks
                    )

            elif lora_enabled:

                p = self.loranado(
                    lora_enabled,
                    lora_folder,
                    lora_amount,
                    lora_min,
                    lora_max,
                    lora_custom_weights,
                    p,
                    lora_lock_prev
                )

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
        mix_prompt,
        mix_amount,
        chaos_mode,
        negative_mode,
        chaos_amount,
        limit_tags,
        max_tags,
        sorting_order,
        mature_rating,
        lora_folder,
        lora_amount,
        lora_min,
        lora_max,
        lora_enabled,
        lora_custom_weights,
        lora_lock_prev,
        use_ip,
        use_search_txt,
        use_remove_txt,
        choose_search_txt,
        choose_remove_txt,
        search_refresh_btn,
        remove_refresh_btn,
        crop_center,
        use_deepbooru,
        type_deepbooru,
        use_same_seed,
        use_cache,
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

            final_prompts = p.prompt

            if use_deepbooru:

                tagged_prompts = self.use_autotagger(
                    'deepbooru'
                )

                if isinstance(p.prompt, list):

                    final_prompts = [
                        modify_prompt(
                            pr,
                            tagged_prompts[num],
                            type_deepbooru
                        )

                        for num, pr
                        in enumerate(p.prompt)
                    ]

                    final_prompts = [
                        remove_repeated_tags(pr)
                        for pr in final_prompts
                    ]

                else:

                    final_prompts = modify_prompt(
                        p.prompt,
                        tagged_prompts,
                        type_deepbooru
                    )

                    final_prompts = remove_repeated_tags(
                        final_prompts
                    )

            p = StableDiffusionProcessingImg2Img(
                sd_model=shared.sd_model,

                outpath_samples=(
                    shared.opts.outdir_samples
                    or shared.opts.outdir_img2img_samples
                ),

                outpath_grids=(
                    shared.opts.outdir_grids
                    or shared.opts.outdir_img2img_grids
                ),

                prompt=final_prompts,

                negative_prompt=p.negative_prompt,

                seed=p.seed,

                sampler_name=p.sampler_name,

                scheduler=p.scheduler,

                batch_size=p.batch_size,

                n_iter=p.n_iter,

                steps=self.real_steps,

                cfg_scale=p.cfg_scale,

                width=width,

                height=height,

                init_images=self.last_img,

                denoising_strength=denoising,
            )

            proc = process_images(p)

            processed.images = proc.images
            processed.infotexts = proc.infotexts

            if use_last_img:

                processed.images.append(
                    self.last_img[0]
                )

            else:

                for num, img in enumerate(
                    self.last_img
                ):

                    processed.images.append(img)

                    processed.infotexts.append(
                        proc.infotexts[num + 1]
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

    def use_autotagger(
        self,
        model
    ):
        """
        Use the autotagger to tag the images

        Args:
            model (str):
                the model to use.
                Right now only
                'deepbooru'
                is supported

        Returns:
            list:
                the tagged prompts
        """

        if model == 'deepbooru':

            if isinstance(
                self.original_prompt,
                str
            ):

                orig_prompt = [
                    self.original_prompt
                ]

            else:
                orig_prompt = self.original_prompt

            deepbooru.model.start()

            final_prompts = []

            for img, prompt in zip(
                self.last_img,
                orig_prompt
            ):

                final_prompts.append(
                    prompt
                    + ','
                    + deepbooru.model.tag_multi(img)
                )

            deepbooru.model.stop()

            return final_prompts
