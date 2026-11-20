# Examples

## Basic Search
To do a basic search you'll need to set the following parameters:

### Required
- **Booru**: The booru you want to search on.
- **Tags to search (pre)**: The tags to search for on the booru. These should be the same used on a booru, but separated by a comma (e.g. `'1girl, solo'`).

### Optional
- **Max Pages**: This is the number of pages to search on the booru. Each page has 100 images, so if you set this to 5, it will search 500 images. This changes the variety of images you'll get.
- **Tags to remove (post)**: These are the tags to remove from the prompt after the search. (e.g. If you search for `'1girl, solo'` and remove `'solo'`, the final prompt won't have 'solo' in it.)
- **Use the same prompt for all images**: By default you'll get a different prompt for each image you create. If you want to use the same prompt for all images, set this to true.

## Post ID
You can search for a specific post ID by setting the following parameters:

### Required
- **Booru**: The booru you want to search on. Every booru except AIBooru supports this.
- **Post ID**: The ID of the post you want to search for. This is the number at the end of the URL of the post. (e.g. `https://danbooru.donmai.us/posts/123456` --> `123456`)

### Optional
- **Tags to remove (post)**: These are the tags to remove from the prompt after the search. (e.g. If you search for `'1girl, solo'` and remove `'solo'`, the final prompt won't have `'solo'` in it.)

## Img2Img
You can use the prompt plus the original image of the random booru pictures to create an img2img image. To do this, set the following parameters:

### Required
- **Booru**: The booru you want to search on.
- **Tags to search (pre)**: The tags to search for on the booru. These should be the same used on a booru, but separated by a comma (e.g. `'1girl, solo'`).
- **Use img2img**: Set this to true to use the img2img mode.

### Optional
- **Max Pages**: This is the number of pages to search on the booru. Each page has 100 images, so if you set this to 5, it will search 500 images. This changes the variety of images you'll get.
- **Tags to remove (post)**: These are the tags to remove from the prompt after the search. (e.g. If you search for `'1girl, solo'` and remove `'solo'`, the final prompt won't have 'solo' in it.)
- **Use the same prompt for all images**: By default you'll get a different prompt for each image you create. If you want to use the same prompt for all images, set this to true.
- **Denoising Strength**: This is the strength of the img2img denoiser. The higher the number, the more the picture will differ from the original. The default is 0.75.
- **Use last image as img2img**: By default you'll get a different image for each image you create. If you want to use the same image for all images, set this to true.

## Additional Modes
These can be used with the Basic Search and img2img examples.
### Mixing prompts
By enabling `Mix prompts` each image will be created using a mixture of N different booru posts where N is defined by the `Mix Amount` parameter.
### Chaos Mode
By enabling `Chaos Mode` each image will have an amount of tags moved to the negative prompt. The amount is defined by the `Chaos Amount` parameter, for example 0.5 means 50% of the tags will be moved to the negative prompt. Using the `Less Chaos` option won't move the tags you manually insert in the negative prompt.
### Negative Mode
By enabling `Negative Mode` each image will have all tags moved to the negative prompt.
### Only Poses
Enable `Only poses` to keep the composition of the random posts (pose, action, expression, camera angle, subject count like `1girl`) while removing the tags that describe *who* is in the picture:
- **Appearance:** hair colour, length and style (`long_hair`, `twintails`, `blunt_bangs`), eyes (`blue_eyes`, `slit_pupils`), skin, body type (`large_breasts`, `muscular`), animal ears, tails, horns, wings and halos, moles, scars and freckles.
- **Character, copyright and artist names:** removed using the booru's own tag categories on **Danbooru and AIBooru**. On **e621**, whose posts no longer include categories, the prompt's tags are looked up with e621's tag API (usually one request per generation, cached), and artist, copyright, character and species tags are removed. This lookup also runs without Only poses: like the old adapter, e621 meta, lore and invalid tags (`hi_res`, `2026`, `16:9`) are always left out of the prompt. If the lookup fails, generation continues without it. On **Gelbooru**, which uses the same tag names but doesn't report categories, a list of Danbooru's most-used names is used instead (`pose_name_tags.json`: 20,000 characters, 5,000 copyrights, ~9,700 artists). Measured on recent Danbooru posts, this list covers about 97% of copyright tags, 83% of character tags and only about 23% of artist tags, because there are so many artists. Expect most artist names to get through. Tags shaped like `name_(series)` are also removed when the series is a known Danbooru copyright or a common series abbreviation such as `kancolle` or `fate`. Costume qualifiers like `_(1st_costume)` or `_(swimsuit)` don't count. This rule catches about half of the characters the list misses, along with series-specific items such as `vision_(genshin_impact)`. Other boorus only catch `*_(artist)` and `*_(cosplay)` tags unless the experimental option below is on.
- **Experimental: strip Danbooru name list on every booru:** applies the same name list on every other booru (checked on Safebooru, Xbooru, Konachan and yande.re). A name that doesn't exist on that booru simply matches nothing. Very short artist names are left out of the list because they are often ordinary words elsewhere.

`Strip clothing & accessories` is a separate toggle (off by default). It works with or without `Only poses` and removes outfits, legwear, footwear, headwear, hair ornaments, jewelry, glasses and makeup (`white_shirt`, `pleated_skirt`, `school_uniform`, `hair_ribbon`).

Tags from Danbooru's posture, gestures, hands, sexual positions, face (expressions), image composition, focus and character count [tag groups](https://danbooru.donmai.us/wiki_pages/tag_groups) are never removed, so `closed_eyes`, `looking_back`, `hand_in_own_hair` and `skirt_lift` stay.

Both options also apply to tags added by Deepbooru (including the name list, since Deepbooru outputs Danbooru tag names). The lists use Danbooru's tag names, which Gelbooru, Safebooru and most other boorus share; e621 uses its own vocabulary, so e621 posts also get e621-specific lists built from e621's own [tag groups](https://e621.net/wiki_pages/tag_groups) (anatomy, weight, markings; pose, facial expressions and touching are never removed). These cover tufts, markings, pawpads, countershading, tail and ear shapes and body weight. Body-plan tags such as `anthro`, `feral` and `biped` are kept. With **Strip clothing & accessories**, e621 posts also lose e621's garment vocabulary (`topwear`, `bottomwear`, `legwear`, `black_bottomwear`, `denim_bottomwear`, piercings, uniforms, costumes), from its clothes, outfits and piercing tag groups. As on Danbooru, tags saying whether someone is dressed (`clothing`, `clothed`, `nude`, `topless`, `underwear_only`) and clothing actions or states (`shirt_lift`, `open_clothing`, `torn_topwear`, `wet_clothing`) are kept. Anything that gets through can be removed with **Tags to remove (post)**.

The lists live in `pose_tag_groups.json` and `pose_name_tags.json`. To refresh them from Danbooru, run `python tools/build_pose_tag_groups.py` (or add `groups` / `names` to rebuild one file).

### LoRAnado
By enabling `Use LoRAnado` you can add random LoRAs to the prompt. This can lead to interesting results. Once you have enabled the option you have to choose a folder from your LoRA path, this can be your main folder or any subfolder (right now only the LoRAs contained in the set folder will be used, not the ones in other subfolders). You now need to set the min and max weight values you want to generate and the amount of random LoRAs to use. You can also set custom weights for each LoRA by separating them with a comma (e.g. 0.2,0.3,0.5). If you want to use the same LoRAs of the previous generation, enable `Lock Previous LoRAs`.