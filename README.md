# Ranbooru
![Alt text](pics/ranbooru.png)
Ranbooru is an extension for the [automatic111 Stable Diffusion UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). The purpose of this extension is to add a panel that gets a random set of tags from boorus pictures. This is mostly being used to help me test my checkpoints on a large variety of tags.
![Alt text](pics/image.png)

## Installation
This fork targets [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo).

1. Open the **Extensions** tab, then **Install from URL**.
2. Paste `https://github.com/Dusky-dev/sd-webui-ranbooru-forge-neo` and click **Install**.
3. Restart the UI. You should find a new panel called "Ranbooru" in the txt2img and img2img tabs.

You can also `git clone` the repository into the `extensions` folder. Install the whole repository, not just the script: the extension also needs the files next to it (`pose_tag_groups.json`, `pose_name_tags.json`, `javascript/`).

## Features
### Ranbooru
This is the main part of the extension. It gets a random set of tags from boorus pictures.  
Here's an explanation of all the parameters:
- **Booru**: The booru to get the tags from. Right now Gelbooru, Rule34, Safebooru, Konachan, AIBooru, Danbooru, XBooru and e621 are implemented (yande.re was removed in November 2026: its tag searches returned no posts). You can easily add more creating a class for the booru and adding it to the booru list in the script.

### ⚠️ API credentials (Gelbooru, Rule34, e621)
Gelbooru (since June 2025) and Rule34 require an API key and user ID; e621 works anonymously but accepts an optional login. When you select one of these boorus, the Ranbooru panel shows fields for that site's credentials. You can optionally save them to disk (`user/credentials/credentials.json`, never uploaded by Git). Once saved, the fields are hidden and a message shows the path to the credentials file; **Clear saved credentials** removes them for the selected booru only. Each site's credentials are only ever sent to that site.

- **Gelbooru:** API key and user ID from your Gelbooru account options.
- **Rule34:** API key and user ID from **My Account > Options** on rule34.xxx (see [api.rule34.xxx](https://api.rule34.xxx/)). Without them, Rule34 returns no posts and Ranbooru reports that a login is needed.
- **e621 (optional):** your username and an API key from **Account > Manage API Access**. They are sent as an HTTP login, not in the URL. About 8% of e621 posts have no image link for anonymous users (these are skipped); logging in may make some of them available.

- **Max Pages**: Random posts are drawn from the **whole search**: Ranbooru asks the booru to order the search randomly (`sort:random` on Gelbooru, Safebooru, Rule34 and XBooru; `order:random` on Konachan, AIBooru and e621; `random:100` on Danbooru) and takes posts from that one reply, so Max Pages doesn't apply. Max Pages is used when that isn't possible: when your tags set their own order (`sort:`, `order:` or `random:`), when the booru refuses or returns nothing, and on Danbooru when your search already uses both of its two tag slots (`rating:` doesn't count). Then Ranbooru picks a random page among the pages that exist, within each site's own page limit (Gelbooru: 10 pages, as it only pages about 1,000 posts deep).
- **Post ID**: Here you can specify the ID of the post to get the tags from. If you leave it blank, the extension will get a random post (or more than one). Supported on every booru.
- **Tags to Search (Pre)**: This add the tags you define (this should be separated by commas e.g: 1girl,solo,short_hair) to the search query. This is useful if you want to get tags from a specific category, like "1girl" or "solo".
- **Tags to Remove (Post)**: This remove the tags you define (this should be separated by commas e.g: 1girl,solo,short_hair) from the result query. This is useful if you want to remove tags that are too generic, like "1girl" or "solo". Spaces around entries, letter case, and spaces vs. underscores don't matter (`solo, short hair` works). You can also use * with any tag to remove every tags which contains the related word. e.g: *hair will remove every tag that contains the word "hair".
- **Mature Rating**: This sets the mature rating of the booru. This is useful if you want to get only SFW or NSFW tags. It only works on supported boorus (right now it has been tested only on Gelbooru).
- **Remove Bad Tags**: This remove tags that you usually don't need (watermarks,text,censor)
- **Shuffle Tags**: This shuffle the tags before adding them to the text.
- **Only poses**: Removes appearance tags (hair, eyes, skin, body traits) plus character, copyright and artist names, keeping pose, expression and composition tags. **Strip clothing & accessories** also removes outfits and accessories. See [usage.md](usage.md#only-poses) for what each booru supports.
- **Skip tags already in prompt** (off by default): Leaves out fetched tags you already typed. Weights and attention brackets are ignored, so `(blue hair:1.2)` in your prompt skips a fetched `blue_hair`; LoRA tags are ignored.
- **Convert** "\_" to Spaces": This convert \_ to spaces in the tags.
- **Use the same prompt for all images**: This use the same prompt for all the generated images in the same batch. If not selected, each image will have a different prompt.
- **Fringe Benefits**: This option is available only for Gelbooru and enables all the hidden content available on the website. This is equivalent to enabling the "Display all site content" flag in the Gelbooru website settings.
- **Limit Tags**: This limits the number of tags to use in percentage of the original tags. For example, if you set it to 50, it will use only half of the tags.
- **Max Tags**: This limits the number of tags to use.
- **Change Background**: This tries to change the background of the parsed tags by adding or removing specific tags
- **Change Color**: This tries to change the color of the parsed tags by adding or removing specific tags
- **Sorting Order**: Makes high- or low-score posts more likely among the up to 100 posts Ranbooru fetched (a random draw from the search, see Max Pages). It is applied after the posts are fetched; to search by score on the booru itself, add the booru's own order to your tags, such as `sort:score` (Gelbooru family) or `order:score` (others).
- **Use img2img**: This uses not only the tags from the random image, but also the original picture to generate the final result. Forge still runs its txt2img job first, so Ranbooru makes that a cheap throwaway pass (1 step, 256 × 256, no Hires fix, not saved), then runs one real img2img pass per batch on that batch's booru image(s) with your size, steps, prompt, sampler, scheduler, CFG, seed, styles and setting overrides (such as clip skip). Other extensions (ADetailer, hires fix) do not run on the img2img passes.
  - Each image is downloaded at the smallest size the booru offers that still covers the output size (often a few hundred KB instead of several MB); otherwise the original is used.
  - If an image can't be downloaded, Ranbooru uses another post from the same search, so every image keeps its own post's tags. Posts without an image link are skipped.
- **Send to ControlNet** (with Use img2img): Sends the first booru image to **ControlNet Unit 1** of Forge Neo's built-in ControlNet instead of running img2img. Choose the model and preprocessor in Unit 1; Ranbooru enables the unit and supplies the image, and the Denoising slider sets the unit's weight, so you shouldn't need a dummy image.
- **Denoising Strength**: This is the strength of the denoising filter. The higher the value, the more the picture will change from the original.
- **Use last image as img2img** This uses the same picture for all the img2img generations in the same batch.
- **Crop Center**: This crops the parsed images to the center before processing. This is useful if you want to keep the width/height you set.
- **Use Cache**: Reuses booru API replies for an hour (`user/ranbooru_cache.sqlite`). This improves the speed of the search, but it can return the same results while the cache lasts. Only Ranbooru's own booru requests are cached: not image downloads or other extensions. Error and login-refusal replies are never cached, and API keys are not stored. Random draws (`sort:random` and the like) are never cached. Cached replies are shared between logged-in and anonymous e621 requests, so after logging in to e621 turn Use cache off for one generation. Older versions kept the cache in `ranbooru_cache.sqlite` in the WebUI folder; that file can be deleted.

## How to use
Check the usage.md file for a detailed explanation of how to use the extension.  
Also check my article on [CivitAI](https://civitai.com/articles/3357/ranbooru-the-comprehensive-guide).

## Changelog
### Forge Neo fork
- Removed LoRAnado, the File panel (tags_search.txt / tags_remove.txt), Deepbooru (deleted from Forge Neo in July 2025), and Mix prompts, Chaos mode, Negative mode and Use same seed. Old `user/search` and `user/remove` files are left on disk but no longer used.
- Send to ControlNet now targets Forge Neo's built-in ControlNet.
- Use img2img works with Batch count above 1 (one img2img pass per batch); Gelbooru uses `sort:random`; Use cache only caches Ranbooru's booru requests; AIBooru supports post IDs; new Skip tags already in prompt option.
- Use img2img downloads Gelbooru images again (Gelbooru's hotlink protection had served a web page in their place).
- Random posts come from the whole search on every booru (on Danbooru when a tag slot is free), not one random page. Use img2img: the throwaway pass is small and no longer saved or upscaled by Hires fix; smaller image copies are downloaded when big enough; a failed download is replaced by another post instead of shifting images onto other posts' prompts. Removed yande.re.

### 1.2
- Added the ability to use * in the remove tags to remove every tag that contains the word

## Known Issues
- "sd-dynamic-prompts" creates problems with the multiple prompts option. Disabling the extension is the only solution for now.
- Use img2img still makes Forge run a throwaway txt2img pass before the real img2img passes. It is now 1 step at 256 × 256 and not saved, but it shows briefly in the preview and other extensions (such as ADetailer) still run on it.

## Found an issue?  
If you found an issue with the extension, please report it in the issues section of this repository.  
Special thanks to [TheGameratorT](https://github.com/TheGameratorT), [SmashinFries](https://github.com/SmashinFries), and [w-e-w](https://github.com/w-e-w) for contributing.

## Check out my other scripts
- [Ranbooru for ComfyUI](https://github.com/Inzaniak/comfyui-ranbooru)
- [Workflow](https://github.com/Inzaniak/sd-webui-workflow)

---
## Made by Inzaniak
![Alt text](pics/logo.png) 


If you'd like to support my work feel free to check out my Patreon: https://www.patreon.com/Inzaniak

Also check my other links:
- **Personal Website**: https://inzaniak.github.io 
- **Deviant Art**: https://www.deviantart.com/inzaniak
- **CivitAI**: https://civitai.com/user/Inzaniak/models


## Forge Neo notes

### Persistent settings
When you click **Generate** with Ranbooru enabled, the extension saves its controls to `user/last_settings.json`. txt2img and img2img have separate saved values. The settings restore after a Forge Neo restart **and after a browser reload or new tab without restarting the server**. Credentials are not saved into this file; they stay under the opt-in credential saving above. See [PERSISTENCE.md](PERSISTENCE.md).

### Recently used boorus and progress
The booru dropdown lists the boorus you generated with most recently first (shared by both tabs in `user/booru_recent.json`); reload the page to see the new order. While searching, Ranbooru shows messages such as "Searching danbooru..." and "Processing 201 results" in Forge Neo's progress bar, next to the percentage and ETA.

### Back up `user/` before updating from old versions
Older versions accidentally tracked files under `user/`. This has been fixed: credentials, saved settings, recent-booru order, and custom tag files are now all local-only. **On the first pull that removes those tracked files, Git can delete your existing local versions.** Back up `user/` before pulling, then restore it afterward. See [USER_DATA.md](USER_DATA.md) for the safe migration commands and recovery advice.

### Booru errors
Ranbooru reports what went wrong instead of failing silently: refused logins (HTTP 401/403), rate limits (HTTP 429), web pages served instead of API data, and API error messages. Gelbooru requests are retried up to three times on rate limits, server errors and dropped connections. Credentials are never printed: for Gelbooru the console only says `API credentials: key and user ID supplied` or `MISSING/INCOMPLETE (search is anonymous)`. If a Gelbooru tag search fails while a post ID works, check the key and user ID: Gelbooru has required them for tag searches since June 2025.
