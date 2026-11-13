#Bear Classifier

An image classifier that tells the difference between **grizzly bears**, **black bears**, and **teddy bears**, and recognizes when a photo **isn't a bear at all**.

**👉 [Try the live app](https://bearclassifier-torian.streamlit.app/)**

(It's hosted on a free tier, so if it's been idle, it may take about a minute to wake up.)

---

## Results

| Metric | Value |
|---|---|
| Validation accuracy | **93.7%** |
| Error rate | 6.3% |
| Model | ResNet-18 (fine-tuned) |
| Classes | grizzly, black, teddy, not_bear |

**Every bear in the validation set was classified correctly.** All of the model's mistakes were non-bear images predicted as bears (see [What I learned](#what-i-learned)).

---

## How I built it

1. **Collected data:** downloaded images of each bear type using DuckDuckGo image search, plus a varied set of non-bear images (animals, people, objects, scenes, and cartoons).
2. **Cleaned the data:** removed images that failed to download or open.
3. **Augmented the data:** used random crops, flips, rotations, and lighting changes so the model learns what a bear looks like from many angles, not specific photos.
4. **Trained the model:** fine-tuned a ResNet-18 pretrained on ImageNet using [fastai](https://docs.fast.ai/). Since the model already knows general visual features like edges, textures, and shapes, it only needed a few epochs and a small dataset to learn this task.
5. **Evaluated it:** checked accuracy on a held-out validation set (20% of the images) and used a confusion matrix to see exactly where the model made mistakes.
6. **Deployed it:** built a web interface with [Streamlit](https://streamlit.io/) and deployed it on Streamlit Community Cloud.

Training was done on a free GPU in a Kaggle notebook. When I retrain, the notebook automatically pushes the new model to this repo, and the live app updates.

---

## What I learned

### Models will confidently answer questions they shouldn't
My first version only knew three classes. When I uploaded a photo that wasn't a bear, it still predicted "teddy bear" with **96% confidence**. A classifier has to pick one of the classes it knows, and its probabilities always add up to 100%, so "96% teddy" really meant "out of these three options, teddy is the closest match."

**Fix:** I added a `not_bear` class trained on a wide variety of non-bear images, which gave the model a way to say "none of the above."

### The model may be using the background, not just the bear
After adding `not_bear`, the confusion matrix showed that **all remaining errors were non-bear images predicted as bears**:

- A **forest** photo was predicted as a grizzly or black bear. Most wild bear photos are taken in forests, so the model has likely learned "forest background → bear" as a shortcut.
- **Cartoon** images, including a cartoon mouse, were predicted.  A cartoon mouse has round ears, a small rounded body, and soft colors, all of which are features the model associates with teddy bears.