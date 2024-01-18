# Faces - Face detection for Stable Diffusion training data processing

Transforms a batch of photos of the same person (even taken with other people) into cropped photos around their head. Useful to process all photos necessary to train a Stable Diffusion LoRA (or Dreambooth).

I also included a notebook to circumscribe rectangle pictures in a square so they can be uploaded on Insta without losing a part of the picture (at the time of writing, Insta requires square photos)