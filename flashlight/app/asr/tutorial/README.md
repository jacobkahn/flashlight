# Getting Started with Automatic Speech Recognition in Flashlight

This tutorial uses the following binaries with the following capabilities:
- [`fl_asr_tutorial_inference_ctc`](https://github.com/jacobkahn/flashlight/blob/tutorial_docs/flashlight/app/asr/tutorial/README.md#inference-with-an-existing-ctc-model): perform inference with an existing model with CTC loss [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/todo)
- [`fl_asr_tutorial_finetune_ctc`](https://github.com/jacobkahn/flashlight/blob/tutorial_docs/flashlight/app/asr/tutorial/README.md#finetuning-with-an-existing-ctc-model): finetune an existing CTC model with additional data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/todo)
- [`fl_asr_align`](https://github.com/facebookresearch/flashlight/blob/master/flashlight/app/asr/tools/alignment): force align audio and transcriptions using a CTC model
- [`fl_asr_voice_activity_detection_ctc`](https://github.com/facebookresearch/flashlight/blob/master/flashlight/app/asr/tools): [coming soon] detect speech and perform audio analysis

See the [full documentation](https://github.com/jacobkahn/flashlight/blob/tutorial_docs/flashlight/app/asr) for more general training or decoding instructions.

## Finetuning with Existing Model with Already-Labeled Audio

The outline below describes the end-to-end process of finetuning an existing acoustic model. In several steps:
1. *Preprocessing the audio.*
  a. Most [audio formats](http://libsndfile.github.io/libsndfile/formats.html) are supported and are automatically detected.
  b. All audio used in training or inference must have the same sample rate; up/downsampling audio may be necessary. We recommend 16 kHz.
2. *Force-aligning the labeled audio.*
  a. Using the existing transcriptions, generate alignments for audio. Using the `fl_asr_align` binary. See the [full alignment documentation](https://github.com/facebookresearch/flashlight/blob/master/flashlight/app/asr/tools/alignment).
  b. Based on the alignments, trim the existing audio to include sections containing speech. Doing so typically increases training speed.
3. Generate a final list file for training and validation sets using the trimmed audio and transcriptions. See the [list file documentation](https://github.com/jacobkahn/flashlight/blob/tutorial_docs/flashlight/app/asr/README.md#audio-and-transcriptions-data) for more details.
4. Use the `fl_asr_tutorial_finetune_ctc` binaruy to [finetune the existing model](https://github.com/jacobkahn/flashlight/blob/tutorial_docs/flashlight/app/asr/tutorial/README.md#finetuning-with-an-existing-ctc-model) (or [train your own](https://github.com/jacobkahn/flashlight/blob/tutorial_docs/flashlight/app/asr/README.md#how-to-train-acoustic-model) from scratch). List files can be passed to finetuning or inference binaries using the `train`/`valid` or `test` flags, respectively.

## Inference with an Existing CTC Model

See this colab notebook for a step-by-step tutorial.

The [`fl_asr_tutorial_inference_ctc`](https://github.com/facebookresearch/flashlight/blob/master/flashlight/app/asr/tutorial/InferenceCTC.cpp) binary provides a way to perform inference with CTC-trained acoustic models. To perform inference, you'll need the following components (with their corresponding `flags`):
- An acoustic model (AM) (`am_path`)
- A token set with which the AM was trained (`tokens_path`)
- A lexicon (`lexicon_path`)
- A language model for decoding (`lm_path`)

The following parameters are also configurable when performing inference:
- The sample rate of input audio (`sample_rate`)
- The beam size when decoding (`beam_size`)
- The beam size of the token beam when decoding (`beam_size_token`)
- The beam threshold (`beam_threshold`)
- The LM weight score for decoding (`lm_weight`)
- The word score for decoding (`word_score`).

See the [complete ASR app documentation](https://github.com/jacobkahn/flashlight/blob/tutorial_docs/flashlight/app/asr/README.md) for a more detailed explanation of each of these flags. See the aforementioned colab tutorial for sensible values used in a demo.

## Finetuning with an Existing CTC Model

See this colab notebook for a step-by-step tutorial.

The [`fl_asr_tutorial_finetune_ctc`](https://github.com/jacobkahn/flashlight/blob/tutorial_docs/flashlight/app/asr/tutorial/FinetuneCTC.cpp) binary provides a means of finetuning an existing trained acoustic model on additional labeled audio. Usage of the binary is as follows:
```
./fl_asr_tutorial_finetune_ctc [path to directory containing model] [...flags]
```
To finetune, you'll need the following components (with their corresponding `flags`):
- An acoustic model (AM) to finetune (the first argument to the binary invocation, e.g. `fl_asr_tutorial_finetune_ctc [path] [...flags]`)
- A token set with which the AM was trained (`tokens`)*
- A lexicon (`lexicon`)
- Validation sets to use for finetuning (`valid`)
- Train sets with data on which to finetune (`train`)
- Other training flags for flashlight training or audio processing as per the [ASR documentation](https://github.com/jacobkahn/flashlight/blob/tutorial_docs/flashlight/app/asr/README.md).

* Should be identical to that with which the original AM was trained. Will be provided with the AM in recipes/tutorials.

See the aforementioned colab tutorial for robust pre-trained models and their accompanying components that can be easily used for finetuning. The [wav2letter Robust ASR (RASR) recipe](https://github.com/facebookresearch/wav2letter/tree/master/recipes/rasr) contains robust pre-trained models and resources for finetuning.