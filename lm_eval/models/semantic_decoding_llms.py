from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval import utils
from lm_eval.models.utils import (
    Collator,
)

import os
from typing import List, Dict, Union, Optional, Tuple, Literal
from copy import deepcopy

from semantic_decoding.generators.semantic import SemanticGenerationConfig
from semantic_decoding.generators.generator import Generator
from transformers import GenerationConfig, AutoConfig
import torch
from tqdm import tqdm

eval_logger = utils.eval_logger

# for defaults
checkpoints = [
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-v0.3",
    "gpt2",
]

semantic_extractors = {
    "noun_chunks": "en_core_web_sm",
    "ner": "dslim/distilbert-NER"
}

@register_model("semantic_decoding_llm", "sem")
class SemanticDecodingModel(LM):

    _DEFAULT_MAX_LENGTH = 2048 # max lenght a model can work with

    def __init__(
        self,
        model_name: Union[int, str] = 0,
        batch_size: Union[int, str] = 1,
        max_length: Optional[int] = None, # this is about the max length a model can work with
        device: Optional[str] = None,
        truncation: bool = False,
        # for syntactic generation config
        max_new_tokens: int = 4,
        num_syntactic_beams: int = 20,
        num_return_sequences: Optional[int] = None,
        do_sample: bool = False,
        access_token: Optional[str] = None,
        no_repeat_ngram_size: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        length_penalty: Optional[float] = None,
        # for semantic generation config
        semantic_num_beams: int = 4,
        semantic_num_return_sequences: Optional[int] = None,
        sem_length_penalty: Optional[int] = None,
        max_overall_tokens: Optional[int] = None,
        max_overall_generated_tokens: Optional[int] = None,
        best_sequence_strategy: Literal["syntactic_sequence_score", "semantic_sequence_score"] = "syntactic_sequence_score",
        # use same setup but for regular decoding
        use_regular_decoding: bool = False,
        **kwargs
    ):
        assert isinstance(device, (str, type(None)))
        assert isinstance(model_name, (int, str))
        assert isinstance(batch_size, (int, str))
        semantic_generator_names = []
        for key, value in kwargs.items():
            if "semantic_generator" in key:
                semantic_generator_names.append(value)
        if len(semantic_generator_names) == 0:
            semantic_generator_names = [semantic_extractors["noun_chunks"]]
        if device is None:
            # detect device
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if isinstance(model_name, int):
            model_name = checkpoints[model_name]
        self.model_name = model_name
        generator = Generator(
            model_name,
            semantic_generator_names,
            device
        )
        self.truncation = truncation
        self.generator = generator
        self.tokenizer = generator.syntactic_generator.tokenizer
        self.model = generator.syntactic_generator.model
        # ? to have identical treatment for regular decoding, flip switch with this argument
        self.use_regular_decoding = use_regular_decoding
        if use_regular_decoding:
            self.generator = None
        self._model = self.model
        self._max_length = max_length # manually set max_length
        # since using internal device_map = "auto", no need for accelerate here
        self._rank = 0
        self._world_size = 1

        # syntactic generation config
        syntactic_config_args = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_syntactic_beams,
            "num_return_sequences": num_return_sequences if num_return_sequences is not None else num_syntactic_beams,
            "do_sample": do_sample,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "access_token": access_token,
        }
        # Filter out None values
        syntactic_config_args = {k: v for k, v in syntactic_config_args.items() if v is not None}
        self.syntactic_generation_config = GenerationConfig(**syntactic_config_args)
        if use_regular_decoding:
            self.syntactic_generation_config.output_scores = True
            self.syntactic_generation_config.return_dict_in_generate = True

        # semantic generation config
        semantic_generation_config = {
            "num_beams": semantic_num_beams,
            "num_return_sequences": semantic_num_return_sequences,
            "length_penalty": sem_length_penalty,
            "max_overall_tokens": max_overall_tokens,
            "max_overall_generated_tokens": max_overall_generated_tokens,
        }
        semantic_generation_config = {k: v for k, v in semantic_generation_config.items() if v is not None}
        self.semantic_generation_config = SemanticGenerationConfig(**semantic_generation_config)
        if use_regular_decoding:
            self.syntactic_generation_config.max_length = self.semantic_generation_config.max_overall_tokens
            self.syntactic_generation_config.max_new_tokens = self.semantic_generation_config.max_overall_generated_tokens
        self.best_sequence_strategy = best_sequence_strategy

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "This model does not support `loglikelihoods` as the decoding is integral part to the performance."
        )

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "This model does not support `loglikelihoods` as the decoding is integral part to the performance."
        )

    def tok_encode_decode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> List[str]:
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        
        encoded_input = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
        )
        
        if left_truncate_len:
            encoded_input["input_ids"] = encoded_input["input_ids"][:, -left_truncate_len:]
            encoded_input["attention_mask"] = encoded_input["attention_mask"][:, -left_truncate_len:]
        
        decoded_input = self.tokenizer.batch_decode(
            encoded_input["input_ids"],
            skip_special_tokens=True
        )

        self.tokenizer.padding_side = old_padding_side
        return decoded_input

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        res = []

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer(req[0]).input_ids
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm),
            desc="Running generate_until requests",
        )
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched()

        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)

            gen_kwargs = all_gen_kwargs[0] # we are only batching with size 1 anyways
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks) # max tokens to generate
            until = gen_kwargs.get("until", None)
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(
                    f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                )

            eos_string = self.tokenizer.decode(self.tokenizer.eos_token_id)
            if not until:
                until = [eos_string]
            else:
                until.append(eos_string)

            max_ctx_len = self.max_length - max_gen_toks

            contexts = self.tok_encode_decode(
                contexts,
                padding_side="left",
                left_truncate_len=max_ctx_len,
                truncation=self.truncation
            )

            if self.use_regular_decoding:
                # as done in semantic decoding
                model_inputs = self.tokenizer(
                    contexts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                results = self.model.generate(
                    **model_inputs,
                    generation_config=self.syntactic_generation_config
                )
            else:
                results = self.generator.generate(
                    contexts,
                    self.semantic_generation_config,
                    self.syntactic_generation_config
                )

            sequences = None
            if self.use_regular_decoding:
                sequences = results["sequences"]
            else:
                sequences = results["syntactic_sequences"]
            
            sequence = None
            if self.use_regular_decoding:
                sequence = sequences[0]
            else:
                if self.best_sequence_strategy == "syntactic_sequence_score":
                    syntactic_transition_scores = results["syntactic_transition_scores"]
                    syntactic_sequence_scores = torch.div(
                        syntactic_transition_scores.sum(-1),
                        torch.pow(
                            syntactic_transition_scores.shape[-1] - (syntactic_transition_scores >= 0).sum(-1), # length of generated sequences
                            self.syntactic_generation_config.length_penalty # length penalty
                        )
                    )

                    best_sequ_index = syntactic_sequence_scores.argmax().item()
                    sequence = sequences[best_sequ_index]
                elif self.best_sequence_strategy == "semantic_sequence_score":
                    semantic_sequences_scores = results["semantic_sequences_scores"]
                    best_sequ_index = semantic_sequences_scores.argmax().item()
                    # could be an alternative
                    # ! could also be done in the future
                    raise NotImplementedError("This feature is not fully implemented yet.")
            
            # decode the sequences
            entire_decoded_res = self.tokenizer.batch_decode(
                [sequence]
            )
            
            for context, decoded_res in zip(contexts, entire_decoded_res):
                # discard context + left-padding toks if using causal decoder-only LM
                generated_res = decoded_res.split(context)[1:]
                # in case the context was repeated (which would result in a list of generated res above, merge here)
                generated_res = context.join(generated_res)

                # check if until was reached prior
                for term in until:
                    if len(term) > 0:
                        generated_res = generated_res.split(term)[0]

                res.append(generated_res)
                pbar.update(1)
        
        res = re_ords.get_original(res)

        pbar.close()

        return res
            
    @property
    def max_gen_toks(self) -> int:
        return 256 # analogus to hf models (max toks produced for request)

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")
    
    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        """
        Set and get the appropriate chat template for the model.
        This method sets the tokenizer's chat_template and returns the template string for reproducibility.

        The template selection logic is adapted from the Transformers library's `apply_chat_template`
        method in the Tokenizer class. The original implementation can be found at:
        https://github.com/huggingface/transformers/blob/fc35907f95459d7a6c5281dfadd680b6f7b620e3/src/transformers/tokenization_utils_base.py#L1687

        This method ensures that the right template is chosen based on the following:
        0. If the model has no 'tokenizer' attribute: assumes that there is only a single possible chat template, handled on the model provider side internally. Returns the empty string.
        1. If the model's tokenizer has multiple templates:
            a. Use the specified template if it exists in the dictionary.
            b. Use the default template from the list if no specific template is provided.
            c. Raise an error if no default template exists and no specific template is provided.
        2. If the model's tokenizer has a single template or no template:
            a. Use the tokenizer's chat template if available.
            b. Fall back to the default chat template if no tokenizer chat template exists.

        Args:
            chat_template (Union[bool, str]): Specifies the chat template to use.
                - If False or None, no template is applied.
                - If True, the default or only available template is used.
                - If a string, the template with the matching name is used.

        Returns:
            Optional[str]: The selected chat template, or None if no template is applied.
        """
        if self.tokenizer is None:
            return ""

        if chat_template is False or chat_template is None:
            eval_logger.warning(
                "model.chat_template was called with the chat_template set to False or None. "
                "Therefore no chat template will be applied. Make sure this is an intended behavior."
            )
            return None

        # Convert boolean chat_template to None to ensure compatibility with the adapted logic
        if isinstance(chat_template, bool):
            chat_template = None
        using_default_template = False

        # First, handle the cases when the model has a dict of multiple templates
        template = self.tokenizer.chat_template or self.tokenizer.default_chat_template

        if isinstance(template, dict):
            using_default_dict = self.tokenizer.chat_template is None

            if chat_template is not None:
                if chat_template in template:
                    selected_template = template[chat_template]
                    if using_default_dict:
                        using_default_template = True
                else:
                    raise ValueError(
                        f"The specified chat template '{chat_template}' is not available. "
                        f"Available template names are {sorted(template.keys())}."
                    )
            else:
                # If user didn't pass a chat template, use the default template from the dict
                if "default" in template:
                    selected_template = template["default"]
                    using_default_template = True
                else:
                    raise ValueError(
                        "This model has multiple chat templates with no default specified! Please either pass a chat "
                        "template or the name of the template you wish to use to the `chat_template` argument. Available "
                        f"template names are {sorted(template.keys())}."
                    )

        # Cases when the model has a single template or no template
        else:
            # priority: `chat_template` argument > `tokenizer.chat_template` > `tokenizer.default_chat_template
            if isinstance(chat_template, str):
                eval_logger.warning(
                    "Chat template name provided, but the tokenizer's chat template is not a dictionary. "
                    "Using the tokenizer's chat template or the default template instead."
                )
            if self.tokenizer.chat_template is not None:
                selected_template = self.tokenizer.chat_template
            else:
                selected_template = self.tokenizer.default_chat_template
                using_default_template = True

        if using_default_template:
            eval_logger.warning(
                "No chat template is set for this tokenizer, falling back to a default class-level template. This is "
                "very error-prone, because models are often trained with templates different from the class default! "
                "Default chat templates are a legacy feature and will be removed in Transformers v4.43, at which "
                "point any code depending on them will stop working. We recommend setting a valid chat template before "
                "then to ensure that this model continues working without issues."
            )

        return selected_template

    def get_model_info(self) -> dict:
        """
        Method to get Hugging Face model information for experiment reproducibility.
        """

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return ""


        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
            "model_revision": self.revision,
        }
        return model_info
