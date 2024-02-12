import torch
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)


class Multimodal:
    model_path = "LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384"
    device = "cuda"
    load_4bit, load_8bit = False, False
    conv_mode = "phi"

    def __init__(self):
        disable_torch_init()
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, processor, self.context_len = load_pretrained_model(
            self.model_path,
            None,
            self.model_name,
            self.load_8bit,
            self.load_4bit,
            device=self.device,
        )
        self.image_processor = processor["image"]

    def process(self, images, prompt, custom_params: dict = {}):
        conv = conv_templates[self.conv_mode].copy()
        image_tensor = self.image_processor.preprocess(images, return_tensors="pt")[
            "pixel_values"
        ].to(self.model.device, dtype=torch.float16)
        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                use_cache=False,
                stopping_criteria=[stopping_criteria],
                **(
                    dict(
                        temperature=0.2,
                        max_new_tokens=2048,
                    )
                    | custom_params
                ),
            )
        text = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        return dict(text=text)
